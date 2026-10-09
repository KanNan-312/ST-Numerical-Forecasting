from __future__ import annotations
from typing import Any, Dict, Optional

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.registry import register


def _make_patches(x: torch.Tensor, *, patch_len: int, stride: int) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected [B,L,D], got {tuple(x.shape)}")
    B, L, D = x.shape
    p = int(patch_len)
    s = int(stride)
    if p <= 0:
        raise ValueError("patch_len must be positive")
    if s <= 0:
        raise ValueError("stride must be positive")
    if p > L:
        # fall back to a single patch covering the whole window
        p = L
        s = L

    # unfold: [B, N, p, D]
    patches = x.unfold(dimension=1, size=p, step=s)
    if patches.ndim != 4:
        patches = patches.contiguous().view(B, -1, p, D)
    return patches


class _PatchTSTNet(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        out_dim: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        max_patches: int = 512,
    ) -> None:
        super().__init__()

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.d_model = int(d_model)

        # Patch embedding: flatten -> d_model
        self.patch_embed = nn.Linear(int(self.patch_len) * int(self.input_dim), int(self.d_model))

        # We pre-allocate positional embeddings up to max_patches and slice.
        self.max_patches = int(max_patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_patches, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(self.d_model),
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(e_layers))
        self.norm = nn.LayerNorm(int(self.d_model))

        self.dropout = nn.Dropout(float(dropout))

        # Prediction head: pooled token representation -> pred_len*out_dim
        self.head = nn.Linear(int(self.d_model), int(self.pred_len) * int(self.out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, Dx]
        if x.ndim != 3:
            raise ValueError(f"Expected [B,L,D], got {tuple(x.shape)}")

        B, L, Dx = x.shape
        if L != self.seq_len:
            if L < self.seq_len:
                raise ValueError(f"Input length L={L} < expected seq_len={self.seq_len}")
            x = x[:, -self.seq_len :, :]
            L = self.seq_len

        patches = _make_patches(x, patch_len=self.patch_len, stride=self.stride)  # [B, N, p, Dx]
        B, N, p, Dx = patches.shape

        if N > self.max_patches:
            raise ValueError(f"n_patches={N} exceeds max_patches={self.max_patches}. Increase max_patches.")

        tokens = patches.reshape(B, N, p * Dx)
        tokens = self.patch_embed(tokens)  # [B, N, d_model]
        tokens = tokens + self.pos_embed[:, :N, :]
        tokens = self.dropout(tokens)

        h = self.encoder(tokens)  # [B,N,d_model]
        h = self.norm(h)

        # simple pooling over patch tokens
        pooled = h.mean(dim=1)  # [B,d_model]
        out = self.head(pooled)  # [B, pred_len*out_dim]
        return out.view(B, self.pred_len, self.out_dim)


@register("patchtst")
class PatchTSTForecaster(BaseForecaster):
    name: str = "patchtst"

    # model hyper-parameters
    patch_len: int = 3
    stride: int = 1
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1

    # training hyper-parameters
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    patience: int = 5
    max_train_batches: Optional[int] = None
    seed: int = 0

    def __init__(self) -> None:
        self._net: Optional[_PatchTSTNet] = None
        self.train_history: list[dict[str, Any]] = []

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        dev = device if device is not None else torch.device("cpu")
        torch.manual_seed(int(self.seed))

        train_dl = bundle.dataloaders["train"]
        val_dl = bundle.dataloaders["val"]

        Dx = int(len(bundle.x_cols))
        Dy = int(len(bundle.y_cols))
        seq_len = int(bundle.raw.spec.seq_len)
        pred_len = int(bundle.raw.spec.pred_len)

        # ensure sane patch settings
        p = int(self.patch_len)
        s = int(self.stride)
        if p <= 0:
            p = max(1, seq_len)
        if s <= 0:
            s = 1
        if p > seq_len:
            p = seq_len
            s = seq_len

        # pick a conservative max_patches based on seq_len/stride
        n_patches = (seq_len - p) // s + 1
        max_patches = max(32, int(n_patches) + 8)

        net = _PatchTSTNet(
            seq_len=seq_len,
            pred_len=pred_len,
            input_dim=Dx,
            out_dim=Dy,
            patch_len=p,
            stride=s,
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            e_layers=int(self.e_layers),
            d_ff=int(self.d_ff),
            dropout=float(self.dropout),
            max_patches=max_patches,
        ).to(dev)

        opt = torch.optim.Adam(net.parameters(), lr=float(self.lr), weight_decay=float(self.weight_decay))

        best_val = math.inf
        best_state: Optional[Dict[str, torch.Tensor]] = None
        bad_epochs = 0

        self.train_history = []
        _max_bt = int(self.max_train_batches) if self.max_train_batches is not None else None
        _train_total = min(len(train_dl), _max_bt) if _max_bt is not None else len(train_dl)

        epoch_bar = tqdm(range(int(self.epochs)), desc=f"[{self.name}]", unit="ep")
        for ep in epoch_bar:
            t_ep0 = time.perf_counter()
            net.train()
            train_sse = 0.0
            train_n = 0
            train_bar = tqdm(train_dl, desc="  train", total=_train_total, leave=False, unit="bt")
            for bi, batch in enumerate(train_bar):
                if self.max_train_batches is not None and bi >= int(self.max_train_batches):
                    train_bar.close()
                    break

                x = batch["x"].to(dev)
                y_true = batch["y"][:, -pred_len:, :].to(dev)

                y_pred = net(x)
                loss = F.mse_loss(y_pred, y_true)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if float(self.grad_clip) > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(self.grad_clip))
                opt.step()

                train_sse += float(loss.detach().item()) * int(y_true.numel())
                train_n += int(y_true.numel())
                train_bar.set_postfix({"loss": f"{float(loss.detach().item()):.4g}"})

            # val MSE
            net.eval()
            sse = 0.0
            n = 0
            with torch.no_grad():
                val_bar = tqdm(val_dl, desc="  val  ", leave=False, unit="bt")
                for batch in val_bar:
                    x = batch["x"].to(dev)
                    y_true = batch["y"][:, -pred_len:, :].to(dev)
                    y_pred = net(x)
                    diff = (y_pred - y_true).float()
                    sse += float((diff * diff).sum().item())
                    n += int(diff.numel())
                    val_bar.set_postfix({"mse": f"{sse / max(n, 1):.4g}"})

            val_mse = sse / max(n, 1)

            train_mse = train_sse / max(train_n, 1)
            ep_time = time.perf_counter() - t_ep0
            rec = {
                "epoch": int(ep + 1),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
                "epoch_time_sec": float(ep_time),
            }
            self.train_history.append(rec)

            if val_mse < best_val - 1e-12:
                best_val = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.patience):
                    epoch_bar.close()
                    break
            epoch_bar.set_postfix({"train": f"{train_mse:.4g}", "val": f"{val_mse:.4g}", "best": f"{best_val:.4g}"})

        if best_state is not None:
            net.load_state_dict(best_state)

        self._net = net

    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("PatchTSTForecaster must be fit() before predict_batch()")

        dev = device if device is not None else next(self._net.parameters()).device
        self._net.to(dev)
        self._net.eval()

        x = batch["x"].to(dev)
        with torch.no_grad():
            y_hat = self._net(x)
        return y_hat
