"""iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.

Liu et al., ICLR 2024 — https://github.com/thuml/iTransformer

Key idea: instead of attending across *time steps* (standard transformer),
each input *variate* becomes a token whose embedding is its full L-step series.
Self-attention then captures cross-variate dependencies; a per-variate linear
head maps each token's representation to the pred_len future values.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.registry import register


class _iTransformerNet(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        out_dim: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)

        # Each variate's L-step history is one token
        self.embed = nn.Linear(int(seq_len), int(d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(e_layers))
        self.norm = nn.LayerNorm(int(d_model))

        # Per-variate forecast head
        self.head = nn.Linear(int(d_model), int(pred_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, Dx]
        B, L, Dx = x.shape
        if L != self.seq_len:
            if L < self.seq_len:
                raise ValueError(f"Input length L={L} < seq_len={self.seq_len}")
            x = x[:, -self.seq_len :, :]

        # Per-variate instance normalisation (handles distribution shift)
        means = x.mean(dim=1, keepdim=True).detach()          # [B, 1, Dx]
        x = x - means
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, Dx]
        x = x / std

        # Transpose: variates → tokens  [B, Dx, L]
        z = x.transpose(1, 2)
        # Project L → d_model: each variate gets one token embedding
        z = self.embed(z)          # [B, Dx, d_model]

        # Self-attention across variates (no positional embedding — order-free)
        z = self.encoder(z)        # [B, Dx, d_model]
        z = self.norm(z)

        # Per-variate forecast: d_model → pred_len
        y = self.head(z)           # [B, Dx, pred_len]
        y = y.transpose(1, 2)      # [B, pred_len, Dx]

        # Denormalise (bring back to pipeline's processed space)
        y = y * std + means        # broadcast [B, 1, Dx] over [B, H, Dx]

        # Target column is always index 0 in continuous_cols (schema guarantee)
        return y[:, :, : self.out_dim]   # [B, H, Dy]


@register("itransformer")
class iTransformerForecaster(BaseForecaster):
    """iTransformer: inverted attention over variate tokens (ICLR 2024)."""

    name: str = "itransformer"

    # model hyper-parameters
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
        self._net: Optional[_iTransformerNet] = None
        self.train_history: list[dict[str, Any]] = []

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        dev = device or torch.device("cpu")
        torch.manual_seed(int(self.seed))

        train_dl = bundle.dataloaders["train"]
        val_dl = bundle.dataloaders["val"]
        Dx = int(len(bundle.x_cols))
        Dy = int(len(bundle.y_cols))
        seq_len = int(bundle.raw.spec.seq_len)
        pred_len = int(bundle.raw.spec.pred_len)

        net = _iTransformerNet(
            seq_len=seq_len,
            pred_len=pred_len,
            input_dim=Dx,
            out_dim=Dy,
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            e_layers=int(self.e_layers),
            d_ff=int(self.d_ff),
            dropout=float(self.dropout),
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
            t0 = time.perf_counter()
            net.train()
            tr_sse, tr_n = 0.0, 0
            tr_bar = tqdm(train_dl, desc="  train", total=_train_total, leave=False, unit="bt")
            for bi, batch in enumerate(tr_bar):
                if _max_bt is not None and bi >= _max_bt:
                    tr_bar.close()
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
                tr_sse += float(loss.detach()) * int(y_true.numel())
                tr_n += int(y_true.numel())
                tr_bar.set_postfix({"loss": f"{float(loss.detach()):.4g}"})

            net.eval()
            v_sse, v_n = 0.0, 0
            with torch.no_grad():
                v_bar = tqdm(val_dl, desc="  val  ", leave=False, unit="bt")
                for batch in v_bar:
                    x = batch["x"].to(dev)
                    y_true = batch["y"][:, -pred_len:, :].to(dev)
                    y_pred = net(x)
                    diff = (y_pred - y_true).float()
                    v_sse += float((diff * diff).sum())
                    v_n += int(diff.numel())
                    v_bar.set_postfix({"mse": f"{v_sse / max(v_n, 1):.4g}"})

            tr_mse = tr_sse / max(tr_n, 1)
            val_mse = v_sse / max(v_n, 1) if v_n > 0 else float("inf")
            self.train_history.append({
                "epoch": ep + 1, "train_mse": tr_mse, "val_mse": val_mse,
                "epoch_time_sec": time.perf_counter() - t0,
            })

            if val_mse < best_val - 1e-12:
                best_val = val_mse
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= int(self.patience):
                    epoch_bar.close()
                    break
            epoch_bar.set_postfix({"train": f"{tr_mse:.4g}", "val": f"{val_mse:.4g}", "best": f"{best_val:.4g}"})

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
            raise RuntimeError("iTransformerForecaster must be fit() before predict_batch()")
        dev = device or next(self._net.parameters()).device
        self._net.to(dev).eval()
        with torch.no_grad():
            return self._net(batch["x"].to(dev))
