"""GPT4TS: One Fits All — repurposing a pretrained GPT-2 for time series.

Zhou et al., NeurIPS 2023 — https://github.com/DAMO-DI-ML/NLP4TS

Key idea: patch the input time series into tokens, project them to GPT-2's
embedding dimension, then pass through the frozen GPT-2 transformer blocks
(only LayerNorm layers are fine-tuned). A linear head maps the final hidden
states to the forecast horizon.

Requires: pip install transformers
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

try:
    from transformers import GPT2Config, GPT2Model
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class _GPT4TSNet(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        out_dim: int,
        patch_len: int,
        stride: int,
        gpt_layers: int,
        dropout: float,
        pretrained: bool,
    ) -> None:
        super().__init__()
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "GPT4TSForecaster requires the `transformers` package.\n"
                "Install it with:  pip install transformers"
            )

        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)
        self.patch_len = int(patch_len)
        self.stride = int(stride)

        # Number of patches that fit in seq_len
        self.patch_num = (int(seq_len) - int(patch_len)) // int(stride) + 1

        D_LLM = 768  # GPT-2 hidden size

        # Project flattened patch (patch_len × Dx features) to GPT-2 space
        self.in_layer = nn.Linear(int(patch_len) * int(input_dim), D_LLM)

        # Load GPT-2 backbone
        if pretrained:
            self.gpt2 = GPT2Model.from_pretrained("gpt2")
        else:
            cfg = GPT2Config(n_embd=D_LLM, n_layer=max(1, int(gpt_layers)), n_head=12)
            self.gpt2 = GPT2Model(cfg)

        # Truncate to the requested number of transformer blocks
        self.gpt2.h = self.gpt2.h[: int(gpt_layers)]

        # Freeze the entire backbone
        for param in self.gpt2.parameters():
            param.requires_grad = False
        # Unfreeze LayerNorm weights — the only adaptation that is trained
        for name, param in self.gpt2.named_parameters():
            if "ln_" in name:
                param.requires_grad = True

        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

        # Flatten all patch hidden states and project to pred_len × out_dim
        self.out_layer = nn.Linear(D_LLM * self.patch_num, int(pred_len) * int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, Dx]
        B, L, Dx = x.shape
        if L != self.seq_len:
            if L < self.seq_len:
                raise ValueError(f"Input length L={L} < seq_len={self.seq_len}")
            x = x[:, -self.seq_len :, :]

        # Per-variate instance normalisation
        means = x.mean(dim=1, keepdim=True).detach()          # [B, 1, Dx]
        x = x - means
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / std

        # Build patches: [B, patch_num, Dx, patch_len]
        xp = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        Bn, N, Dx2, P = xp.shape
        xp = xp.reshape(Bn, N, Dx2 * P)      # [B, N, Dx*patch_len]

        # Project to GPT-2 embedding space
        xp = self.in_layer(xp)               # [B, N, D_LLM]

        # GPT-2 forward — positional embeddings are added internally
        out = self.gpt2(inputs_embeds=xp).last_hidden_state   # [B, N, D_LLM]

        out = self.act(out)
        out = self.dropout(out)
        out = out.reshape(B, -1)              # [B, N * D_LLM]
        out = self.out_layer(out)             # [B, pred_len * out_dim]
        out = out.view(B, self.pred_len, self.out_dim)

        # Denormalise to pipeline's processed space
        out = out * std[:, :1, : self.out_dim] + means[:, :1, : self.out_dim]
        return out


@register("gpt4ts")
class GPT4TSForecaster(BaseForecaster):
    """GPT4TS: fine-tunes only GPT-2 LayerNorms for time-series forecasting (NeurIPS 2023).

    Requires the ``transformers`` library (``pip install transformers``).
    The GPT-2 weights are downloaded automatically from HuggingFace on first use.
    """

    name: str = "gpt4ts"

    # model hyper-parameters
    patch_len: int = 3
    stride: int = 1
    gpt_layers: int = 6
    dropout: float = 0.3
    pretrained: bool = True

    # training hyper-parameters
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    patience: int = 3
    max_train_batches: Optional[int] = None
    seed: int = 0

    def __init__(self) -> None:
        self._net: Optional[_GPT4TSNet] = None
        self.train_history: list[dict[str, Any]] = []

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "GPT4TSForecaster requires the `transformers` package.\n"
                "Install it with:  pip install transformers"
            )

        dev = device or torch.device("cpu")
        torch.manual_seed(int(self.seed))

        train_dl = bundle.dataloaders["train"]
        val_dl = bundle.dataloaders["val"]
        Dx = int(len(bundle.x_cols))
        Dy = int(len(bundle.y_cols))
        seq_len = int(bundle.raw.spec.seq_len)
        pred_len = int(bundle.raw.spec.pred_len)

        patch_len = int(self.patch_len)
        stride = int(self.stride)
        # Clamp patch_len so it fits in seq_len
        if patch_len > seq_len:
            patch_len = seq_len
            stride = seq_len

        net = _GPT4TSNet(
            seq_len=seq_len,
            pred_len=pred_len,
            input_dim=Dx,
            out_dim=Dy,
            patch_len=patch_len,
            stride=stride,
            gpt_layers=int(self.gpt_layers),
            dropout=float(self.dropout),
            pretrained=bool(self.pretrained),
        ).to(dev)

        # Only train non-frozen parameters
        trainable = [p for p in net.parameters() if p.requires_grad]
        opt = torch.optim.Adam(trainable, lr=float(self.lr), weight_decay=float(self.weight_decay))

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
            raise RuntimeError("GPT4TSForecaster must be fit() before predict_batch()")
        dev = device or next(self._net.parameters()).device
        self._net.to(dev).eval()
        with torch.no_grad():
            return self._net(batch["x"].to(dev))
