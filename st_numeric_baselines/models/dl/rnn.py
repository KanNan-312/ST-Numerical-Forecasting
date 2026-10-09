from __future__ import annotations

from dataclasses import dataclass
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


def _gather_last_state(seq: torch.Tensor, x_mask: Optional[torch.Tensor]) -> torch.Tensor:
    B, L, H = seq.shape
    if x_mask is None:
        return seq[:, -1, :]

    pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)  # [B,L]
    masked_pos = pos * (x_mask > 0).to(pos.dtype)
    idx_last = masked_pos.max(dim=1).values.long()  # [B]
    return seq[torch.arange(B, device=seq.device), idx_last, :]


class _RNNNet(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        out_dim: int,
        pred_len: int,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)

        self.rnn = nn.RNN(
            input_size=int(input_dim),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
        )
        self.proj = nn.Linear(int(hidden_size), int(pred_len) * int(out_dim))

    def forward(self, x: torch.Tensor, *, x_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # x: [B,L,D]
        seq_out, _ = self.rnn(x)  # [B,L,H]
        last = _gather_last_state(seq_out, x_mask)  # [B,H]
        y = self.proj(last)  # [B, pred_len*out_dim]
        return y.view(x.shape[0], self.pred_len, self.out_dim)


@register("rnn")
class RNNForecaster(BaseForecaster):
    name: str = "rnn"

    # public hyper-parameters
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1

    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    patience: int = 3
    max_train_batches: Optional[int] = None
    seed: int = 0

    def __init__(self) -> None:
        self._net: Optional[_RNNNet] = None
        self.train_history: list[dict[str, Any]] = []

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        dev = device if device is not None else torch.device("cpu")

        torch.manual_seed(int(self.seed))

        train_dl = bundle.dataloaders["train"]
        val_dl = bundle.dataloaders["val"]

        Dx = int(len(bundle.x_cols))
        Dy = int(len(bundle.y_cols))
        pred_len = int(bundle.raw.spec.pred_len)

        net = _RNNNet(
            input_dim=Dx,
            hidden_size=int(self.hidden_size),
            num_layers=int(self.num_layers),
            dropout=float(self.dropout),
            out_dim=Dy,
            pred_len=pred_len,
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
                x_mask = batch.get("x_mask", None)
                if x_mask is not None:
                    x_mask = x_mask.to(dev)

                y_true = batch["y"][:, -pred_len:, :].to(dev)

                y_pred = net(x, x_mask=x_mask)
                loss = F.mse_loss(y_pred, y_true)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                if float(self.grad_clip) > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(self.grad_clip))
                opt.step()

                train_sse += float(loss.detach().item()) * int(y_true.numel())
                train_n += int(y_true.numel())
                train_bar.set_postfix({"loss": f"{float(loss.detach().item()):.4g}"})

            # val
            net.eval()
            sse = 0.0
            n = 0
            with torch.no_grad():
                val_bar = tqdm(val_dl, desc="  val  ", leave=False, unit="bt")
                for batch in val_bar:
                    x = batch["x"].to(dev)
                    x_mask = batch.get("x_mask", None)
                    if x_mask is not None:
                        x_mask = x_mask.to(dev)
                    y_true = batch["y"][:, -pred_len:, :].to(dev)
                    y_pred = net(x, x_mask=x_mask)
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
            raise RuntimeError("RNNForecaster must be fit() before predict_batch()")

        dev = device if device is not None else next(self._net.parameters()).device
        self._net.to(dev)
        self._net.eval()

        x = batch["x"].to(dev)
        x_mask = batch.get("x_mask", None)
        if x_mask is not None:
            x_mask = x_mask.to(dev)

        with torch.no_grad():
            y_hat = self._net(x, x_mask=x_mask)
        return y_hat
