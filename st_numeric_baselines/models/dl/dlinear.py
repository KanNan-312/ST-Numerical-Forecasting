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


class _MovingAvg(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        k = int(kernel_size)
        if k <= 0:
            raise ValueError("kernel_size must be positive")
        if k % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = k
        self.avg = nn.AvgPool1d(kernel_size=k, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        B, L, C = x.shape
        pad = (self.kernel_size - 1) // 2
        if pad > 0:
            front = x[:, 0:1, :].repeat(1, pad, 1)
            end = x[:, -1:, :].repeat(1, pad, 1)
            x_pad = torch.cat([front, x, end], dim=1)  # [B, L+2*pad, C]
        else:
            x_pad = x

        # [B, C, L]
        y = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return y  # [B, L, C]


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, C]
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class _DLinearNet(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        out_dim: int,
        kernel_size: int,
        individual: bool,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.individual = bool(individual)

        self.decomp = _SeriesDecomp(int(kernel_size))

        if self.individual:
            # per-channel linear mapping along time: [L] -> [H]
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.input_dim)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.input_dim)]
            )
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

        self.proj_out = nn.Identity() if self.out_dim == self.input_dim else nn.Linear(self.input_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, Dx]
        if x.ndim != 3:
            raise ValueError(f"Expected [B,L,D], got {tuple(x.shape)}")
        B, L, D = x.shape
        if L != self.seq_len:
            if L < self.seq_len:
                raise ValueError(f"Input length L={L} < expected seq_len={self.seq_len}")
            x = x[:, -self.seq_len :, :]
            L = self.seq_len

        seasonal, trend = self.decomp(x)  # [B,L,D] each

        # [B, D, L]
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            s_out = torch.zeros((B, D, self.pred_len), device=x.device, dtype=x.dtype)
            t_out = torch.zeros((B, D, self.pred_len), device=x.device, dtype=x.dtype)
            for i in range(D):
                s_out[:, i, :] = self.linear_seasonal[i](seasonal[:, i, :])
                t_out[:, i, :] = self.linear_trend[i](trend[:, i, :])
        else:
            # shared linear operates on last dim
            s_out = self.linear_seasonal(seasonal)  # [B,D,H]
            t_out = self.linear_trend(trend)  # [B,D,H]

        out = (s_out + t_out).permute(0, 2, 1)  # [B,H,D]
        out = self.proj_out(out)  # [B,H,Dy]
        return out


@register("dlinear")
class DLinearForecaster(BaseForecaster):

    name: str = "dlinear"

    # model hyper-parameters
    kernel_size: int = 3
    individual: bool = True

    # training hyper-parameters
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    patience: int = 5
    max_train_batches: Optional[int] = None
    seed: int = 0

    def __init__(self) -> None:
        self._net: Optional[_DLinearNet] = None
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

        # ensure odd kernel size
        k = int(self.kernel_size)
        if k % 2 == 0:
            k += 1

        net = _DLinearNet(
            seq_len=seq_len,
            pred_len=pred_len,
            input_dim=Dx,
            out_dim=Dy,
            kernel_size=k,
            individual=bool(self.individual),
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

            # validation MSE (processed space)
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
            raise RuntimeError("DLinearForecaster must be fit() before predict_batch()")

        dev = device if device is not None else next(self._net.parameters()).device
        self._net.to(dev)
        self._net.eval()

        x = batch["x"].to(dev)
        with torch.no_grad():
            y_hat = self._net(x)
        return y_hat
