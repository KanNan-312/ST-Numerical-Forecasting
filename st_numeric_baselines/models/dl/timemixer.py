from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
            x_pad = torch.cat([front, x, end], dim=1)
        else:
            x_pad = x

        # AvgPool1d expects [B, C, L]
        y = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return y


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, C]
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class _ScaleMixer(nn.Module):

    def __init__(
        self,
        *,
        seq_len: int,
        d_model: int,
        d_ff: int,
        dropout: float,
        act: str = "gelu",
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)

        if act.lower() == "relu":
            activation: nn.Module = nn.ReLU()
        else:
            activation = nn.GELU()

        self.norm_time = nn.LayerNorm(int(d_model))
        self.token_mlp = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len),
            activation,
            nn.Dropout(float(dropout)),
            nn.Linear(self.seq_len, self.seq_len),
            nn.Dropout(float(dropout)),
        )

        self.norm_chan = nn.LayerNorm(int(d_model))
        self.channel_mlp = nn.Sequential(
            nn.Linear(int(d_model), int(d_ff)),
            activation,
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_ff), int(d_model)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C] with L == self.seq_len
        if x.shape[1] != self.seq_len:
            raise ValueError(f"ScaleMixer expects L={self.seq_len}, got L={x.shape[1]}")

        # token mixing 
        y = self.norm_time(x)
        y = y.transpose(1, 2)  # [B,C,L]
        y = self.token_mlp(y)
        y = y.transpose(1, 2)  # [B,L,C]
        x = x + y

        # channel mixing (along features)
        y = self.norm_chan(x)
        y = self.channel_mlp(y)
        x = x + y
        return x


class _MultiScaleMixerBlock(nn.Module):
    def __init__(
        self,
        *,
        base_len: int,
        d_model: int,
        d_ff: int,
        dropout: float,
        scales: Sequence[int],
        act: str = "gelu",
    ) -> None:
        super().__init__()
        self.base_len = int(base_len)

        # keep only valid scales
        uniq: List[int] = []
        for s in scales:
            s = int(s)
            if s <= 0:
                continue
            if s > self.base_len:
                continue
            if s not in uniq:
                uniq.append(s)
        if 1 not in uniq:
            uniq = [1] + uniq

        self.scales = tuple(uniq)

        mixers: List[nn.Module] = []
        pools: List[Optional[nn.Module]] = []

        for s in self.scales:
            if s == 1:
                Ls = self.base_len
                pool = None
            else:
                # AvgPool1d with kernel=stride=s gives output length floor((L-s)/s + 1) == L//s
                Ls = self.base_len // s
                pool = nn.AvgPool1d(kernel_size=s, stride=s, padding=0)
                if Ls <= 0:
                    continue

            mixers.append(_ScaleMixer(seq_len=Ls, d_model=d_model, d_ff=d_ff, dropout=dropout, act=act))
            pools.append(pool)

        self.mixers = nn.ModuleList(mixers)
        self.pools = pools

        self.norm_out = nn.LayerNorm(int(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L0, C]
        B, L0, C = x.shape
        if L0 != self.base_len:
            if L0 < self.base_len:
                raise ValueError(f"Expected L >= {self.base_len}, got {L0}")
            x = x[:, -self.base_len :, :]
            L0 = self.base_len

        outs: List[torch.Tensor] = []

        for s, mix, pool in zip(self.scales, self.mixers, self.pools):
            if s == 1:
                xs = x
            else:
                xs = x.transpose(1, 2)  # [B,C,L]
                xs = pool(xs)  # [B,C,Ls]
                xs = xs.transpose(1, 2)  # [B,Ls,C]

            xs = mix(xs)

            # upsample to base length
            if xs.shape[1] != L0:
                xu = xs.transpose(1, 2)  # [B,C,Ls]
                xu = F.interpolate(xu, size=L0, mode="linear", align_corners=False)
                xu = xu.transpose(1, 2)
            else:
                xu = xs
            outs.append(xu)

        y = torch.stack(outs, dim=0).mean(dim=0)  # [B,L0,C]
        y = self.norm_out(y)
        return y


class _TimeMixerNet(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        pred_len: int,
        input_dim: int,
        out_dim: int,
        d_model: int,
        d_ff: int,
        n_blocks: int,
        scales: Sequence[int],
        dropout: float,
        kernel_size: int,
        act: str = "gelu",
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.d_model = int(d_model)

        # embedding projection
        self.in_proj = nn.Identity() if self.d_model == self.input_dim else nn.Linear(self.input_dim, self.d_model)

        self.decomp = _SeriesDecomp(int(kernel_size))

        # separate stacks for seasonal
        self.blocks_s = nn.ModuleList(
            [
                _MultiScaleMixerBlock(
                    base_len=self.seq_len,
                    d_model=self.d_model,
                    d_ff=int(d_ff),
                    dropout=float(dropout),
                    scales=scales,
                    act=act,
                )
                for _ in range(int(n_blocks))
            ]
        )
        self.blocks_t = nn.ModuleList(
            [
                _MultiScaleMixerBlock(
                    base_len=self.seq_len,
                    d_model=self.d_model,
                    d_ff=int(d_ff),
                    dropout=float(dropout),
                    scales=scales,
                    act=act,
                )
                for _ in range(int(n_blocks))
            ]
        )

        self.norm = nn.LayerNorm(self.d_model)

        # time projection [L] -> [H] applied per-channel on [B,C,L]
        self.time_proj = nn.Linear(self.seq_len, self.pred_len)

        # output projection to Dy
        self.out_proj = nn.Identity() if self.out_dim == self.d_model else nn.Linear(self.d_model, self.out_dim)

    def _forward_comp(self, x: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        # x: [B,L,C]
        for blk in blocks:
            x = blk(x)
        x = self.norm(x)

        # [B,L,C] -> [B,C,L] -> [B,C,H] -> [B,H,C]
        y = self.time_proj(x.transpose(1, 2)).transpose(1, 2)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,Dx]
        if x.ndim != 3:
            raise ValueError(f"Expected [B,L,D], got {tuple(x.shape)}")
        B, L, Dx = x.shape
        if L != self.seq_len:
            if L < self.seq_len:
                raise ValueError(f"Input length L={L} < expected seq_len={self.seq_len}")
            x = x[:, -self.seq_len :, :]

        x = self.in_proj(x)
        seasonal, trend = self.decomp(x)

        y_s = self._forward_comp(seasonal, self.blocks_s)
        y_t = self._forward_comp(trend, self.blocks_t)
        y = y_s + y_t
        y = self.out_proj(y)
        return y


def _parse_scales(scales: object) -> Tuple[int, ...]:
    if isinstance(scales, str):
        parts = [p.strip() for p in scales.split(",") if p.strip()]
        out: List[int] = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                continue
        return tuple(out)
    if isinstance(scales, (list, tuple)):
        out = []
        for s in scales:
            try:
                out.append(int(s))
            except Exception:
                continue
        return tuple(out)
    return (1, 2, 4)


@register("timemixer")
class TimeMixerForecaster(BaseForecaster):
    name: str = "timemixer"

    # model hyper-parameters
    d_model: int = 64
    d_ff: int = 256
    n_blocks: int = 2
    scales: Tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1
    kernel_size: int = 3
    act: str = "gelu"

    # training hyper-parameters
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    patience: int = 5
    max_train_batches: Optional[int] = None
    seed: int = 0

    def __init__(self) -> None:
        self._net: Optional[_TimeMixerNet] = None
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

        scales = _parse_scales(self.scales)

        net = _TimeMixerNet(
            seq_len=seq_len,
            pred_len=pred_len,
            input_dim=Dx,
            out_dim=Dy,
            d_model=int(self.d_model),
            d_ff=int(self.d_ff),
            n_blocks=int(self.n_blocks),
            scales=scales,
            dropout=float(self.dropout),
            kernel_size=k,
            act=str(self.act),
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

            # validation MSE in processed space
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
            raise RuntimeError("TimeMixerForecaster must be fit() before predict_batch()")

        dev = device if device is not None else next(self._net.parameters()).device
        self._net.to(dev)
        self._net.eval()

        x = batch["x"].to(dev)
        with torch.no_grad():
            y_hat = self._net(x)
        return y_hat
