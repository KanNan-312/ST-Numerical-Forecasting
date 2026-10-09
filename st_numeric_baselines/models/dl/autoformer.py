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


def _causal_mask(L: int, *, device: torch.device) -> torch.Tensor:
    L = int(L)
    return torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)


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
        self.moving_avg = _MovingAvg(int(kernel_size))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, C]
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class _DataEmbedding(nn.Module):
    def __init__(
        self,
        *,
        value_dim: int,
        time_dim: int,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.value_dim = int(value_dim)
        self.time_dim = int(time_dim)
        self.d_model = int(d_model)

        self.value_proj = nn.Linear(self.value_dim, self.d_model)
        self.time_proj = nn.Linear(self.time_dim, self.d_model) if self.time_dim > 0 else None

        self.max_len = int(max_len)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, self.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor]) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,L,D], got {tuple(x.shape)}")
        B, L, D = x.shape
        if D != self.value_dim:
            raise ValueError(f"value_dim mismatch: expected {self.value_dim}, got {D}")
        if L > self.max_len:
            raise ValueError(f"Sequence length L={L} exceeds max_len={self.max_len}")

        v = self.value_proj(x)

        if self.time_proj is not None:
            if x_mark is None:
                t = torch.zeros_like(v)
            else:
                if x_mark.ndim != 3:
                    raise ValueError(f"Expected x_mark [B,L,T], got {tuple(x_mark.shape)}")
                if x_mark.shape[1] != L:
                    raise ValueError(f"x_mark length mismatch: {x_mark.shape[1]} vs L={L}")

                # tolerate extra/fewer time features
                if int(x_mark.shape[2]) != self.time_dim:
                    if int(x_mark.shape[2]) > self.time_dim:
                        xm = x_mark[:, :, : self.time_dim]
                    else:
                        pad = self.time_dim - int(x_mark.shape[2])
                        xm = torch.cat(
                            [x_mark, torch.zeros((B, L, pad), device=x_mark.device, dtype=x_mark.dtype)],
                            dim=2,
                        )
                    x_mark = xm

                t = self.time_proj(x_mark)
        else:
            t = 0.0

        p = self.pos_embed[:, :L, :]
        out = v + t + p
        return self.dropout(out)


class _AutoformerNet(nn.Module):
    def __init__(
        self,
        *,
        seq_len: int,
        label_len: int,
        pred_len: int,
        enc_in: int,
        dec_in: int,
        out_dim: int,
        time_dim: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_layers: int,
        d_ff: int,
        dropout: float,
        kernel_size: int,
    ) -> None:
        super().__init__()

        self.seq_len = int(seq_len)
        self.label_len = int(label_len)
        self.pred_len = int(pred_len)
        self.enc_in = int(enc_in)
        self.dec_in = int(dec_in)
        self.out_dim = int(out_dim)
        self.time_dim = int(time_dim)
        self.d_model = int(d_model)

        self.decomp = _SeriesDecomp(int(kernel_size))

        self.enc_embedding = _DataEmbedding(
            value_dim=self.enc_in,
            time_dim=self.time_dim,
            d_model=self.d_model,
            max_len=max(512, self.seq_len + 16),
            dropout=float(dropout),
        )
        self.dec_embedding = _DataEmbedding(
            value_dim=self.dec_in,
            time_dim=self.time_dim,
            d_model=self.d_model,
            max_len=max(512, self.label_len + self.pred_len + 16),
            dropout=float(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(e_layers))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(d_layers))

        self.projection = nn.Linear(self.d_model, self.out_dim)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor],
        x_dec: torch.Tensor,
        x_mark_dec: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if x_enc.ndim != 3 or x_dec.ndim != 3:
            raise ValueError(f"Expected x_enc/x_dec [B,L,D], got {tuple(x_enc.shape)} / {tuple(x_dec.shape)}")

        # Allow padded or longer sequences: take the last tokens.
        if x_enc.shape[1] != self.seq_len:
            if x_enc.shape[1] < self.seq_len:
                raise ValueError(f"x_enc length {x_enc.shape[1]} < seq_len={self.seq_len}")
            x_enc = x_enc[:, -self.seq_len :, :]
            if x_mark_enc is not None:
                x_mark_enc = x_mark_enc[:, -self.seq_len :, :]

        dec_len = int(self.label_len + self.pred_len)
        if x_dec.shape[1] != dec_len:
            if x_dec.shape[1] < dec_len:
                raise ValueError(f"x_dec length {x_dec.shape[1]} < dec_len={dec_len}")
            x_dec = x_dec[:, -dec_len:, :]
            if x_mark_dec is not None:
                x_mark_dec = x_mark_dec[:, -dec_len:, :]

        # Encoder: decompose and encode seasonal component
        seasonal_enc, _trend_enc = self.decomp(x_enc)
        enc_in = self.enc_embedding(seasonal_enc, x_mark_enc)
        mem = self.encoder(enc_in)

        B = int(x_enc.shape[0])
        Dy = int(self.dec_in)

        # Decoder init built from label part only
        if self.label_len > 0:
            y_label = x_dec[:, : self.label_len, :]
            seasonal_label, trend_label = self.decomp(y_label)
            last_trend = trend_label[:, -1:, :]
        else:
            seasonal_label = torch.zeros((B, 0, Dy), device=x_enc.device, dtype=x_enc.dtype)
            trend_label = torch.zeros((B, 0, Dy), device=x_enc.device, dtype=x_enc.dtype)
            last_trend = torch.zeros((B, 1, Dy), device=x_enc.device, dtype=x_enc.dtype)

        zeros_future = torch.zeros((B, self.pred_len, Dy), device=x_enc.device, dtype=x_enc.dtype)
        seasonal_init = torch.cat([seasonal_label, zeros_future], dim=1)  # [B, dec_len, Dy]

        trend_future = last_trend.repeat(1, self.pred_len, 1)
        trend_init = torch.cat([trend_label, trend_future], dim=1)  # [B, dec_len, Dy]

        dec_in = self.dec_embedding(seasonal_init, x_mark_dec)
        tgt_mask = _causal_mask(dec_len, device=dec_in.device)
        dec_out = self.decoder(tgt=dec_in, memory=mem, tgt_mask=tgt_mask)

        seasonal_out = self.projection(dec_out)  # [B, dec_len, Dy]
        out = seasonal_out + trend_init
        return out[:, -self.pred_len :, :]


@register("autoformer")
class AutoformerForecaster(BaseForecaster):
    name: str = "autoformer"

    # model hyper-parameters
    kernel_size: int = 3
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 2
    d_layers: int = 1
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
        self._net: Optional[_AutoformerNet] = None
        self.train_history: list[dict[str, Any]] = []

    @staticmethod
    def _make_decoder_input(y: torch.Tensor, *, label_len: int, pred_len: int) -> torch.Tensor:
        B, L, Dy = y.shape
        dec_len = int(label_len + pred_len)
        if L < dec_len:
            raise ValueError(f"y length {L} < label_len+pred_len={dec_len}")
        y = y[:, :dec_len, :]
        dec = torch.zeros((B, dec_len, Dy), device=y.device, dtype=y.dtype)
        if label_len > 0:
            dec[:, :label_len, :] = y[:, :label_len, :]
        return dec

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        dev = device if device is not None else torch.device("cpu")
        torch.manual_seed(int(self.seed))

        train_dl = bundle.dataloaders["train"]
        val_dl = bundle.dataloaders["val"]

        Dx = int(len(bundle.x_cols))
        Dy = int(len(bundle.y_cols))
        seq_len = int(bundle.raw.spec.seq_len)
        label_len = int(bundle.raw.spec.label_len)
        pred_len = int(bundle.raw.spec.pred_len)

        time_dim = 2  # [year, month] in this refactor

        k = int(self.kernel_size)
        if k % 2 == 0:
            k += 1

        net = _AutoformerNet(
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            enc_in=Dx,
            dec_in=Dy,
            out_dim=Dy,
            time_dim=time_dim,
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            e_layers=int(self.e_layers),
            d_layers=int(self.d_layers),
            d_ff=int(self.d_ff),
            dropout=float(self.dropout),
            kernel_size=k,
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
                x_mark = batch.get("x_mark", None)
                if x_mark is not None:
                    x_mark = x_mark.to(dev)

                y_full = batch["y"].to(dev)
                y_mark = batch.get("y_mark", None)
                if y_mark is not None:
                    y_mark = y_mark.to(dev)

                dec_in = self._make_decoder_input(y_full, label_len=label_len, pred_len=pred_len)
                y_true = y_full[:, -pred_len:, :]

                y_pred = net(x, x_mark, dec_in, y_mark)
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
                    x_mark = batch.get("x_mark", None)
                    if x_mark is not None:
                        x_mark = x_mark.to(dev)

                    y_full = batch["y"].to(dev)
                    y_mark = batch.get("y_mark", None)
                    if y_mark is not None:
                        y_mark = y_mark.to(dev)

                    dec_in = self._make_decoder_input(y_full, label_len=label_len, pred_len=pred_len)
                    y_true = y_full[:, -pred_len:, :]

                    y_pred = net(x, x_mark, dec_in, y_mark)
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
            raise RuntimeError("AutoformerForecaster must be fit() before predict_batch()")

        dev = device if device is not None else next(self._net.parameters()).device
        self._net.to(dev)
        self._net.eval()

        label_len = int(bundle.raw.spec.label_len)
        pred_len = int(bundle.raw.spec.pred_len)

        x = batch["x"].to(dev)
        x_mark = batch.get("x_mark", None)
        if x_mark is not None:
            x_mark = x_mark.to(dev)

        y_full = batch["y"].to(dev)
        y_mark = batch.get("y_mark", None)
        if y_mark is not None:
            y_mark = y_mark.to(dev)

        dec_in = self._make_decoder_input(y_full, label_len=label_len, pred_len=pred_len)

        with torch.no_grad():
            y_hat = self._net(x, x_mark, dec_in, y_mark)
        return y_hat
