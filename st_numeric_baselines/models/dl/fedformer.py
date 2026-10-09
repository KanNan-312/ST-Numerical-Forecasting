from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import time
import math
import random
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
            x = torch.cat([front, x, end], dim=1)

        #[B, C, L]
        y = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return y


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = _MovingAvg(int(kernel_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend

class _DataEmbedding(nn.Module):
    """Value + time + learnable positional embedding."""

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
        # x: [B, L, D]
        if x.ndim != 3:
            raise ValueError(f"Expected [B,L,D], got {tuple(x.shape)}")
        B, L, D = x.shape
        if D != self.value_dim:
            raise ValueError(f"value_dim mismatch: expected {self.value_dim}, got {D}")
        if L > self.max_len:
            raise ValueError(f"L={L} exceeds max_len={self.max_len}")

        v = self.value_proj(x)

        if self.time_proj is not None:
            if x_mark is None:
                t = torch.zeros_like(v)
            else:
                if x_mark.ndim != 3 or x_mark.shape[1] != L:
                    raise ValueError(f"x_mark shape mismatch: {tuple(x_mark.shape)} vs L={L}")
                # tolerate extra/fewer time features
                if int(x_mark.shape[2]) != self.time_dim:
                    if int(x_mark.shape[2]) > self.time_dim:
                        x_mark = x_mark[:, :, : self.time_dim]
                    else:
                        pad = self.time_dim - int(x_mark.shape[2])
                        x_mark = torch.cat(
                            [x_mark, torch.zeros((B, L, pad), device=x_mark.device, dtype=x_mark.dtype)],
                            dim=2,
                        )
                t = self.time_proj(x_mark)
        else:
            t = 0.0

        p = self.pos_embed[:, :L, :]
        return self.dropout(v + t + p)


def _get_frequency_modes(seq_len: int, modes: int, method: str) -> List[int]:
    L = int(seq_len)
    if L <= 0:
        raise ValueError("seq_len must be positive")
    max_bins = L // 2 + 1
    K = min(int(modes), max_bins)
    if K <= 0:
        return [0]

    method = str(method).lower().strip()
    if method not in {"low", "random"}:
        raise ValueError(f"Unsupported mode_select_method={method!r}, expected 'low' or 'random'")

    if method == "low":
        idx = list(range(K))
    else:
        pool = list(range(max_bins))
        if len(pool) <= K:
            idx = pool
        else:
            idx = [0]
            remain = K - 1
            if remain > 0:
                cand = pool[1:]
                idx += random.sample(cand, k=min(remain, len(cand)))
        idx = sorted(set(idx))
        if len(idx) < K:
            for j in range(max_bins):
                if j not in idx:
                    idx.append(j)
                if len(idx) >= K:
                    break
        idx = sorted(idx[:K])

    return idx


def _compl_mul1d(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhem,heom->bhom", x, w)


class _FourierBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        seq_len: int,
        modes: int,
        mode_select_method: str = "low",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        self.head_dim = self.d_model // self.n_heads

        self.seq_len = int(seq_len)
        self.modes = int(modes)
        self.mode_select_method = str(mode_select_method)

        self.index = _get_frequency_modes(self.seq_len, self.modes, self.mode_select_method)
        self.n_modes = int(len(self.index))

        # complex weights: [H, E, E, M]
        self.weights = nn.Parameter(torch.randn(self.n_heads, self.head_dim, self.head_dim, self.n_modes, 2) * 0.02)

        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        if D != self.d_model:
            raise ValueError(f"d_model mismatch: expected {self.d_model}, got {D}")
        if L != self.seq_len:
            seq_len = int(L)
            index = _get_frequency_modes(seq_len, self.modes, self.mode_select_method)
        else:
            index = self.index

        # [B, L, H, E] -> [B, H, E, L]
        xh = x.view(B, L, self.n_heads, self.head_dim).permute(0, 2, 3, 1)

        xf = torch.fft.rfft(xh, dim=-1)  # [B, H, E, F] 
        Fbins = int(xf.shape[-1])
        index = [i for i in index if 0 <= int(i) < Fbins]
        if len(index) == 0:
            # fallback to DC
            index = [0]

        K = len(index)
        w = torch.view_as_complex(self.weights[:, :, :, :K, :])  # [H, E, E, K]
        x_sel = xf[:, :, :, index]  # [B, H, E, K]
        out_sel = _compl_mul1d(x_sel, w)  # [B, H, E, K]

        out_f = torch.zeros_like(xf)
        out_f[:, :, :, index] = out_sel

        yt = torch.fft.irfft(out_f, n=L, dim=-1)  # [B, H, E, L]
        y = yt.permute(0, 3, 1, 2).contiguous().view(B, L, D)
        return self.dropout(y)


class _FourierCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        modes: int,
        mode_select_method: str = "low",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        self.head_dim = self.d_model // self.n_heads

        self.seq_len_q = int(seq_len_q)
        self.seq_len_kv = int(seq_len_kv)
        self.modes = int(modes)
        self.mode_select_method = str(mode_select_method)
        base_len = min(self.seq_len_q, self.seq_len_kv)
        self.index = _get_frequency_modes(base_len, self.modes, self.mode_select_method)
        self.n_modes = int(len(self.index))

        self.weights = nn.Parameter(torch.randn(self.n_heads, self.head_dim, self.head_dim, self.n_modes, 2) * 0.02)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q: [B, Lq, D], k/v: [B, Lk, D]
        B, Lq, Dq = q.shape
        Bk, Lk, Dk = k.shape
        if Bk != B or Dq != self.d_model or Dk != self.d_model:
            raise ValueError("cross-attn shape mismatch")

        # reshape to heads: [B, H, E, L]
        qh = q.view(B, Lq, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
        kh = k.view(B, Lk, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
        vh = v.view(B, Lk, self.n_heads, self.head_dim).permute(0, 2, 3, 1)

        qf = torch.fft.rfft(qh, dim=-1)  # [B, H, E, Fq]
        kf = torch.fft.rfft(kh, dim=-1)  # [B, H, E, Fk]
        vf = torch.fft.rfft(vh, dim=-1)  # [B, H, E, Fk]

        Fq = int(qf.shape[-1])
        Fk = int(kf.shape[-1])

        # ensure indices are in range for both
        index = [i for i in self.index if 0 <= int(i) < Fq and 0 <= int(i) < Fk]
        if len(index) == 0:
            index = [0]
        K = len(index)

        q_sel = qf[:, :, :, index]  # [B, H, E, K]
        k_sel = kf[:, :, :, index]  # [B, H, E, K]
        v_sel = vf[:, :, :, index]  # [B, H, E, K]

        # correlation over channels -> [B, H, K]
        score = (q_sel * k_sel.conj()).sum(dim=2)  # [B, H, K] 
        att = torch.softmax(score.real / math.sqrt(float(self.head_dim)), dim=-1)  # [B, H, K]

        v_weighted = v_sel * att.unsqueeze(2)  # [B, H, E, K]

        w = torch.view_as_complex(self.weights[:, :, :, :K, :])  # [H, E, E, K]
        out_sel = _compl_mul1d(v_weighted, w)  # [B, H, E, K]

        out_f = torch.zeros((B, self.n_heads, self.head_dim, Fq), device=q.device, dtype=qf.dtype)
        out_f[:, :, :, index] = out_sel

        yt = torch.fft.irfft(out_f, n=Lq, dim=-1)  # [B, H, E, Lq]
        y = yt.permute(0, 3, 1, 2).contiguous().view(B, Lq, self.d_model)
        return self.dropout(y)

class _EncoderLayer(nn.Module):
    def __init__(self, *, d_model: int, n_heads: int, seq_len: int, modes: int, mode_select_method: str, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = _FourierBlock(
            d_model=int(d_model),
            n_heads=int(n_heads),
            seq_len=int(seq_len),
            modes=int(modes),
            mode_select_method=str(mode_select_method),
            dropout=float(dropout),
        )
        self.norm1 = nn.LayerNorm(int(d_model))
        self.dropout = nn.Dropout(float(dropout))
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(d_ff)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_ff), int(d_model)),
        )
        self.norm2 = nn.LayerNorm(int(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class _DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dec_len: int,
        enc_len: int,
        modes: int,
        mode_select_method: str,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = _FourierBlock(
            d_model=int(d_model),
            n_heads=int(n_heads),
            seq_len=int(dec_len),
            modes=int(modes),
            mode_select_method=str(mode_select_method),
            dropout=float(dropout),
        )
        self.cross_attn = _FourierCrossAttention(
            d_model=int(d_model),
            n_heads=int(n_heads),
            seq_len_q=int(dec_len),
            seq_len_kv=int(enc_len),
            modes=int(modes),
            mode_select_method=str(mode_select_method),
            dropout=float(dropout),
        )
        self.norm1 = nn.LayerNorm(int(d_model))
        self.norm2 = nn.LayerNorm(int(d_model))
        self.norm3 = nn.LayerNorm(int(d_model))
        self.dropout = nn.Dropout(float(dropout))
        self.ffn = nn.Sequential(
            nn.Linear(int(d_model), int(d_ff)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_ff), int(d_model)),
        )

    def forward(self, x: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        x = x + self.dropout(self.cross_attn(self.norm2(x), mem, mem))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x

class _FEDformerNet(nn.Module):
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
        modes: int,
        mode_select_method: str,
        e_layers: int,
        d_layers: int,
        n_heads: int,
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

        self.encoder = nn.ModuleList(
            [
                _EncoderLayer(
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    seq_len=self.seq_len,
                    modes=int(modes),
                    mode_select_method=str(mode_select_method),
                    d_ff=int(d_ff),
                    dropout=float(dropout),
                )
                for _ in range(int(e_layers))
            ]
        )

        dec_len = int(self.label_len + self.pred_len)
        self.decoder = nn.ModuleList(
            [
                _DecoderLayer(
                    d_model=self.d_model,
                    n_heads=int(n_heads),
                    dec_len=dec_len,
                    enc_len=self.seq_len,
                    modes=int(modes),
                    mode_select_method=str(mode_select_method),
                    d_ff=int(d_ff),
                    dropout=float(dropout),
                )
                for _ in range(int(d_layers))
            ]
        )

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

        # allow padded or longer sequences: take the last tokens
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
        seasonal_enc, _trend_enc = self.decomp(x_enc)
        enc = self.enc_embedding(seasonal_enc, x_mark_enc)
        for layer in self.encoder:
            enc = layer(enc)
        mem = enc  # [B, seq_len, d_model]

        B = int(x_enc.shape[0])
        Dy = int(self.dec_in)
        if self.label_len > 0:
            y_label = x_dec[:, : self.label_len, :]
            seasonal_label, trend_label = self.decomp(y_label)
            last_trend = trend_label[:, -1:, :]
        else:
            seasonal_label = torch.zeros((B, 0, Dy), device=x_enc.device, dtype=x_enc.dtype)
            trend_label = torch.zeros((B, 0, Dy), device=x_enc.device, dtype=x_enc.dtype)
            last_trend = torch.zeros((B, 1, Dy), device=x_enc.device, dtype=x_enc.dtype)

        zeros_future = torch.zeros((B, self.pred_len, Dy), device=x_enc.device, dtype=x_enc.dtype)
        seasonal_init = torch.cat([seasonal_label, zeros_future], dim=1)

        trend_future = last_trend.repeat(1, self.pred_len, 1)
        trend_init = torch.cat([trend_label, trend_future], dim=1)

        dec = self.dec_embedding(seasonal_init, x_mark_dec)
        for layer in self.decoder:
            dec = layer(dec, mem)

        seasonal_out = self.projection(dec)  # [B, dec_len, Dy]
        out = seasonal_out + trend_init
        return out[:, -self.pred_len :, :]

@register("fedformer")
class FEDformerForecaster(BaseForecaster):
    name: str = "fedformer"

    # model hyper-parameters
    kernel_size: int = 3
    modes: int = 16
    mode_select_method: str = "low"  # 'low' or 'random'
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
        self._net: Optional[_FEDformerNet] = None
        self.train_history: list[dict[str, Any]] = []

    @staticmethod
    def _make_decoder_input(y: torch.Tensor, *, label_len: int, pred_len: int) -> torch.Tensor:
        """Known label part + zeros for future."""
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
        random.seed(int(self.seed))

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

        net = _FEDformerNet(
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            enc_in=Dx,
            dec_in=Dy,
            out_dim=Dy,
            time_dim=time_dim,
            d_model=int(self.d_model),
            modes=int(self.modes),
            mode_select_method=str(self.mode_select_method),
            e_layers=int(self.e_layers),
            d_layers=int(self.d_layers),
            n_heads=int(self.n_heads),
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
        for _ep in epoch_bar:
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

            # validation MSE
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
                "epoch": int(_ep + 1),
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
            raise RuntimeError("FEDformerForecaster must be fit() before predict_batch()")

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
