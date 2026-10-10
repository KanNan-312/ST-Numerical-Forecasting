"""GMAN — Graph Multi-Attention Network (Zheng et al., AAAI 2020).

Reference: https://github.com/zhengchuanpan/GMAN  (``METR/model.py``, TensorFlow)

Port of the reference architecture: an encoder-decoder built from ST-Attention
blocks (spatial attention over nodes + causally-masked temporal attention over
time, merged by a gated fusion, with a residual connection), bridged by a
*transform attention* that maps the P encoder steps directly onto the Q future
steps (no autoregressive decoding). Each attention input is the hidden state
concatenated with a spatio-temporal embedding (STE = spatial + temporal).

Deviations from the reference, all forced by this benchmark's data contract:

* **Spatial embedding.** The reference feeds in fixed node2vec vectors
  precomputed on the road graph (``SE``). Here it is a learned per-node
  embedding passed through the same 2-layer FC, so the model needs no
  adjacency (``requires_graph = False``) -- same convention as ``staeformer``
  / ``stid`` / ``agcrn``.
* **Temporal embedding.** The reference one-hot encodes (day-of-week,
  time-of-day) of all P+Q steps. Monthly housing/crime data has neither, so
  by default (``use_tod_dow=False``) a learned per-position embedding over the
  P+Q steps is used instead. With ``use_tod_dow=True`` (5-min traffic data)
  the reference's one-hot (dow, tod) scheme is used; ``GraphWindowDataset``
  only supplies marks for the P encoder steps, so the Q future marks are
  extrapolated from the last encoder mark at ``1 / steps_per_day`` per step.
* **Loss.** The reference's masked MAE is replaced by the benchmark's standard
  MSE loss, as for every other GNN here. Multi-feature input / multi-channel
  output are supported (reference: 1 in, 1 out).
* BatchNorm momentum is PyTorch's fixed 0.1 rather than the reference's
  decaying ``bn_decay``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from st_numeric_baselines.models.gnn.gnn_forecaster import GNNForecasterBase
from st_numeric_baselines.models.registry import register


class _FC(nn.Module):
    """Stack of 1x1 convs (= Linear on the last axis), each with optional BN,
    mirroring the reference ``FC`` (linear -> BN -> activation)."""

    def __init__(
        self,
        in_dim: int,
        units: Sequence[int],
        activations: Sequence[Optional[str]],
        *,
        bn: bool = True,
        use_bias: bool = True,
        drop: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.drop = drop
        self.linears = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts: List[Optional[str]] = list(activations)
        d = in_dim
        for u in units:
            self.linears.append(nn.Linear(d, int(u), bias=use_bias))
            self.bns.append(nn.BatchNorm1d(int(u)) if bn else nn.Identity())
            d = int(u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin, bn, act in zip(self.linears, self.bns, self.acts):
            if self.drop:
                x = F.dropout(x, p=self.drop, training=self.training)
            x = lin(x)
            if isinstance(bn, nn.BatchNorm1d):
                shp = x.shape
                x = bn(x.reshape(-1, shp[-1])).reshape(shp)
            if act == "relu":
                x = F.relu(x)
        return x


class _STEmbedding(nn.Module):
    """Spatio-temporal embedding -> [B, P+Q, N, D]."""

    def __init__(
        self, n_nodes: int, n_steps: int, D: int, *, bn: bool,
        use_tod_dow: bool, steps_per_day: int,
    ) -> None:
        super().__init__()
        self.use_tod_dow = bool(use_tod_dow)
        self.steps_per_day = int(steps_per_day)
        self.n_steps = int(n_steps)
        self.node_emb = nn.Parameter(torch.randn(n_nodes, D) * 0.1)
        self.fc_s = _FC(D, [D, D], ["relu", None], bn=bn)
        t_in = (7 + self.steps_per_day) if self.use_tod_dow else n_steps
        self.fc_t = _FC(t_in, [D, D], ["relu", None], bn=bn)

    def _tod_dow_onehot(self, x_mark: torch.Tensor, P: int, Q: int) -> torch.Tensor:
        # x_mark: [B, P, 2] = (tod_frac, dow); extrapolate Q future steps.
        spd = self.steps_per_day
        tod0 = torch.round(x_mark[:, -1, 0] * spd).long()          # [B]
        dow0 = x_mark[:, -1, 1].round().long()                     # [B]
        k = torch.arange(1, Q + 1, device=x_mark.device)           # [Q]
        tod_f = tod0[:, None] + k[None]                            # [B, Q]
        dow_f = (dow0[:, None] + tod_f // spd) % 7
        tod_f = tod_f % spd
        tod_p = torch.round(x_mark[..., 0] * spd).long() % spd
        dow_p = x_mark[..., 1].round().long().clamp(0, 6)
        tod = torch.cat([tod_p, tod_f], dim=1)                     # [B, P+Q]
        dow = torch.cat([dow_p, dow_f], dim=1)
        return torch.cat([F.one_hot(dow, 7), F.one_hot(tod, spd)], dim=-1).float()

    def forward(self, B: int, P: int, Q: int, x_mark: Optional[torch.Tensor], device) -> torch.Tensor:
        se = self.fc_s(self.node_emb)[None, None]                  # [1, 1, N, D]
        if self.use_tod_dow:
            if x_mark is None:
                raise ValueError("use_tod_dow=True requires x_mark [B,L,2] (tod_frac, dow)")
            te_in = self._tod_dow_onehot(x_mark, P, Q)             # [B, P+Q, 7+spd]
        else:
            te_in = torch.eye(P + Q, device=device)[None].expand(B, -1, -1)
        te = self.fc_t(te_in)[:, :, None, :]                       # [B, P+Q, 1, D]
        return se + te


def _split_heads(x: torch.Tensor, K: int) -> torch.Tensor:
    # [B, T, N, K*d] -> [B, K, T, N, d]
    B, T, N, D = x.shape
    return x.view(B, T, N, K, D // K).permute(0, 3, 1, 2, 4)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    # [B, K, T, N, d] -> [B, T, N, K*d]
    B, K, T, N, d = x.shape
    return x.permute(0, 2, 3, 1, 4).reshape(B, T, N, K * d)


class _SpatialAttention(nn.Module):
    def __init__(self, D: int, K: int, bn: bool) -> None:
        super().__init__()
        self.K = K
        self.q = _FC(2 * D, [D], ["relu"], bn=bn)
        self.k = _FC(2 * D, [D], ["relu"], bn=bn)
        self.v = _FC(2 * D, [D], ["relu"], bn=bn)
        self.out = _FC(D, [D, D], ["relu", None], bn=bn)

    def forward(self, X: torch.Tensor, STE: torch.Tensor) -> torch.Tensor:
        X = torch.cat([X, STE], dim=-1)
        q, k, v = (_split_heads(f(X), self.K) for f in (self.q, self.k, self.v))  # [B,K,T,N,d]
        att = torch.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)  # [B,K,T,N,N]
        return self.out(_merge_heads(att @ v))


class _TemporalAttention(nn.Module):
    def __init__(self, D: int, K: int, bn: bool, mask: bool = True) -> None:
        super().__init__()
        self.K, self.mask = K, mask
        self.q = _FC(2 * D, [D], ["relu"], bn=bn)
        self.k = _FC(2 * D, [D], ["relu"], bn=bn)
        self.v = _FC(2 * D, [D], ["relu"], bn=bn)
        self.out = _FC(D, [D, D], ["relu", None], bn=bn)

    def forward(self, X: torch.Tensor, STE: torch.Tensor) -> torch.Tensor:
        X = torch.cat([X, STE], dim=-1)
        q, k, v = (_split_heads(f(X), self.K).transpose(2, 3) for f in (self.q, self.k, self.v))  # [B,K,N,T,d]
        att = q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5)       # [B,K,N,T,T]
        if self.mask:
            T = att.shape[-1]
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=att.device))
            att = att.masked_fill(~causal, -(2 ** 15) + 1)
        att = torch.softmax(att, dim=-1)
        return self.out(_merge_heads((att @ v).transpose(2, 3)))


class _GatedFusion(nn.Module):
    def __init__(self, D: int, bn: bool) -> None:
        super().__init__()
        self.fs = _FC(D, [D], [None], bn=bn, use_bias=False)
        self.ft = _FC(D, [D], [None], bn=bn, use_bias=True)
        self.out = _FC(D, [D, D], ["relu", None], bn=bn)

    def forward(self, HS: torch.Tensor, HT: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.fs(HS) + self.ft(HT))
        return self.out(z * HS + (1.0 - z) * HT)


class _STAttBlock(nn.Module):
    def __init__(self, D: int, K: int, bn: bool) -> None:
        super().__init__()
        self.sa = _SpatialAttention(D, K, bn)
        self.ta = _TemporalAttention(D, K, bn)
        self.fusion = _GatedFusion(D, bn)

    def forward(self, X: torch.Tensor, STE: torch.Tensor) -> torch.Tensor:
        return X + self.fusion(self.sa(X, STE), self.ta(X, STE))


class _TransformAttention(nn.Module):
    def __init__(self, D: int, K: int, bn: bool) -> None:
        super().__init__()
        self.K = K
        self.q = _FC(D, [D], ["relu"], bn=bn)
        self.k = _FC(D, [D], ["relu"], bn=bn)
        self.v = _FC(D, [D], ["relu"], bn=bn)
        self.out = _FC(D, [D, D], ["relu", None], bn=bn)

    def forward(self, X: torch.Tensor, STE_P: torch.Tensor, STE_Q: torch.Tensor) -> torch.Tensor:
        q = _split_heads(self.q(STE_Q), self.K).transpose(2, 3)   # [B,K,N,Q,d]
        k = _split_heads(self.k(STE_P), self.K).transpose(2, 3)   # [B,K,N,P,d]
        v = _split_heads(self.v(X), self.K).transpose(2, 3)       # [B,K,N,P,d]
        att = torch.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)  # [B,K,N,Q,P]
        return self.out(_merge_heads((att @ v).transpose(2, 3)))  # [B,Q,N,D]


class GMANNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        seq_len: int,
        *,
        n_heads: int = 4,
        head_dim: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        bn: bool = True,
        use_tod_dow: bool = False,
        steps_per_day: int = 288,
    ) -> None:
        super().__init__()
        self.P, self.Q = int(seq_len), int(pred_len)
        D = int(n_heads) * int(head_dim)
        self.in_fc = _FC(int(input_dim), [D, D], ["relu", None], bn=bn)
        self.ste = _STEmbedding(
            n_nodes, self.P + self.Q, D, bn=bn, use_tod_dow=use_tod_dow, steps_per_day=steps_per_day,
        )
        self.encoder = nn.ModuleList([_STAttBlock(D, n_heads, bn) for _ in range(int(n_layers))])
        self.trans = _TransformAttention(D, n_heads, bn)
        self.decoder = nn.ModuleList([_STAttBlock(D, n_heads, bn) for _ in range(int(n_layers))])
        self.out_fc = _FC(D, [D, int(out_dim)], ["relu", None], bn=bn, drop=float(dropout) or None)

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, P, N, Dx] -> [B, Q, N, Dy]
        B = x.shape[0]
        X = self.in_fc(x)
        STE = self.ste(B, self.P, self.Q, x_mark, x.device)
        STE_P, STE_Q = STE[:, : self.P], STE[:, self.P:]
        for blk in self.encoder:
            X = blk(X, STE_P)
        X = self.trans(X, STE_P, STE_Q)
        for blk in self.decoder:
            X = blk(X, STE_Q)
        return self.out_fc(X)


@register("gman")
class GMANForecaster(GNNForecasterBase):
    """GMAN forecaster (port — spatial/temporal/transform attention, no adjacency)."""

    name: str = "gman"
    requires_graph: bool = False  # learned node embedding replaces the reference's node2vec SE
    n_heads: int = 4
    head_dim: int = 8
    n_layers: int = 2
    dropout: float = 0.1
    bn: bool = True
    use_tod_dow: bool = False       # set True for sub-daily data (e.g. 5-min traffic)
    steps_per_day: int = 288        # 24h / 5min

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        return GMANNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            seq_len=int(bundle.raw.spec.seq_len),
            n_heads=int(self.n_heads),
            head_dim=int(self.head_dim),
            n_layers=int(self.n_layers),
            dropout=float(self.dropout),
            bn=bool(self.bn),
            use_tod_dow=bool(self.use_tod_dow),
            steps_per_day=int(self.steps_per_day),
        )

    def _graph_forward(self, net, x):
        return net(x, x_mark=self._x_mark)
