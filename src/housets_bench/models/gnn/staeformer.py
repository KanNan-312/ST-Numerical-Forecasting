"""STAEformer — Spatio-Temporal Adaptive Embedding Transformer (AAAI 2024).

Reference: https://github.com/XDZhelheim/STAEformer

Direct port, confirmed via source inspection: STAEformer does **not** use
graph convolution or the supplied adjacency at all — its spatial modeling is
pure multi-head self-attention over the node axis, informed only by a
learned per-node identity embedding (``node_emb``) and a learned
per-(timestep, node) "adaptive embedding" (``adaptive_embedding``), both
fully data-agnostic (no external graph needed). Each layer stack applies
temporal self-attention (over the lookback window, per node) followed by
spatial self-attention (over all nodes, per timestep) — all temporal layers
first, then all spatial layers, matching the reference implementation.
``requires_graph = False`` below: this model runs on a dataset with no
``graph.npz``/``dataset.graph.path`` at all (still uses ``GraphWindowDataset``
for its all-nodes-per-window batch shape, just no adjacency).

The reference's ``tod_embedding``/``dow_embedding`` (time-of-day /
day-of-week, wall-clock-periodicity features) are **off by default** since
this benchmark's original use case is monthly housing/crime data, where
sub-daily/weekly periodicity doesn't exist. Set ``use_tod_dow=True`` (e.g.
for 5-minute traffic data) to enable them — confirmed from source to be
``nn.Embedding(steps_per_day, dim)`` / ``nn.Embedding(7, dim)`` lookup
tables, concatenated into the same per-timestep feature stack as
``node_emb``/``adaptive_embedding``. They read ``tod_frac``/``dow`` from
``GNNForecasterBase._x_mark`` (``[B, L, 2]``, populated by
``GraphWindowDataset`` from ``AlignedData.time_marks`` — see
``data/io.py::_build_time_marks``), not from ``x_cols``, matching the
reference's own embedding-table mechanism rather than feeding them to the
model as raw values. The reference's masked MAE/Huber loss is replaced by
the benchmark's standard MSE training loss, matching every other GNN
baseline in this registry.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from housets_bench.models.gnn.gnn_forecaster import GNNForecasterBase
from housets_bench.models.registry import register


class SelfAttentionLayer(nn.Module):
    """Standard transformer block, applied along a chosen axis of a [B,L,N,D] tensor."""

    def __init__(self, model_dim: int, n_heads: int, feed_forward_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(model_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim), nn.ReLU(), nn.Linear(feed_forward_dim, model_dim)
        )
        self.norm2 = nn.LayerNorm(model_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # x: [B, L, N, D]; dim=1 -> attend over time (per node); dim=2 -> attend over nodes (per timestep)
        B, L, N, D = x.shape
        if dim == 1:
            x_flat = x.permute(0, 2, 1, 3).reshape(B * N, L, D)
        elif dim == 2:
            x_flat = x.reshape(B * L, N, D)
        else:
            raise ValueError("dim must be 1 (time) or 2 (node)")

        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm1(x_flat + self.drop(attn_out))
        ffn_out = self.ffn(x_flat)
        x_flat = self.norm2(x_flat + self.drop(ffn_out))

        if dim == 1:
            return x_flat.reshape(B, N, L, D).permute(0, 2, 1, 3)
        return x_flat.reshape(B, L, N, D)


class STAEformerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        seq_len: int,
        *,
        input_embedding_dim: int = 24,
        spatial_embedding_dim: int = 8,
        adaptive_embedding_dim: int = 24,
        n_heads: int = 4,
        n_layers: int = 3,
        feed_forward_dim: int = 64,
        dropout: float = 0.1,
        use_tod_dow: bool = False,
        steps_per_day: int = 288,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)
        self.seq_len = int(seq_len)

        self.input_proj = nn.Linear(int(input_dim), int(input_embedding_dim))

        self.spatial_embedding_dim = int(spatial_embedding_dim)
        if self.spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(torch.randn(n_nodes, self.spatial_embedding_dim) * 0.1)

        self.adaptive_embedding_dim = int(adaptive_embedding_dim)
        if self.adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.randn(self.seq_len, n_nodes, self.adaptive_embedding_dim) * 0.1
            )

        self.use_tod_dow = bool(use_tod_dow)
        self.steps_per_day = int(steps_per_day)
        self.tod_embedding_dim = int(tod_embedding_dim) if self.use_tod_dow else 0
        self.dow_embedding_dim = int(dow_embedding_dim) if self.use_tod_dow else 0
        if self.use_tod_dow:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)

        model_dim = (
            int(input_embedding_dim) + self.spatial_embedding_dim + self.adaptive_embedding_dim
            + self.tod_embedding_dim + self.dow_embedding_dim
        )
        if model_dim % n_heads != 0:
            raise ValueError(
                f"model_dim={model_dim} (input_embedding_dim + spatial_embedding_dim + "
                f"adaptive_embedding_dim + tod_embedding_dim + dow_embedding_dim) must be "
                f"divisible by n_heads={n_heads}"
            )
        self.model_dim = model_dim

        n_layers = int(n_layers)
        self.attn_layers_t = nn.ModuleList([
            SelfAttentionLayer(model_dim, n_heads, feed_forward_dim, dropout) for _ in range(n_layers)
        ])
        self.attn_layers_s = nn.ModuleList([
            SelfAttentionLayer(model_dim, n_heads, feed_forward_dim, dropout) for _ in range(n_layers)
        ])

        self.out_proj = nn.Linear(self.seq_len * model_dim, self.pred_len * self.out_dim)

    def forward(self, x: torch.Tensor, A_norm=None, x_mark: torch.Tensor = None) -> torch.Tensor:
        # A_norm is unused — STAEformer has no graph-convolution path.
        B, L, N, _ = x.shape
        h = self.input_proj(x)  # [B, L, N, input_embedding_dim]

        feats = [h]
        if self.spatial_embedding_dim > 0:
            se = self.node_emb.view(1, 1, N, self.spatial_embedding_dim).expand(B, L, -1, -1)
            feats.append(se)
        if self.adaptive_embedding_dim > 0:
            ae = self.adaptive_embedding.unsqueeze(0).expand(B, -1, -1, -1)
            feats.append(ae)
        if self.use_tod_dow:
            if x_mark is None:
                raise ValueError("use_tod_dow=True requires x_mark [B,L,2] (tod_frac, dow)")
            tod_idx = (x_mark[..., 0] * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
            dow_idx = x_mark[..., 1].long().clamp(0, 6)
            tod_emb = self.tod_embedding(tod_idx)  # [B, L, tod_dim]
            dow_emb = self.dow_embedding(dow_idx)  # [B, L, dow_dim]
            feats.append(tod_emb.unsqueeze(2).expand(-1, -1, N, -1))
            feats.append(dow_emb.unsqueeze(2).expand(-1, -1, N, -1))
        h = torch.cat(feats, dim=-1)  # [B, L, N, model_dim]

        for attn in self.attn_layers_t:
            h = attn(h, dim=1)
        for attn in self.attn_layers_s:
            h = attn(h, dim=2)

        h_flat = h.permute(0, 2, 1, 3).reshape(B, N, -1)  # [B, N, L*model_dim]
        out = self.out_proj(h_flat)  # [B, N, pred_len*out_dim]
        return out.view(B, N, self.pred_len, self.out_dim).permute(0, 2, 1, 3)


@register("staeformer")
class STAEformerForecaster(GNNForecasterBase):
    """STAEformer forecaster (direct port — pure attention, no graph convolution)."""

    name: str = "staeformer"
    requires_graph: bool = False  # pure attention + learned node/adaptive embeddings, no A_norm use
    input_embedding_dim: int = 24
    spatial_embedding_dim: int = 8
    adaptive_embedding_dim: int = 24
    n_heads: int = 4
    n_layers: int = 3
    feed_forward_dim: int = 64
    dropout: float = 0.1
    use_tod_dow: bool = False       # set True for sub-daily data (e.g. 5-min traffic)
    steps_per_day: int = 288        # 24h / 5min
    tod_embedding_dim: int = 24
    dow_embedding_dim: int = 24

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        return STAEformerNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            seq_len=int(bundle.raw.spec.seq_len),
            input_embedding_dim=int(self.input_embedding_dim),
            spatial_embedding_dim=int(self.spatial_embedding_dim),
            adaptive_embedding_dim=int(self.adaptive_embedding_dim),
            n_heads=int(self.n_heads),
            n_layers=int(self.n_layers),
            feed_forward_dim=int(self.feed_forward_dim),
            dropout=float(self.dropout),
            use_tod_dow=bool(self.use_tod_dow),
            steps_per_day=int(self.steps_per_day),
            tod_embedding_dim=int(self.tod_embedding_dim),
            dow_embedding_dim=int(self.dow_embedding_dim),
        )

    def _graph_forward(self, net, x):
        return net(x, x_mark=self._x_mark)
