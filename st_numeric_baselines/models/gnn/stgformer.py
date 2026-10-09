"""STGformer — Spatiotemporal Graph Transformer (Dreamzz5/STGformer).

Reference: https://github.com/Dreamzz5/STGformer

Each block alternates temporal self-attention (per node, across the lookback
window) with graph propagation over two supports: the benchmark's static,
symmetric-normalized adjacency (``self._A_norm``, reused as-is — this is the
same normalization STGformer's default config uses) and a purely internal
learned "adaptive graph" (softmax over a trainable node-embedding similarity
matrix, needs no external data). Time-of-day/day-of-week features from the
original paper are omitted here since this benchmark's datasets are monthly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from st_numeric_baselines.models.gnn.gnn_forecaster import GNNForecasterBase
from st_numeric_baselines.models.registry import register


class AdaptiveGraph(nn.Module):
    """Purely internal learned graph: softmax(relu(E @ E.T)) over node embeddings."""

    def __init__(self, n_nodes: int, emb_dim: int = 10) -> None:
        super().__init__()
        self.emb = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)

    def forward(self) -> torch.Tensor:
        scores = F.relu(self.emb @ self.emb.t())
        return F.softmax(scores, dim=-1)


class GraphPropagate(nn.Module):
    """K-hop graph propagation over one dense support, concat hops then project."""

    def __init__(self, d_model: int, order: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        self.order = int(order)
        c_cat = d_model * (1 + self.order)
        self.linear = nn.Linear(c_cat, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # h: [B, T, N, D]   A: [N, N] dense
        outs = [h]
        x1 = h
        for _ in range(self.order):
            x1 = torch.einsum("nm,btmd->btnd", A, x1)
            outs.append(x1)
        out = torch.cat(outs, dim=-1)
        return self.drop(self.linear(out))


class STGformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, order: int, dropout: float) -> None:
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.static_prop = GraphPropagate(d_model, order, dropout)
        self.adaptive_prop = GraphPropagate(d_model, order, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, A_static: torch.Tensor, A_adapt: torch.Tensor) -> torch.Tensor:
        B, T, N, D = h.shape

        # temporal self-attention, independently per node
        h_bn = h.permute(0, 2, 1, 3).reshape(B * N, T, D)
        attn_out, _ = self.temporal_attn(h_bn, h_bn, h_bn)
        h_bn = self.norm1(h_bn + self.drop(attn_out))
        h_t = h_bn.reshape(B, N, T, D).permute(0, 2, 1, 3)  # [B, T, N, D]

        # spatial graph propagation: static + adaptive supports
        h_sp = self.static_prop(h_t, A_static) + self.adaptive_prop(h_t, A_adapt)
        h_t = self.norm2(h_t + self.drop(h_sp))

        h_ffn = self.ffn(h_t)
        return self.norm3(h_t + self.drop(h_ffn))


class STGformerNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        seq_len: int,
        *,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        order: int = 2,
        adaptive_emb_dim: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)

        self.in_proj = nn.Linear(int(input_dim), int(d_model))
        self.adaptive_graph = AdaptiveGraph(n_nodes, int(adaptive_emb_dim))
        self.blocks = nn.ModuleList([
            STGformerBlock(int(d_model), int(n_heads), int(order), float(dropout))
            for _ in range(int(n_layers))
        ])
        self.out_proj = nn.Linear(int(d_model) * int(seq_len), self.pred_len * self.out_dim)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        B, L, N, _ = x.shape
        h = self.in_proj(x)
        A_adapt = self.adaptive_graph()
        for block in self.blocks:
            h = block(h, A_norm, A_adapt)

        h_flat = h.permute(0, 2, 1, 3).reshape(B, N, -1)  # [B, N, L*D]
        out = self.out_proj(h_flat)  # [B, N, pred_len*out_dim]
        out = out.view(B, N, self.pred_len, self.out_dim).permute(0, 2, 1, 3)
        return out  # [B, pred_len, N, out_dim]


@register("stgformer")
class STGformerForecaster(GNNForecasterBase):
    """STGformer forecaster (direct port — no repo-level extensions needed)."""

    name: str = "stgformer"
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    order: int = 2
    adaptive_emb_dim: int = 10
    dropout: float = 0.1

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        return STGformerNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            seq_len=int(bundle.raw.spec.seq_len),
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            n_layers=int(self.n_layers),
            order=int(self.order),
            adaptive_emb_dim=int(self.adaptive_emb_dim),
            dropout=float(self.dropout),
        )

    def _graph_forward(self, net, x):
        return net(x, self._A_norm.to(x.device).to_dense())
