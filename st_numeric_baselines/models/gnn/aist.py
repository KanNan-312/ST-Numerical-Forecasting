"""AIST — Attention-based Interpretable Spatio-Temporal network (simplified port).

Reference: https://github.com/YeasirRayhanPrince/aist

**This is an explicitly simplified adaptation, not the paper's full model.**
The research pass on this model found it hard-requires taxi inflow/outflow and
POI data this benchmark's flat crime panel doesn't have, and natively trains
one target region at a time rather than jointly over all nodes. Per the user's
"best-effort simplified port" decision: this keeps only the graph-attention
mechanism (masked multi-head attention over the supplied adjacency, à la GAT)
applied to crime counts, batched over **all nodes jointly** in one forward
pass (fixing the per-region training loop) — no taxi/POI/street-crime
branches. A per-node GRU handles the temporal side before spatial attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from st_numeric_baselines.models.gnn.gnn_forecaster import GNNForecasterBase
from st_numeric_baselines.models.registry import register


class MaskedGraphAttention(nn.Module):
    """Multi-head self-attention over nodes, masked to the supplied adjacency (+ self-loops)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        # h: [B, N, D]   adj_mask: [N, N] in {0, 1}, includes self-loops
        B, N, D = h.shape
        q = self.q(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhid,bhjd->bhij", q, k) / (self.head_dim ** 0.5)
        mask = adj_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class AISTNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        *,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)

        self.in_proj = nn.Linear(int(input_dim), int(d_model))
        self.temporal = nn.GRU(int(d_model), int(d_model), batch_first=True)
        self.attn_layers = nn.ModuleList([
            MaskedGraphAttention(int(d_model), int(n_heads), float(dropout)) for _ in range(int(n_layers))
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(int(d_model)) for _ in range(int(n_layers))])
        self.out_proj = nn.Linear(int(d_model), self.pred_len * self.out_dim)

    def forward(self, x: torch.Tensor, adj_mask: torch.Tensor) -> torch.Tensor:
        B, L, N, _ = x.shape
        h = self.in_proj(x)  # [B, L, N, D]

        h_bn = h.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        _, h_last = self.temporal(h_bn)
        h_node = h_last.squeeze(0).view(B, N, -1)  # [B, N, D]

        for attn, norm in zip(self.attn_layers, self.norms):
            a_out = attn(h_node, adj_mask)
            h_node = norm(h_node + a_out)

        out = self.out_proj(h_node)  # [B, N, pred_len*out_dim]
        return out.view(B, N, self.pred_len, self.out_dim).permute(0, 2, 1, 3)


@register("aist")
class AISTForecaster(GNNForecasterBase):
    """Simplified AIST forecaster: masked graph attention over crime counts, all nodes jointly."""

    name: str = "aist"
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        A_dense = self._A_raw.to(device).to_dense()
        adj_mask = (A_dense > 0).float()
        eye = torch.eye(n_nodes, device=device)
        adj_mask = torch.clamp(adj_mask + eye, max=1.0)

        net = AISTNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            d_model=int(self.d_model),
            n_heads=int(self.n_heads),
            n_layers=int(self.n_layers),
            dropout=float(self.dropout),
        )
        net.register_buffer("adj_mask", adj_mask)
        return net

    def _graph_forward(self, net, x):
        return net(x, net.adj_mask.to(x.device))
