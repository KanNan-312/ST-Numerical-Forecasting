"""D2STGNN — Decoupled Dynamic Spatial-Temporal Graph Neural Network (VLDB 2022).

Reference: https://github.com/GestaltCogTeam/D2STGNN

Each layer decouples an "inherent" per-node temporal signal (a plain causal
conv, no graph) from a "diffusion" spatial signal — the paper's core idea.
The diffusion path combines three supports: the benchmark's static adjacency
(``self._A_norm``, from ``graph.npz``), a purely internal learned adaptive
graph (softmax over trainable node embeddings, no external data), and a
per-window **dynamic** graph computed from the current input's own hidden
states each forward pass (attention over a mean-pooled per-node summary) —
this last one is fully self-contained (no externally supplied dynamic-graph
data), matching how the original derives its dynamic graph model-internally.

Time-of-day/day-of-week embeddings from the original paper are omitted here
since this benchmark's datasets are monthly.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from st_numeric_baselines.models.gnn.gnn_forecaster import GNNForecasterBase
from st_numeric_baselines.models.registry import register


class AdaptiveStaticGraph(nn.Module):
    """Purely internal learned graph: softmax(relu(E1 @ E2.T))."""

    def __init__(self, n_nodes: int, emb_dim: int = 10) -> None:
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.e2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)

    def forward(self) -> torch.Tensor:
        return F.softmax(F.relu(self.e1 @ self.e2.t()), dim=-1)


class DynamicGraph(nn.Module):
    """Per-window dynamic graph: attention over a mean-pooled per-node summary."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, N, D] -> per-node summary over the lookback window
        summary = h.mean(dim=1)  # [B, N, D]
        q = self.q(summary)
        k = self.k(summary)
        scores = torch.einsum("bnd,bmd->bnm", q, k) / (q.shape[-1] ** 0.5)
        return F.softmax(scores, dim=-1)  # [B, N, N]


class InherentBlock(nn.Module):
    """Per-node temporal-only processing (no graph) — the "inherent" signal path."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, N, D]
        B, T, N, D = h.shape
        h_bn = h.permute(0, 2, 3, 1).reshape(B * N, D, T)
        out = self.conv(h_bn).permute(0, 2, 1).reshape(B, N, T, D).permute(0, 2, 1, 3)
        return self.norm(h + self.drop(out))


class DiffusionBlock(nn.Module):
    """K-hop diffusion over static supports (external + adaptive) plus one dynamic support."""

    def __init__(self, d_model: int, n_static: int, order: int, dropout: float) -> None:
        super().__init__()
        self.order = int(order)
        c_cat = d_model * (1 + n_static * self.order + self.order)
        self.linear = nn.Linear(c_cat, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, static_supports: Sequence[torch.Tensor], A_dyn: torch.Tensor) -> torch.Tensor:
        outs: List[torch.Tensor] = [h]
        for A in static_supports:
            x1 = h
            for _ in range(self.order):
                x1 = torch.einsum("nm,btmd->btnd", A, x1)
                outs.append(x1)
        x2 = h
        for _ in range(self.order):
            x2 = torch.einsum("bnm,btmd->btnd", A_dyn, x2)
            outs.append(x2)
        out = torch.cat(outs, dim=-1)
        return self.drop(self.linear(out))


class D2STGNNNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        seq_len: int,
        *,
        d_model: int = 32,
        n_layers: int = 2,
        order: int = 2,
        adaptive_emb_dim: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)

        self.in_proj = nn.Linear(int(input_dim), int(d_model))
        self.static_adaptive = AdaptiveStaticGraph(n_nodes, int(adaptive_emb_dim))

        n_layers = int(n_layers)
        self.dyn_graphs = nn.ModuleList([DynamicGraph(int(d_model)) for _ in range(n_layers)])
        self.inherent_blocks = nn.ModuleList([InherentBlock(int(d_model), float(dropout)) for _ in range(n_layers)])
        self.diffusion_blocks = nn.ModuleList([
            DiffusionBlock(int(d_model), n_static=2, order=int(order), dropout=float(dropout))
            for _ in range(n_layers)
        ])
        self.gates = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(int(d_model)) for _ in range(n_layers)])

        self.out_proj = nn.Linear(int(d_model) * int(seq_len), self.pred_len * self.out_dim)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        B, L, N, _ = x.shape
        h = self.in_proj(x)
        A_adaptive = self.static_adaptive()
        static_supports = [A_norm, A_adaptive]

        for inherent, dyn_g, diffusion, gate, norm in zip(
            self.inherent_blocks, self.dyn_graphs, self.diffusion_blocks, self.gates, self.norms
        ):
            h_inh = inherent(h)
            A_dyn = dyn_g(h)
            h_diff = diffusion(h, static_supports, A_dyn)
            h = norm(h + h_inh + torch.sigmoid(gate) * h_diff)

        h_flat = h.permute(0, 2, 1, 3).reshape(B, N, -1)  # [B, N, L*D]
        out = self.out_proj(h_flat).view(B, N, self.pred_len, self.out_dim).permute(0, 2, 1, 3)
        return out  # [B, pred_len, N, out_dim]


@register("d2stgnn")
class D2STGNNForecaster(GNNForecasterBase):
    """D2STGNN forecaster: decoupled inherent/diffusion paths with a self-contained dynamic graph."""

    name: str = "d2stgnn"
    d_model: int = 32
    n_layers: int = 2
    order: int = 2
    adaptive_emb_dim: int = 10
    dropout: float = 0.1

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        return D2STGNNNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            seq_len=int(bundle.raw.spec.seq_len),
            d_model=int(self.d_model),
            n_layers=int(self.n_layers),
            order=int(self.order),
            adaptive_emb_dim=int(self.adaptive_emb_dim),
            dropout=float(self.dropout),
        )

    def _graph_forward(self, net, x):
        return net(x, self._A_norm.to(x.device).to_dense())
