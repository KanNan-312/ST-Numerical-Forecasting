"""ST-HHOL — hierarchical hypergraph idea, simplified static-graph port.

Reference (original): https://github.com/777Rebecca/ST-HHOL
("Spatio-Temporal Hierarchical Hypergraph Online Learning for Crime Prediction")

**This is an explicitly simplified adaptation, not the paper's full model.**
The research pass on this model found it requires multiple heterogeneous +
homogeneous hypergraphs built from weather/POI/socioeconomic/311 data, plus an
online/streaming/concept-drift training loop — none of which this benchmark
provides. Per the user's "best-effort, static-graph simplification" decision:
this keeps only the hierarchical-hypergraph *idea* — a trainable hypergraph
incidence matrix learned end-to-end over the crime-count panel (standard HGNN
message passing: ``X' = Dv^-1 H W De^-1 H^T X Theta``), stacked with a
per-node temporal encoder — trained with the benchmark's normal offline
train/val/test time split and a standard multi-horizon output head (not the
paper's single-step head). No grid-coordinate requirement: the incidence
matrix is learned directly over the benchmark's id-ordered ``N`` nodes, and no
external adjacency is needed (the hypergraph is fully self-contained/trainable,
so ``graph.npz`` is not required for this model).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from st_numeric_baselines.models.gnn.gnn_forecaster import GNNForecasterBase
from st_numeric_baselines.models.registry import register


class HypergraphConv(nn.Module):
    """Trainable hypergraph convolution: ``X' = Dv^-1 H De^-1 H^T X Theta``."""

    def __init__(self, d_model: int, n_nodes: int, n_hyperedges: int, dropout: float) -> None:
        super().__init__()
        self.H_logit = nn.Parameter(torch.randn(n_nodes, n_hyperedges) * 0.1)
        self.theta = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, D]
        H = torch.sigmoid(self.H_logit)  # soft incidence, [N, K]
        Dv_inv = H.sum(dim=1).clamp_min(1e-8).pow(-1.0)  # [N]
        De_inv = H.sum(dim=0).clamp_min(1e-8).pow(-1.0)  # [K]

        x_theta = self.theta(x)  # [B, T, N, D]
        HT_X = torch.einsum("nk,btnd->btkd", H, x_theta) * De_inv.view(1, 1, -1, 1)
        H_HTX = torch.einsum("nk,btkd->btnd", H, HT_X) * Dv_inv.view(1, 1, -1, 1)
        return self.drop(H_HTX)


class STHHOLNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        *,
        d_model: int = 32,
        n_hyperedges: int = 16,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)

        self.in_proj = nn.Linear(int(input_dim), int(d_model))
        self.hg_layers = nn.ModuleList([
            HypergraphConv(int(d_model), n_nodes, int(n_hyperedges), float(dropout)) for _ in range(int(n_layers))
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(int(d_model)) for _ in range(int(n_layers))])
        self.temporal = nn.GRU(int(d_model), int(d_model), batch_first=True)
        self.out_proj = nn.Linear(int(d_model), self.pred_len * self.out_dim)

    def forward(self, x: torch.Tensor, A_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        # A_norm is unused: the hypergraph here is fully self-contained/trainable.
        B, L, N, _ = x.shape
        h = self.in_proj(x)
        for hg, norm in zip(self.hg_layers, self.norms):
            h = norm(h + hg(h))

        h_bn = h.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        _, h_last = self.temporal(h_bn)
        h_node = h_last.squeeze(0).view(B, N, -1)

        out = self.out_proj(h_node)  # [B, N, pred_len*out_dim]
        return out.view(B, N, self.pred_len, self.out_dim).permute(0, 2, 1, 3)


@register("st_hhol")
class STHHOLForecaster(GNNForecasterBase):
    """Simplified ST-HHOL forecaster: trainable hierarchical hypergraph over crime counts."""

    name: str = "st_hhol"
    requires_graph: bool = False  # self-contained learned hypergraph, no A_norm use
    d_model: int = 32
    n_hyperedges: int = 16
    n_layers: int = 2
    dropout: float = 0.1

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        return STHHOLNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            d_model=int(self.d_model),
            n_hyperedges=int(self.n_hyperedges),
            n_layers=int(self.n_layers),
            dropout=float(self.dropout),
        )

    def _graph_forward(self, net, x):
        return net(x)
