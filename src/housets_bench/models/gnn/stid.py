"""STID — Spatial-Temporal Identity (Shao et al., CIKM'22).

Reference: https://github.com/GestaltCogTeam/STID  (arXiv:2208.05233)

The paper's headline finding: most of what spatiotemporal GNNs buy you over
a plain MLP comes from breaking *sample indistinguishability* (giving the
model a way to tell "this is region A" and "this is May" apart), not from
graph message-passing itself. STID has **no graph convolution and consults
no adjacency at all** — each node's flattened lookback window is encoded by
a shared MLP, concatenated with a learned per-node "spatial identity"
embedding, then pushed through a residual-MLP stack and a direct
multi-horizon regression head.

The paper's day-of-week/time-of-day ("temporal identity") embeddings are
**off by default**, matching this registry's original monthly-data
convention — this keeps only the spatial-identity half of the paper's
contribution, which its own ablation reports as the larger of the two. Set
``use_tod_dow=True`` (e.g. for 5-minute traffic data) to enable them:
confirmed from source as ``nn.Embedding(steps_per_day, dim)`` /
``nn.Embedding(7, dim)`` lookup tables, keyed by the lookback window's
*last* timestep (STID has no per-timestep output structure to attach a
per-step embedding to — it's one "temporal identity" per window, same as
``node_emb`` is one "spatial identity" per node), concatenated alongside
``node_emb``. Reads ``tod_frac``/``dow`` from
``GNNForecasterBase._x_mark`` (populated by ``GraphWindowDataset`` from
``AlignedData.time_marks``), not from ``x_cols``.
``requires_graph = False``: runs on a dataset with no
``graph.npz``/``dataset.graph.path`` at all.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from housets_bench.models.gnn.gnn_forecaster import GNNForecasterBase
from housets_bench.models.registry import register


class _ResidualMLP(nn.Module):
    """Two-layer residual MLP block (STID's ``MultiLayerPerceptron``)."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.act(self.fc1(x)))
        h = self.fc2(h)
        return x + h


class STIDNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        seq_len: int,
        *,
        embed_dim: int = 32,
        node_emb_dim: int = 16,
        n_layers: int = 3,
        dropout: float = 0.15,
        use_tod_dow: bool = False,
        steps_per_day: int = 288,
        tod_embedding_dim: int = 16,
        dow_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)

        self.history_encoder = nn.Linear(int(seq_len) * int(input_dim), int(embed_dim))
        self.node_emb = nn.Parameter(torch.randn(int(n_nodes), int(node_emb_dim)) * 0.1)

        self.use_tod_dow = bool(use_tod_dow)
        self.steps_per_day = int(steps_per_day)
        self.tod_embedding_dim = int(tod_embedding_dim) if self.use_tod_dow else 0
        self.dow_embedding_dim = int(dow_embedding_dim) if self.use_tod_dow else 0
        if self.use_tod_dow:
            self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
            self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)

        hidden_dim = int(embed_dim) + int(node_emb_dim) + self.tod_embedding_dim + self.dow_embedding_dim
        self.mlp_stack = nn.ModuleList([_ResidualMLP(hidden_dim, dropout) for _ in range(int(n_layers))])
        self.regression = nn.Linear(hidden_dim, self.pred_len * self.out_dim)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor = None) -> torch.Tensor:
        # x: [B, L, N, Dx]
        B, L, N, Dx = x.shape
        h_hist = x.permute(0, 2, 1, 3).reshape(B, N, L * Dx)  # [B, N, L*Dx]
        h_hist = self.history_encoder(h_hist)  # [B, N, embed_dim]

        node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)  # [B, N, node_emb_dim]
        feats = [h_hist, node_emb]
        if self.use_tod_dow:
            if x_mark is None:
                raise ValueError("use_tod_dow=True requires x_mark [B,L,2] (tod_frac, dow)")
            last_mark = x_mark[:, -1, :]  # [B, 2] -- one temporal identity per window, not per step
            tod_idx = (last_mark[:, 0] * self.steps_per_day).long().clamp(0, self.steps_per_day - 1)
            dow_idx = last_mark[:, 1].long().clamp(0, 6)
            tod_emb = self.tod_embedding(tod_idx).unsqueeze(1).expand(-1, N, -1)  # [B, N, tod_dim]
            dow_emb = self.dow_embedding(dow_idx).unsqueeze(1).expand(-1, N, -1)  # [B, N, dow_dim]
            feats.extend([tod_emb, dow_emb])
        h = torch.cat(feats, dim=-1)  # [B, N, hidden_dim]

        for block in self.mlp_stack:
            h = block(h)

        out = self.regression(h)  # [B, N, pred_len*out_dim]
        return out.view(B, N, self.pred_len, self.out_dim).permute(0, 2, 1, 3)  # [B, pred_len, N, out_dim]


@register("stid")
class STIDForecaster(GNNForecasterBase):
    """STID forecaster: spatial-identity MLP, no graph consulted at all."""

    name: str = "stid"
    requires_graph: bool = False
    embed_dim: int = 32
    node_emb_dim: int = 16
    n_layers: int = 3
    dropout: float = 0.15
    use_tod_dow: bool = False       # set True for sub-daily data (e.g. 5-min traffic)
    steps_per_day: int = 288        # 24h / 5min
    tod_embedding_dim: int = 16
    dow_embedding_dim: int = 16

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        return STIDNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            seq_len=int(bundle.raw.spec.seq_len),
            embed_dim=int(self.embed_dim),
            node_emb_dim=int(self.node_emb_dim),
            n_layers=int(self.n_layers),
            dropout=float(self.dropout),
            use_tod_dow=bool(self.use_tod_dow),
            steps_per_day=int(self.steps_per_day),
            tod_embedding_dim=int(self.tod_embedding_dim),
            dow_embedding_dim=int(self.dow_embedding_dim),
        )

    def _graph_forward(self, net, x):
        return net(x, x_mark=self._x_mark)
