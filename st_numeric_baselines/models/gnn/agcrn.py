"""AGCRN — Adaptive Graph Convolutional Recurrent Network (Bai et al., NeurIPS 2020).

Reference: https://github.com/LeiBAI/AGCRN

Learns its own adjacency purely from trainable per-node embeddings — DAGG
("Data-Adaptive Graph Generation": ``A = softmax(relu(E @ E.T))``) — and
gives every node its own graph-conv weight matrix via NAPL ("Node-Adaptive
Parameter Learning": a node's weights are a linear combination, indexed by
its own embedding, of a small shared weight pool). **No external adjacency
is ever consulted** — ``requires_graph = False``, runs on a dataset with no
``graph.npz``/``dataset.graph.path`` at all. The resulting node-adaptive
graph convolution ("AGCN") replaces a GRU's linear gates ("AGCRN cell"),
stacked into a recurrent encoder over the lookback window.

Adaptation: the paper's own decoder is a second AGCRN-cell stack run
autoregressively with scheduled sampling — the same mechanism this
registry's ``dcrnn.py`` already implements. To keep this addition's
differentiator to the adaptive-graph + node-adaptive-parameter mechanism
(rather than a second copy of scheduled-sampling decoding), the multi-step
forecast head here is a single direct linear projection from the encoder's
final hidden state, matching the direct-multi-horizon-head convention
already used by ``gcn_tcn``/``stgcn`` in this registry.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from st_numeric_baselines.models.gnn.gnn_forecaster import GNNForecasterBase
from st_numeric_baselines.models.registry import register


class DAGG(nn.Module):
    """Data-Adaptive Graph Generation: softmax(relu(E @ E.T)) over trainable node embeddings."""

    def __init__(self, n_nodes: int, emb_dim: int) -> None:
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)

    def forward(self):
        scores = F.relu(self.node_emb @ self.node_emb.t())
        A = F.softmax(scores, dim=-1)
        return A, self.node_emb


class AGCN(nn.Module):
    """Node-Adaptive-Parameter graph convolution over an order-K polynomial of ``A``."""

    def __init__(self, in_dim: int, out_dim: int, emb_dim: int, order: int = 2) -> None:
        super().__init__()
        self.order = int(order)
        n_supports = self.order + 1  # A^0 = I, A^1, ..., A^order
        self.weight_pool = nn.Parameter(torch.randn(emb_dim, n_supports, in_dim, out_dim) * 0.1)
        self.bias_pool = nn.Parameter(torch.randn(emb_dim, out_dim) * 0.1)

    def forward(self, x: torch.Tensor, A: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, N, in_dim]   A: [N, N]   node_emb: [N, emb_dim]
        n = A.shape[0]
        supports = [torch.eye(n, device=A.device, dtype=A.dtype)]
        a_pow = A
        supports.append(a_pow)
        for _ in range(2, self.order + 1):
            a_pow = A @ a_pow
            supports.append(a_pow)
        supports_t = torch.stack(supports, dim=0)  # [n_supports, N, N]

        x_g = torch.einsum("knm,bmi->bkni", supports_t, x)  # [B, n_supports, N, in_dim]
        x_g = x_g.permute(0, 2, 1, 3)  # [B, N, n_supports, in_dim]

        weights = torch.einsum("ne,ekio->nkio", node_emb, self.weight_pool)  # [N, n_supports, in_dim, out_dim]
        bias = node_emb @ self.bias_pool  # [N, out_dim]

        out = torch.einsum("bnki,nkio->bno", x_g, weights) + bias.unsqueeze(0)
        return out  # [B, N, out_dim]


class AGCRNCell(nn.Module):
    """GRU cell with AGCN replacing the linear gates."""

    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int, order: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gate = AGCN(in_dim + hidden_dim, 2 * hidden_dim, emb_dim, order)
        self.update = AGCN(in_dim + hidden_dim, hidden_dim, emb_dim, order)

    def forward(self, x: torch.Tensor, h: torch.Tensor, A: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=-1)
        zr = torch.sigmoid(self.gate(combined, A, node_emb))
        z, r = zr.chunk(2, dim=-1)
        candidate = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.update(candidate, A, node_emb))
        return z * h + (1.0 - z) * h_tilde


class AGCRNNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        *,
        hidden_dim: int = 32,
        emb_dim: int = 10,
        order: int = 2,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)

        self.dagg = DAGG(n_nodes, emb_dim)
        self.cells = nn.ModuleList()
        for layer in range(self.n_layers):
            in_d = input_dim if layer == 0 else hidden_dim
            self.cells.append(AGCRNCell(in_d, hidden_dim, emb_dim, order))
        self.drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, self.pred_len * self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N, Dx]
        B, L, N, Dx = x.shape
        A, node_emb = self.dagg()

        h_states = [
            torch.zeros(B, N, self.hidden_dim, device=x.device, dtype=x.dtype) for _ in range(self.n_layers)
        ]
        for t in range(L):
            inp = x[:, t, :, :]
            for layer, cell in enumerate(self.cells):
                h_states[layer] = cell(inp, h_states[layer], A, node_emb)
                inp = self.drop(h_states[layer])

        h_final = h_states[-1]  # [B, N, hidden_dim]
        out = self.out_proj(h_final)  # [B, N, pred_len*out_dim]
        return out.view(B, N, self.pred_len, self.out_dim).permute(0, 2, 1, 3)


@register("agcrn")
class AGCRNForecaster(GNNForecasterBase):
    """AGCRN forecaster: fully data-adaptive graph + node-adaptive GCN-GRU, no external adjacency."""

    name: str = "agcrn"
    requires_graph: bool = False
    hidden_dim: int = 32
    emb_dim: int = 10
    order: int = 2
    n_layers: int = 2
    dropout: float = 0.1

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        return AGCRNNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            hidden_dim=int(self.hidden_dim),
            emb_dim=int(self.emb_dim),
            order=int(self.order),
            n_layers=int(self.n_layers),
            dropout=float(self.dropout),
        )

    def _graph_forward(self, net, x):
        return net(x)
