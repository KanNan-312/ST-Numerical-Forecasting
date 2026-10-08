"""MTGNN — "Connecting the Dots": Multivariate Time Series Forecasting with
Graph Neural Networks (Wu et al., KDD 2020).

Reference: https://github.com/nnzhan/MTGNN

Learns a **directed, sparse** adjacency purely from two trainable
node-embedding matrices, then sparsifies it to each node's top-k neighbors —
no external graph is ever consulted (``requires_graph = False``, runs on a
dataset with no ``graph.npz``/``dataset.graph.path`` at all). A stack of
dilated-inception temporal convolutions (parallel branches with kernel sizes
2/3/6/7, concatenated) alternates with mix-hop graph propagation over both
the learned graph and its transpose (the two "directions" — since the
learned adjacency is asymmetric), matching the paper's own two-directional
design.

Adaptations from ``nnzhan/MTGNN``: every temporal conv branch uses causal
left-padding (matching this registry's ``graph_wavenet.py``) instead of the
original's receptive-field-aware cropping, so sequence length is preserved
through the whole stack — simpler skip-connection bookkeeping, no behavior
difference for this benchmark's short (monthly) lookback windows. The
original's subgraph-sampling curriculum (for scaling to thousands of nodes)
is dropped as irrelevant at this benchmark's node counts, and training uses
the benchmark's standard full-horizon MSE loss rather than the paper's
step-wise curriculum.

Traffic data: MTGNN's own traffic experiments follow the same DCRNN/Graph
WaveNet-family convention (confirmed from source) — time-of-day as a plain
extra **input channel**, no model-side embedding. No model change needed
here either; include ``time_of_day`` via the dataset config's
``feature_cols`` (see ``housets_bench.data.io.load_metr_la``/``load_pems08``).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from housets_bench.models.gnn.gnn_forecaster import GNNForecasterBase
from housets_bench.models.registry import register


class GraphLearningLayer(nn.Module):
    """Learns a directed, top-k-sparsified adjacency from two node-embedding matrices.

    ``A = relu(tanh(alpha * (M1 @ M2.T - M2 @ M1.T)))``, ``Mi = tanh(alpha * theta_i(Ei))``
    — asymmetric by construction (the two terms don't cancel), then each row keeps
    only its top-k values (the rest zeroed) so message passing stays sparse.
    """

    def __init__(self, n_nodes: int, emb_dim: int, top_k: int, alpha: float = 3.0) -> None:
        super().__init__()
        self.emb1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.emb2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.lin1 = nn.Linear(emb_dim, emb_dim)
        self.lin2 = nn.Linear(emb_dim, emb_dim)
        self.top_k = int(top_k)
        self.alpha = float(alpha)

    def forward(self) -> torch.Tensor:
        n = self.emb1.shape[0]
        m1 = torch.tanh(self.alpha * self.lin1(self.emb1))
        m2 = torch.tanh(self.alpha * self.lin2(self.emb2))
        scores = F.relu(torch.tanh(self.alpha * (m1 @ m2.t() - m2 @ m1.t())))

        k = min(self.top_k, n)
        _, topk_idx = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(1, topk_idx, 1.0)
        return scores * mask  # [N, N], directed, row-sparse


class DilatedInception(nn.Module):
    """Parallel dilated conv branches (kernel sizes in ``kernel_set``), causal-padded
    to the same output length, concatenated along the channel axis."""

    def __init__(self, c_in: int, c_out: int, dilation: int, kernel_set: Sequence[int] = (2, 3, 6, 7)) -> None:
        super().__init__()
        self.kernel_set = list(kernel_set)
        self.dilation = int(dilation)
        n_k = len(self.kernel_set)
        if c_out % n_k != 0:
            raise ValueError(f"c_out={c_out} must be divisible by len(kernel_set)={n_k}")
        c_each = c_out // n_k
        self.convs = nn.ModuleList(
            [nn.Conv2d(c_in, c_each, kernel_size=(1, k), dilation=(1, self.dilation)) for k in self.kernel_set]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, T] -> each branch causal-padded to preserve T
        outs = []
        for k, conv in zip(self.kernel_set, self.convs):
            pad = (k - 1) * self.dilation
            x_p = F.pad(x, (pad, 0, 0, 0)) if pad > 0 else x
            outs.append(conv(x_p))
        return torch.cat(outs, dim=1)


class MixHopProp(nn.Module):
    """Mix-hop graph propagation: ``H^(k) = beta*H^(0) + (1-beta)*A@H^(k-1)``, concat
    hops 0..order then project — a retain-ratio variant of order-K graph convolution."""

    def __init__(self, c_in: int, c_out: int, order: int = 2, beta: float = 0.05, dropout: float = 0.0) -> None:
        super().__init__()
        self.order = int(order)
        self.beta = float(beta)
        self.mlp = nn.Conv2d(c_in * (self.order + 1), c_out, kernel_size=(1, 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N, T]   A: [N, N] row-normalized
        h0 = x
        h = x
        outs = [h]
        for _ in range(self.order):
            h = self.beta * h0 + (1.0 - self.beta) * torch.einsum("nm,bcmt->bcnt", A, h)
            outs.append(h)
        h_cat = torch.cat(outs, dim=1)
        return self.drop(self.mlp(h_cat))


def _row_normalize(A: torch.Tensor) -> torch.Tensor:
    row_sum = A.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return A / row_sum


class MTGNNNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        *,
        residual_channels: int = 16,
        dilation_channels: int = 16,
        skip_channels: int = 32,
        end_channels: int = 64,
        n_blocks: int = 2,
        n_layers: int = 2,
        gcn_order: int = 2,
        top_k: int = 8,
        emb_dim: int = 16,
        kernel_set: Sequence[int] = (2, 3, 6, 7),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = int(pred_len)
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)

        self.graph_learner = GraphLearningLayer(n_nodes, int(emb_dim), int(top_k))
        self.start_conv = nn.Conv2d(int(input_dim), int(residual_channels), kernel_size=(1, 1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.gconv_fwd = nn.ModuleList()
        self.gconv_bwd = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _b in range(int(n_blocks)):
            for i in range(int(n_layers)):
                dilation = 2**i
                self.filter_convs.append(
                    DilatedInception(residual_channels, dilation_channels, dilation, kernel_set)
                )
                self.gate_convs.append(
                    DilatedInception(residual_channels, dilation_channels, dilation, kernel_set)
                )
                self.gconv_fwd.append(MixHopProp(dilation_channels, residual_channels, gcn_order, dropout=dropout))
                self.gconv_bwd.append(MixHopProp(dilation_channels, residual_channels, gcn_order, dropout=dropout))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.norms.append(nn.BatchNorm2d(residual_channels))

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, self.pred_len * self.out_dim, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N, Dx] -> [B, Dx, N, L]
        B, L, N, Dx = x.shape
        xt = x.permute(0, 3, 2, 1).contiguous()

        A = self.graph_learner()  # [N, N] directed, row-sparse
        A_fwd = _row_normalize(A)
        A_bwd = _row_normalize(A.t())

        h = self.start_conv(xt)
        skip = 0.0
        for filt_conv, gate_conv, gfwd, gbwd, skip_conv, norm in zip(
            self.filter_convs, self.gate_convs, self.gconv_fwd, self.gconv_bwd, self.skip_convs, self.norms
        ):
            residual = h
            filt = torch.tanh(filt_conv(h))
            gate = torch.sigmoid(gate_conv(h))
            h_t = filt * gate
            h_t = F.dropout(h_t, p=self.dropout, training=self.training)

            skip = skip + skip_conv(h_t)
            h_gc = gfwd(h_t, A_fwd) + gbwd(h_t, A_bwd)
            h = norm(h_gc + residual)

        out = F.relu(skip)
        out = F.relu(self.end_conv_1(out))
        out = self.end_conv_2(out)  # [B, pred_len*out_dim, N, L]
        out = out[..., -1]  # [B, pred_len*out_dim, N] — last (fully-causal-context) timestep
        out = out.permute(0, 2, 1)  # [B, N, pred_len*out_dim]
        return out.view(B, N, self.pred_len, self.out_dim).permute(0, 2, 1, 3)


@register("mtgnn")
class MTGNNForecaster(GNNForecasterBase):
    """MTGNN forecaster: self-learned directed sparse graph + mix-hop dilated-inception TCN."""

    name: str = "mtgnn"
    requires_graph: bool = False
    residual_channels: int = 16
    dilation_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    n_blocks: int = 2
    n_layers: int = 2
    gcn_order: int = 2
    top_k: int = 8
    emb_dim: int = 16
    dropout: float = 0.1

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        return MTGNNNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            residual_channels=int(self.residual_channels),
            dilation_channels=int(self.dilation_channels),
            skip_channels=int(self.skip_channels),
            end_channels=int(self.end_channels),
            n_blocks=int(self.n_blocks),
            n_layers=int(self.n_layers),
            gcn_order=int(self.gcn_order),
            top_k=min(int(self.top_k), n_nodes),
            emb_dim=int(self.emb_dim),
            dropout=float(self.dropout),
        )

    def _graph_forward(self, net, x):
        return net(x)
