"""Graph WaveNet (Wu et al., IJCAI 2019).

Reference: https://github.com/nnzhan/Graph-WaveNet

Traffic data (confirmed from the original repo's own
``generate_training_data.py``): time-of-day is a plain extra **input
channel** (no model-side embedding), computed in the data-prep script. No
model change is needed here for that — ``st_numeric_baselines.data.io.load_metr_la``/
``load_pems08`` already emit a ``time_of_day`` column; include it via the
dataset config's ``feature_cols`` and it flows in as an ordinary extra
channel of ``bundle.x_cols``, matching the original convention exactly.
Alignment with the reference (``model.py``/``util.py``/``train.py``):

* Self-adaptive adjacency is the paper's dense ``softmax(relu(E1 @ E2))``
  over all node pairs (it can add edges the predefined graph lacks), seeded
  by default from the SVD of the forward transition matrix (``aptinit``).
* Predefined supports are the ``doubletransition`` pair ``D^-1 A`` and
  ``D^-1 A^T``; with no graph the model runs adaptive-only (``--aptonly``).
* The reference's ``nconv`` einsum multiplies by ``A^T``; supports are passed
  pre-transposed so the result is identical.
* Not copied: the reference's masked-MAE loss / grad-clip 5 / input
  standardisation of the first channel only -- those follow this repo's
  shared GNN training loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from st_numeric_baselines.graph.torch_adj import spmm_nct


class AdaptiveAdj(nn.Module):
    """Self-adaptive adjacency (paper Eq. 5): ``softmax(relu(E1 @ E2), dim=1)``.

    Dense ``[N, N]`` -- every node pair can be connected, so it can discover
    dependencies absent from the predefined graph (or stand in for one
    entirely). ``init_adj`` (the forward transition matrix, as in the
    reference's default ``aptinit=supports[0]``) seeds ``E1``/``E2`` from its
    truncated SVD; ``None`` gives the reference's ``--randomadj`` N(0,1) init.
    """

    def __init__(self, n_nodes: int, emb_dim: int = 10, init_adj: Optional[torch.Tensor] = None):
        super().__init__()
        k = int(min(emb_dim, n_nodes))
        if init_adj is None:
            e1, e2 = torch.randn(n_nodes, k), torch.randn(k, n_nodes)
        else:
            A = init_adj.to_dense().float() if init_adj.is_sparse else init_adj.float()
            if n_nodes <= 2048:
                U, S, Vh = torch.linalg.svd(A)
                U, S, V = U[:, :k], S[:k], Vh[:k].t()
            else:
                U, S, V = torch.svd_lowrank(A, q=k)
            sq = S.sqrt()
            e1, e2 = U * sq, (V * sq).t()
        self.nodevec1 = nn.Parameter(e1.contiguous())
        self.nodevec2 = nn.Parameter(e2.contiguous())

    def forward(self) -> torch.Tensor:
        return F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)


def _apply_support(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``A @ x`` along the node axis of ``x`` [B,C,N,T] (sparse or dense ``A``)."""
    if A.is_sparse:
        return spmm_nct(A, x)
    return torch.einsum("vw,ncwl->ncvl", A, x)


class SparseGraphConv(nn.Module):
    """Mix-hop graph conv: ``[x, A x, A^2 x, ...]`` over all supports -> 1x1 conv."""

    def __init__(self, c_in: int, c_out: int, *, n_supports: int, order: int = 2, dropout: float = 0.0):
        super().__init__()
        self.order = int(order)
        self.dropout = float(dropout)
        if self.order < 1:
            raise ValueError("order must be >= 1")

        c_cat = c_in * (1 + n_supports * self.order)
        self.mlp = nn.Conv2d(c_cat, c_out, kernel_size=(1, 1), bias=True)

    def forward(self, x: torch.Tensor, supports: Sequence[torch.Tensor]) -> torch.Tensor:
        # x: [B,C,N,T]
        out = [x]
        for A in supports:
            x1 = _apply_support(A, x)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = _apply_support(A, x1)
                out.append(x1)
        h = self.mlp(torch.cat(out, dim=1))
        if self.dropout > 0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GraphWaveNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pred_len: int,
        n_nodes: int,
        *,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 128,
        end_channels: int = 256,
        kernel_size: int = 2,
        n_blocks: int = 4,
        n_layers: int = 2,
        gcn_order: int = 2,
        dropout: float = 0.1,
        adaptive_adj: bool = True,
        adaptive_emb_dim: int = 10,
        supports: Optional[Sequence[torch.Tensor]] = None,
        adaptive_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.pred_len = int(pred_len)
        self.n_nodes = int(n_nodes)

        self.residual_channels = int(residual_channels)
        self.dilation_channels = int(dilation_channels)
        self.skip_channels = int(skip_channels)
        self.end_channels = int(end_channels)

        self.kernel_size = int(kernel_size)
        self.n_blocks = int(n_blocks)
        self.n_layers = int(n_layers)
        self.dropout = float(dropout)

        self.gcn_order = int(gcn_order)
        self.adaptive_adj = bool(adaptive_adj)

        self.start_conv = nn.Conv2d(self.input_dim, self.residual_channels, kernel_size=(1, 1))

        # Predefined supports, each already in "A @ x" orientation (the
        # reference's einsum 'ncvl,vw->ncwl' applies A^T, so the forecaster
        # hands over the transposes). Kept as a plain list, not buffers: sparse
        # tensors don't survive state_dict round-trips; moved lazily in forward().
        self.supports: List[torch.Tensor] = list(supports) if supports else []
        self._supports_cache: dict = {}
        if not self.supports and not self.adaptive_adj:
            raise ValueError("graph_wavenet needs predefined supports or adaptive_adj=True (or both)")

        # Dense self-adaptive adjacency (paper Eq. 5); works with or without supports.
        self.adj_learner = None
        if self.adaptive_adj:
            self.adj_learner = AdaptiveAdj(self.n_nodes, int(adaptive_emb_dim), init_adj=adaptive_init)

        # temporal + graph conv stacks
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gconvs = nn.ModuleList()

        # receptive field not used for cropping because we do causal padding per layer
        for b in range(self.n_blocks):
            for i in range(self.n_layers):
                dilation = 2**i

                self.filter_convs.append(
                    nn.Conv2d(
                        self.residual_channels,
                        self.dilation_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        self.residual_channels,
                        self.dilation_channels,
                        kernel_size=(1, self.kernel_size),
                        dilation=(1, dilation),
                    )
                )

                self.skip_convs.append(nn.Conv2d(self.dilation_channels, self.skip_channels, kernel_size=(1, 1)))

                # graph conv expects supports list length: base + (adaptive if enabled)
                n_supports = len(self.supports) + (1 if self.adaptive_adj else 0)
                self.gconvs.append(
                    SparseGraphConv(
                        self.dilation_channels,
                        self.residual_channels,
                        n_supports=n_supports,
                        order=self.gcn_order,
                        dropout=self.dropout,
                    )
                )

                self.norms.append(nn.BatchNorm2d(self.residual_channels))

        self.end_conv_1 = nn.Conv2d(self.skip_channels, self.end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(self.end_channels, self.pred_len, kernel_size=(1, 1))

    def _supports_on(self, device: torch.device) -> List[torch.Tensor]:
        key = str(device)
        if key not in self._supports_cache:
            self._supports_cache[key] = [A.to(device) for A in self.supports]
        return self._supports_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,N,F] -> [B,F,N,L]
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.start_conv(x)
        skip = 0.0

        # Adaptive adjacency computed once per forward, like the reference
        # (transposed into "A @ x" orientation to match its einsum).
        cur_supports = list(self._supports_on(x.device))
        if self.adj_learner is not None:
            cur_supports.append(self.adj_learner().t())

        for layer in range(len(self.filter_convs)):
            residual = x

            # causal left padding on time 
            dilation = self.filter_convs[layer].dilation[1]
            pad = (self.kernel_size - 1) * dilation
            if pad > 0:
                x_p = F.pad(x, (pad, 0, 0, 0))
            else:
                x_p = x

            # gated temporal conv
            filt = torch.tanh(self.filter_convs[layer](x_p))
            gate = torch.sigmoid(self.gate_convs[layer](x_p))
            x_t = filt * gate
            x_t = F.dropout(x_t, p=self.dropout, training=self.training)

            # skip
            s = self.skip_convs[layer](x_t)
            skip = skip + s

            # graph conv
            x_gc = self.gconvs[layer](x_t, cur_supports)

            # residual connection
            x = x_gc + residual   # reference: gcn output is already residual_channels wide, no extra 1x1

            # norm
            x = self.norms[layer](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # [B,pred_len,N,T]
        # take last time step
        x = x[..., -1]  # [B,pred_len,N]
        x = x.unsqueeze(-1)  # [B,pred_len,N,1]
        return x
