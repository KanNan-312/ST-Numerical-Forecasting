"""Graph WaveNet (Wu et al., IJCAI 2019).

Reference: https://github.com/nnzhan/Graph-WaveNet

Traffic data (confirmed from the original repo's own
``generate_training_data.py``): time-of-day is a plain extra **input
channel** (no model-side embedding), computed in the data-prep script. No
model change is needed here for that — ``st_numeric_baselines.data.io.load_metr_la``/
``load_pems08`` already emit a ``time_of_day`` column; include it via the
dataset config's ``feature_cols`` and it flows in as an ordinary extra
channel of ``bundle.x_cols``, matching the original convention exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from st_numeric_baselines.graph.torch_adj import spmm_nct


class EdgeAdaptiveAdj(nn.Module):

    def __init__(self, n_nodes: int, src: torch.Tensor, dst: torch.Tensor, emb_dim: int = 16, eps: float = 1e-8):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.eps = float(eps)

        # store edges as buffers
        self.register_buffer("src", src.long())
        self.register_buffer("dst", dst.long())

        self.emb1 = nn.Parameter(torch.randn(self.n_nodes, emb_dim) * 0.1)
        self.emb2 = nn.Parameter(torch.randn(self.n_nodes, emb_dim) * 0.1)

    def forward(self) -> torch.Tensor:
        src = self.src
        dst = self.dst
        # score per edge
        score = (self.emb1[src] * self.emb2[dst]).sum(dim=-1)  # [E]
        w = torch.sigmoid(score)  # [E] in (0,1)

        # row normalization by src
        row_sum = torch.zeros(self.n_nodes, device=w.device, dtype=w.dtype)
        row_sum.index_add_(0, src, w)
        w_norm = w / (row_sum[src] + self.eps)

        idx = torch.stack([src, dst], dim=0)
        A = torch.sparse_coo_tensor(idx, w_norm, size=(self.n_nodes, self.n_nodes)).coalesce()
        return A


class SparseGraphConv(nn.Module):

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
            x1 = spmm_nct(A, x)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = spmm_nct(A, x1)
                out.append(x1)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
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
        adaptive_emb_dim: int = 16,
        base_edge_index: Optional[torch.Tensor] = None,  # [2,E] on CPU ok; will be moved
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

        # edge-adaptive adjacency over base sparse graph
        self.adj_learner = None
        if self.adaptive_adj:
            if base_edge_index is None:
                raise ValueError("adaptive_adj=True requires base_edge_index [2,E]")
            src = base_edge_index[0].long()
            dst = base_edge_index[1].long()
            self.adj_learner = EdgeAdaptiveAdj(self.n_nodes, src=src, dst=dst, emb_dim=int(adaptive_emb_dim))

        # temporal + graph conv stacks
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
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

                self.residual_convs.append(nn.Conv2d(self.dilation_channels, self.residual_channels, kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(self.dilation_channels, self.skip_channels, kernel_size=(1, 1)))

                # graph conv expects supports list length: base + (adaptive if enabled)
                n_supports = 1 + (1 if self.adaptive_adj else 0)
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

    def forward(self, x: torch.Tensor, supports: Sequence[torch.Tensor]) -> torch.Tensor:
        # x: [B,L,N,F] -> [B,F,N,L]
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.start_conv(x)
        skip = 0.0

        # prepare supports on correct device
        base_supports = list(supports)

        # ensure adaptive adjacency uses correct device buffers
        if self.adj_learner is not None:
            # move stored edges to device if needed 
            pass

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

            # supports: base + optional adaptive
            if self.adj_learner is not None:
                A_adapt = self.adj_learner()
                cur_supports = base_supports + [A_adapt]
            else:
                cur_supports = base_supports

            # graph conv
            x_gc = self.gconvs[layer](x_t, cur_supports)

            # residual connection
            x = self.residual_convs[layer](x_gc) + residual

            # norm
            x = self.norms[layer](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # [B,pred_len,N,T]
        # take last time step
        x = x[..., -1]  # [B,pred_len,N]
        x = x.unsqueeze(-1)  # [B,pred_len,N,1]
        return x
