"""ASTGCN — Attention-based Spatial-Temporal Graph Convolutional Network (AAAI 2019).

Reference: https://github.com/guoshnBJTU/ASTGCN-2019-pytorch  (``model/ASTGCN_r.py``)

Port of the reference ``ASTGCN_submodule`` -- the *recent*-component branch
(the paper's daily/weekly-periodic branches need pre-sliced multi-period
inputs that this benchmark's single-window contract doesn't provide, and the
reference repo's ``ASTGCN_r`` likewise ships a single branch). Each block is
temporal attention (re-weights the time axis) -> spatial attention
(``[B,N,N]`` scores) -> Chebyshev graph convolution whose polynomials are
element-wise modulated by the spatial attention -> a (1,3) temporal conv,
plus a 1x1 residual conv and LayerNorm. The final ``(1, F)`` conv maps the
time axis to the forecast horizon.

Differences from the reference, beyond the PyTorch port itself:

* The reference loops over time steps in ``cheb_conv_withSAt``; here it is a
  single einsum over the time axis (mathematically identical).
* Chebyshev polynomials are built from the dataset graph (``self._A_raw``,
  symmetrised) exactly as ``lib.utils.scaled_Laplacian``/``cheb_polynomial``:
  ``L = D - A``, ``L~ = 2L/lambda_max - I``, ``T_k = 2 L~ T_{k-1} - T_{k-2}``.
  They are registered as buffers so they follow ``.to(device)`` and are saved
  with the checkpoint.
* The reference leaves the attention/Theta parameters to a post-hoc xavier /
  uniform init in ``make_model``; the same is applied here.
* Multi-channel output (``len(y_cols) > 1``) is supported by widening the
  final conv; the loss is the benchmark's standard MSE, as for every GNN here.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from st_numeric_baselines.models.gnn.gnn_forecaster import GNNForecasterBase
from st_numeric_baselines.models.registry import register


def cheb_polynomials_from_adj(A: torch.Tensor, K: int) -> List[torch.Tensor]:
    """``K`` Chebyshev polynomials ``T_0..T_{K-1}`` of the scaled Laplacian of dense ``A``."""
    A = A.to(torch.float64)
    A = torch.maximum(A, A.t())                       # symmetrise
    L = torch.diag(A.sum(dim=1)) - A
    lam_max = torch.linalg.eigvalsh(L)[-1].clamp_min(1e-8)
    N = A.shape[0]
    I = torch.eye(N, dtype=A.dtype)
    L_tilde = 2.0 * L / lam_max - I
    polys = [I, L_tilde]
    for _ in range(2, K):
        polys.append(2.0 * L_tilde @ polys[-1] - polys[-2])
    return [p.float() for p in polys[:K]]


class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int, n_nodes: int, n_steps: int) -> None:
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(n_steps))
        self.W2 = nn.Parameter(torch.empty(in_channels, n_steps))
        self.W3 = nn.Parameter(torch.empty(in_channels))
        self.bs = nn.Parameter(torch.empty(1, n_nodes, n_nodes))
        self.Vs = nn.Parameter(torch.empty(n_nodes, n_nodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F, T] -> [B, N, N]
        lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)         # [B, N, T]
        rhs = torch.matmul(self.W3, x).transpose(-1, -2)              # [B, T, N]
        product = torch.matmul(lhs, rhs)                              # [B, N, N]
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        return F.softmax(S, dim=1)


class TemporalAttention(nn.Module):
    def __init__(self, in_channels: int, n_nodes: int, n_steps: int) -> None:
        super().__init__()
        self.U1 = nn.Parameter(torch.empty(n_nodes))
        self.U2 = nn.Parameter(torch.empty(in_channels, n_nodes))
        self.U3 = nn.Parameter(torch.empty(in_channels))
        self.be = nn.Parameter(torch.empty(1, n_steps, n_steps))
        self.Ve = nn.Parameter(torch.empty(n_steps, n_steps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F, T] -> [B, T, T]
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)  # [B, T, N]
        rhs = torch.matmul(self.U3, x)                                             # [B, N, T]
        product = torch.matmul(lhs, rhs)                                           # [B, T, T]
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        return F.softmax(E, dim=1)


class ChebConvWithSAt(nn.Module):
    """K-order Chebyshev graph conv with spatial-attention-modulated polynomials."""

    def __init__(self, K: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.K = int(K)
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.empty(in_channels, out_channels)) for _ in range(self.K)]
        )

    def forward(self, x: torch.Tensor, spatial_attention: torch.Tensor, cheb: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F_in, T]; spatial_attention: [B, N, N]; cheb: [K, N, N]
        out = 0.0
        for k in range(self.K):
            T_k_at = cheb[k].unsqueeze(0) * spatial_attention          # [B, N, N]
            rhs = torch.einsum("bnm,bnft->bmft", T_k_at, x)            # == T_k_at^T @ x
            out = out + torch.einsum("bmft,fo->bmot", rhs, self.Theta[k])
        return F.relu(out)                                             # [B, N, F_out, T]


class ASTGCNBlock(nn.Module):
    def __init__(
        self, in_channels: int, K: int, nb_chev_filter: int, nb_time_filter: int,
        time_strides: int, n_nodes: int, n_steps: int,
    ) -> None:
        super().__init__()
        self.TAt = TemporalAttention(in_channels, n_nodes, n_steps)
        self.SAt = SpatialAttention(in_channels, n_nodes, n_steps)
        self.cheb_conv_SAt = ChebConvWithSAt(K, in_channels, nb_chev_filter)
        self.time_conv = nn.Conv2d(
            nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1)
        )
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x: torch.Tensor, cheb: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F_in, T] -> [B, N, nb_time_filter, T/stride]
        B, N, Fi, T = x.shape
        temporal_at = self.TAt(x)                                                  # [B, T, T]
        x_tat = torch.matmul(x.reshape(B, -1, T), temporal_at).reshape(B, N, Fi, T)
        spatial_at = self.SAt(x_tat)                                               # [B, N, N]
        spatial_gcn = self.cheb_conv_SAt(x, spatial_at, cheb)                      # [B, N, F, T]
        time_out = self.time_conv(spatial_gcn.permute(0, 2, 1, 3))                 # [B, F, N, T']
        res = self.residual_conv(x.permute(0, 2, 1, 3))                            # [B, F, N, T']
        h = self.ln(F.relu(res + time_out).permute(0, 3, 2, 1))                    # [B, T', N, F]
        return h.permute(0, 2, 3, 1)                                               # [B, N, F, T']


class ASTGCNNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        pred_len: int,
        n_nodes: int,
        seq_len: int,
        cheb: torch.Tensor,
        *,
        n_blocks: int = 2,
        K: int = 3,
        nb_chev_filter: int = 32,
        nb_time_filter: int = 32,
        time_strides: int = 1,
    ) -> None:
        super().__init__()
        self.pred_len, self.out_dim = int(pred_len), int(out_dim)
        self.register_buffer("cheb", cheb.float())                     # [K, N, N]
        blocks = [ASTGCNBlock(input_dim, K, nb_chev_filter, nb_time_filter, time_strides, n_nodes, seq_len)]
        blocks += [
            ASTGCNBlock(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, n_nodes, seq_len // time_strides)
            for _ in range(int(n_blocks) - 1)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.final_conv = nn.Conv2d(
            seq_len // time_strides, self.pred_len * self.out_dim, kernel_size=(1, nb_time_filter)
        )
        for p in self.parameters():                                    # reference make_model init
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N, Dx] -> [B, pred_len, N, Dy]
        h = x.permute(0, 2, 3, 1)                                      # [B, N, Dx, L]
        for blk in self.blocks:
            h = blk(h, self.cheb)
        out = self.final_conv(h.permute(0, 3, 1, 2))[..., -1]          # [B, H*Dy, N]
        B, _, N = out.shape
        return out.view(B, self.pred_len, self.out_dim, N).permute(0, 1, 3, 2)


@register("astgcn")
class ASTGCNForecaster(GNNForecasterBase):
    """ASTGCN forecaster (port — spatial/temporal attention + attention-modulated Chebyshev GCN)."""

    name: str = "astgcn"
    n_blocks: int = 2
    cheb_k: int = 3
    nb_chev_filter: int = 32
    nb_time_filter: int = 32
    time_strides: int = 1

    def _build_net(self, bundle, n_nodes, *, A_norm, device):
        A = self._A_raw.to_dense()
        cheb = torch.stack(cheb_polynomials_from_adj(A, int(self.cheb_k)))
        return ASTGCNNet(
            input_dim=len(bundle.x_cols),
            out_dim=len(bundle.y_cols),
            pred_len=int(bundle.raw.spec.pred_len),
            n_nodes=n_nodes,
            seq_len=int(bundle.raw.spec.seq_len),
            cheb=cheb,
            n_blocks=int(self.n_blocks),
            K=int(self.cheb_k),
            nb_chev_filter=int(self.nb_chev_filter),
            nb_time_filter=int(self.nb_time_filter),
            time_strides=int(self.time_strides),
        )

    def _graph_forward(self, net, x):
        return net(x)
