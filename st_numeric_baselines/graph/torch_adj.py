from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def sparse_adj(
    src: np.ndarray,
    dst: np.ndarray,
    n_nodes: int,
    *,
    weight: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    dev = device if device is not None else torch.device("cpu")
    src_t = torch.as_tensor(src, dtype=torch.long, device=dev)
    dst_t = torch.as_tensor(dst, dtype=torch.long, device=dev)
    idx = torch.stack([src_t, dst_t], dim=0)

    if weight is None:
        val = torch.ones(src_t.numel(), dtype=dtype, device=dev)
    else:
        val = torch.as_tensor(weight, dtype=dtype, device=dev)

    A = torch.sparse_coo_tensor(idx, val, size=(n_nodes, n_nodes)).coalesce()
    return A


def normalize_adj_sym(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    A = A.coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv_sqrt = deg.clamp_min(eps).pow(-0.5)
    idx = A.indices()
    val = A.values()
    val = val * deg_inv_sqrt[idx[0]] * deg_inv_sqrt[idx[1]]
    return torch.sparse_coo_tensor(idx, val, size=A.shape).coalesce()


def normalize_adj_random_walk(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-normalized random-walk adjacency: ``D^-1 A``.

    Used for diffusion convolution (e.g. DCRNN/D2STGNN), where the forward and
    backward diffusion processes are the row-normalized walk on ``A`` and on
    ``A.T`` respectively — pass the transpose in separately for the backward
    direction.
    """
    A = A.coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense()
    deg_inv = deg.clamp_min(eps).pow(-1.0)
    idx = A.indices()
    val = A.values() * deg_inv[idx[0]]
    return torch.sparse_coo_tensor(idx, val, size=A.shape).coalesce()


def spmm_nt(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # X_flat: [N, B*T*C]
    B, T, N, C = X.shape
    X_flat = X.permute(2, 0, 1, 3).reshape(N, B * T * C)
    Y_flat = torch.sparse.mm(A, X_flat)
    Y = Y_flat.reshape(N, B, T, C).permute(1, 2, 0, 3).contiguous()
    return Y


def spmm_nct(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    B, C, N, T = X.shape
    X_flat = X.permute(2, 0, 1, 3).reshape(N, B * C * T)
    Y_flat = torch.sparse.mm(A, X_flat)
    Y = Y_flat.reshape(N, B, C, T).permute(1, 2, 0, 3).contiguous()
    return Y
