from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import torch


def build_flat_lagged_xy(
    *,
    values: np.ndarray,
    indices: Sequence[Tuple[int, int]],
    seq_len: int,
    pred_len: int,
    max_samples: Optional[int] = None,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if values.ndim != 3:
        raise ValueError("values must have shape [Z, T, K]")
    Z, T, K = values.shape
    seq_len = int(seq_len)
    pred_len = int(pred_len)
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError("seq_len and pred_len must be positive")

    idx = list(indices)
    if max_samples is not None and max_samples > 0 and len(idx) > max_samples:
        rng = np.random.default_rng(int(random_state))
        sel = rng.choice(len(idx), size=int(max_samples), replace=False)
        idx = [idx[i] for i in sel]

    N = len(idx)
    X = np.zeros((N, seq_len * K), dtype=np.float32)
    Y = np.zeros((N, pred_len * K), dtype=np.float32)

    for i, (zi, t0) in enumerate(idx):
        x = values[zi, t0 : t0 + seq_len, :]  # [seq_len,K]
        y = values[zi, t0 + seq_len : t0 + seq_len + pred_len, :]  # [pred_len,K]
        if x.shape[0] != seq_len or y.shape[0] != pred_len:
            raise ValueError(f"Bad window at i={i}, zi={zi}, t0={t0}: x={x.shape}, y={y.shape}")
        X[i, :] = x.reshape(-1)
        Y[i, :] = y.reshape(-1)

    return X, Y


def extract_last_n_torch(
    x: torch.Tensor,
    *,
    x_mask: Optional[torch.Tensor],
    n: int,
) -> torch.Tensor:
    B, L, D = x.shape
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if n > L:
        raise ValueError(f"n={n} cannot exceed L={L}")

    if x_mask is None:
        return x[:, -n:, :]

    pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
    masked_pos = pos * (x_mask > 0).to(pos.dtype)
    idx_last = masked_pos.max(dim=1).values.long()

    idx_start = idx_last - (n - 1)
    idx_start = torch.clamp(idx_start, min=0, max=L - n)

    gather_idx = idx_start.unsqueeze(1) + torch.arange(n, device=x.device).unsqueeze(0)  # [B,n]
    gather_idx = gather_idx.unsqueeze(-1).expand(B, n, D)
    return x.gather(1, gather_idx)
