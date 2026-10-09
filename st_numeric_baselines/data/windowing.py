
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class WindowSpec:
    seq_len: int
    label_len: int
    pred_len: int
    test_stride: int = 1


def make_window_spec(
    seq_len: int,
    pred_len: int,
    label_len: Optional[int] = None,
    test_stride: int = 1,
) -> WindowSpec:
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError("seq_len and pred_len must be positive")
    if label_len is None:
        label_len = max(1, seq_len // 2)
    if not (0 <= label_len <= seq_len):
        raise ValueError("label_len must be in [0, seq_len]")
    if test_stride <= 0:
        raise ValueError("test_stride must be positive")
    return WindowSpec(seq_len=seq_len, label_len=label_len, pred_len=pred_len, test_stride=test_stride)


def window_label(spec: WindowSpec) -> str:
    """Short label for run-dir naming / reporting, e.g. ``w12_h6``."""
    return f"w{spec.seq_len}_h{spec.pred_len}"


def generate_window_indices(
    *,
    values: np.ndarray,
    split_range: Tuple[int, int],
    split_start_for_targets: int,
    x_idx: Sequence[int],
    y_idx: Sequence[int],
    spec: WindowSpec,
    allow_history: bool = True,
    require_finite: bool = True,
    stride: int = 1,
) -> List[Tuple[int, int]]:
    Z, T, _ = values.shape
    start, end = split_range
    if not (0 <= start < end <= T):
        raise ValueError(f"Invalid split_range {split_range} for T={T}")
    if stride <= 0:
        raise ValueError("stride must be positive")

    seq_len = spec.seq_len
    pred_len = spec.pred_len
    label_len = spec.label_len

    # t0 search range
    if allow_history:
        t0_min = max(0, start - seq_len)
    else:
        t0_min = start
    t0_max = end - (seq_len + pred_len)
    if t0_max < t0_min:
        return []

    idx_list: List[Tuple[int, int]] = []

    for zi in range(Z):
        for t0 in range(t0_min, t0_max + 1, stride):
            t_pred_start = t0 + seq_len
            t_pred_end = t_pred_start + pred_len

            # forecast window must be within split_range and start within split targets
            if t_pred_start < split_start_for_targets:
                continue
            if t_pred_end > end:
                continue

            x_slice = values[zi, t0 : t0 + seq_len, :][:, x_idx]
            r_begin = t_pred_start - label_len
            r_end = t_pred_start + pred_len
            y_slice = values[zi, r_begin:r_end, :][:, y_idx]

            if require_finite:
                if np.isnan(x_slice).any() or np.isnan(y_slice).any():
                    continue

            idx_list.append((zi, t0))

    return idx_list
