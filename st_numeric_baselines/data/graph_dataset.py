"""Graph-structured windowed dataset for spatiotemporal GNN models.

Key difference from WindowDataset
----------------------------------
``WindowDataset`` (used by DL models) generates one sample per *(ZIP, t₀)* pair.
A batch has shape ``[B, L, Dx]``; each row is one ZIP × one time window, so ZIPs
are treated as **independent** samples with no spatial coupling.

``GraphWindowDataset`` (used by GNN models) generates one sample per *t₀* only
— ALL N ZIP nodes are included in every item.  A batch has shape
``[B, L, N, Dx]``; the model then performs message-passing across N, capturing
geographic dependencies.  After the GNN forward pass the prediction is reshaped
from ``[B, H, N, Dy]`` → ``[B*N, H, Dy]`` so the standard
:class:`~st_numeric_baselines.metrics.evaluator.StreamingEvaluator` receives the same
``(n_samples, horizon, features)`` format it expects.
"""
from __future__ import annotations

from typing import List

import torch
from torch.utils.data import Dataset

from st_numeric_baselines.bundles.datatypes import ProcBundle


class GraphWindowDataset(Dataset):
    """Full-graph time-windowed dataset for GNN-based forecasters.

    Each item covers **all N nodes** for a single time window::

        "x": [L, N, Dx]       — encoder input  (all ZIPs, seq_len steps)
        "y": [pred_len, N, Dy] — forecast target (all ZIPs, pred_len steps)
        "x_mark": [L, Dm]      — time marks for the encoder window (same
                                 regardless of node — sliced from
                                 ``bundle.aligned_proc.time_marks``, the same
                                 array :class:`~st_numeric_baselines.data.dataset.WindowDataset`
                                 already uses for DL models)

    After :func:`graph_collate`, a batch has::

        "x": [B, L, N, Dx]
        "y": [B*N, pred_len, Dy]  — flattened for StreamingEvaluator
        "x_mark": [B, L, Dm]
    """

    def __init__(self, bundle: ProcBundle, split: str) -> None:
        super().__init__()
        split = split.lower()

        values = bundle.aligned_proc.values  # [N, T, D]
        proc_names = list(bundle.aligned_proc.schema.continuous_cols)
        name_to_idx = {n: i for i, n in enumerate(proc_names)}

        self._x_idx = [name_to_idx[c] for c in bundle.x_cols]
        self._y_idx = [name_to_idx[c] for c in bundle.y_cols]
        self._values = torch.tensor(values, dtype=torch.float32)  # [N, T, D]
        self._time_marks = torch.tensor(bundle.aligned_proc.time_marks, dtype=torch.float32)  # [T, Dm]

        split_range = bundle.raw.split.range(split)
        seq_len = bundle.raw.spec.seq_len
        pred_len = bundle.raw.spec.pred_len

        # For val/test, mirror DL's allow_history=True: the encoder window may
        # reach back into the prior split, but the prediction must start within
        # this split.  For train, stay strict (no look-back needed).
        t0_end = split_range[1] - seq_len - pred_len
        if split == "train":
            t0_start = split_range[0]
        else:
            t0_start = max(0, split_range[0] - seq_len)

        # test_stride > 1 strides through non-overlapping test anchors instead
        # of every consecutive t0, to cut evaluation cost on expensive models.
        stride = bundle.raw.spec.test_stride if split == "test" else 1
        self._time_anchors: List[int] = list(range(t0_start, t0_end + 1, stride))
        self.time_anchors = self._time_anchors  # public alias — callers building
        # per-instance metadata (e.g. the case-library builder) index t0 by dataset position
        self._seq_len = seq_len
        self._pred_len = pred_len

    def __len__(self) -> int:
        return len(self._time_anchors)

    def __getitem__(self, idx: int):
        t = self._time_anchors[idx]
        L, H = self._seq_len, self._pred_len

        # [N, L, Dx] → [L, N, Dx]
        x = self._values[:, t : t + L, :][:, :, self._x_idx].permute(1, 0, 2)
        # [N, H, Dy] → [H, N, Dy]
        y = self._values[:, t + L : t + L + H, :][:, :, self._y_idx].permute(1, 0, 2)
        x_mark = self._time_marks[t : t + L, :]  # [L, Dm]

        return {"x": x, "y": y, "x_mark": x_mark}


def graph_collate(batch):
    """Stack a list of graph items into a batch, flattening nodes into batch axis.

    Input items have::

        "x": [L, N, Dx]       per item
        "y": [pred_len, N, Dy] per item

    Output batch::

        "x": [B, L, N, Dx]
        "y": [B*N, pred_len, Dy]  — node axis merged into batch for evaluator
    """
    x = torch.stack([b["x"] for b in batch])  # [B, L, N, Dx]
    y = torch.stack([b["y"] for b in batch])   # [B, H, N, Dy]
    x_mark = torch.stack([b["x_mark"] for b in batch])  # [B, L, Dm]
    B, H, N, Dy = y.shape
    y_flat = y.permute(0, 2, 1, 3).reshape(B * N, H, Dy)  # [B*N, H, Dy]
    return {"x": x, "y": y_flat, "x_mark": x_mark}
