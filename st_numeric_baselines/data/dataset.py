
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .io import AlignedData
from .windowing import WindowSpec


@dataclass(frozen=True)
class SampleMeta:
    zipcode: str
    t0: int
    t_pred_start: int


class WindowDataset(Dataset):
    def __init__(
        self,
        aligned: AlignedData,
        *,
        x_cols: Sequence[str],
        y_cols: Sequence[str],
        indices: Sequence[Tuple[int, int]],
        spec: WindowSpec,
    ) -> None:
        super().__init__()
        self.aligned = aligned
        self.indices = list(indices)
        self.spec = spec

        name_to_idx = {name: i for i, name in enumerate(aligned.schema.continuous_cols)}
        self.x_idx = [name_to_idx[c] for c in x_cols]
        self.y_idx = [name_to_idx[c] for c in y_cols]

        self.x_cols = list(x_cols)
        self.y_cols = list(y_cols)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, object]:
        zi, t0 = self.indices[i]
        seq_len = self.spec.seq_len
        label_len = self.spec.label_len
        pred_len = self.spec.pred_len

        t_pred_start = t0 + seq_len
        r_begin = t_pred_start - label_len
        r_end = t_pred_start + pred_len

        x = self.aligned.values[zi, t0 : t0 + seq_len, :][:, self.x_idx]
        y = self.aligned.values[zi, r_begin:r_end, :][:, self.y_idx]

        x_mark = self.aligned.time_marks[t0 : t0 + seq_len, :]
        y_mark = self.aligned.time_marks[r_begin:r_end, :]

        # to torch
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        xm_t = torch.tensor(x_mark, dtype=torch.float32)
        ym_t = torch.tensor(y_mark, dtype=torch.float32)

        meta = SampleMeta(zipcode=self.aligned.zipcodes[zi], t0=t0, t_pred_start=t_pred_start)

        return {"x": x_t, "y": y_t, "x_mark": xm_t, "y_mark": ym_t, "meta": meta}
