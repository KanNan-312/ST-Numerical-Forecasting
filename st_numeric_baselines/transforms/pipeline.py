from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .base import Transform
from .pca import PCATransform


@dataclass(frozen=True)
class StageSpec:
    transform: Transform
    # feature indices to apply the transform on.
    # None => all features.
    idx: Optional[Sequence[int]] = None


class TransformPipeline:
    def __init__(self, stages: Sequence[StageSpec]) -> None:
        self.stages = list(stages)
        self._fitted = False
        self._in_dim: Optional[int] = None
        self._out_dim: Optional[int] = None

        # basic validation: PCA must be last if present
        for i, st in enumerate(self.stages):
            if isinstance(st.transform, PCATransform) and i != len(self.stages) - 1:
                raise ValueError("PCATransform must be the last stage in the pipeline")

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def in_dim(self) -> int:
        if self._in_dim is None:
            raise RuntimeError("Pipeline not fitted")
        return self._in_dim

    @property
    def out_dim(self) -> int:
        if self._out_dim is None:
            raise RuntimeError("Pipeline not fitted")
        return self._out_dim

    def fit_transform(self, values: np.ndarray, *, train_range: Tuple[int, int]) -> np.ndarray:
        x = values
        if x.ndim != 3:
            raise ValueError("Expected values with shape [Z, T, D]")
        Z, T, D = x.shape
        self._in_dim = D

        t0, t1 = train_range
        if not (0 <= t0 < t1 <= T):
            raise ValueError(f"Invalid train_range={train_range} for T={T}")

        cur = x.astype(np.float32, copy=False)

        for st in self.stages:
            fit_slice = cur[:, t0:t1, :]
            st.transform.fit(fit_slice, idx=st.idx)
            cur = st.transform.transform(cur, idx=st.idx)

        self._fitted = True
        self._out_dim = int(cur.shape[-1])
        return cur

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TransformPipeline must be fitted before transform()")
        cur = values
        for st in self.stages:
            cur = st.transform.transform(cur, idx=st.idx)
        return cur

    def inverse(self, values: np.ndarray, *, keep_log: bool = False) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TransformPipeline must be fitted before inverse()")

        cur = values
        for st in reversed(self.stages):
            if keep_log and st.transform.name == "log":
                break
            cur = st.transform.inverse(cur, idx=st.idx)
        return cur

    def summary(self) -> str:
        parts: List[str] = []
        for st in self.stages:
            idx_desc = "all" if st.idx is None else f"{len(list(st.idx))} cols"
            parts.append(f"{st.transform.name}({idx_desc})")
        return " -> ".join(parts) if parts else "<empty pipeline>"
