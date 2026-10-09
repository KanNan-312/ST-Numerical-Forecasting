from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .base import Transform, apply_on_last_dim


@dataclass
class LogTransform(Transform):
    mode: str = "log1p"
    eps: float = 1e-8

    def __init__(self, mode: str = "log1p", eps: float = 1e-8) -> None:
        super().__init__(name="log")
        self.mode = str(mode).lower()
        self.eps = float(eps)
        if self.mode not in ("log", "log1p"):
            raise ValueError("mode must be 'log' or 'log1p'")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def transform(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        if self.mode == "log":
            def _fn(a: np.ndarray) -> np.ndarray:
                return np.log(np.clip(a, self.eps, None))
        else:
            def _fn(a: np.ndarray) -> np.ndarray:
                return np.log1p(np.clip(a, 0.0, None))

        return apply_on_last_dim(x, idx, _fn)

    def _inverse(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        if self.mode == "log":
            def _fn(a: np.ndarray) -> np.ndarray:
                return np.exp(a)
        else:
            def _fn(a: np.ndarray) -> np.ndarray:
                a64 = np.asarray(a, dtype=np.float64)
                max_log = np.log(np.finfo(np.float64).max) 
                a64 = np.clip(a64, None, max_log - 1.0)   
                out = np.expm1(a64)
                return np.clip(out, 0.0, None)

        return apply_on_last_dim(x, idx, _fn)
