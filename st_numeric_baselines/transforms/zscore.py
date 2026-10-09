from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Literal, Optional, Sequence

import numpy as np

from .base import Transform, apply_on_last_dim


@dataclass
class ZScoreTransform(Transform):
    scope: Literal["global", "per_zip"] = "global"
    eps: float = 1e-6

    # fitted params
    _mean: Optional[np.ndarray] = None
    _std: Optional[np.ndarray] = None

    def __init__(self, scope: Literal["global", "per_zip"] = "global", eps: float = 1e-6) -> None:
        super().__init__(name="zscore")
        self.scope = str(scope).lower()
        self.eps = float(eps)
        if self.scope not in ("global", "per_zip"):
            raise ValueError("scope must be 'global' or 'per_zip'")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def _fit(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> None:
        x_sel = x if idx is None else x[..., list(idx)]
        
        if os.environ.get("HOUSETS_DEBUG_TRANSFORM_FIT", "0") not in ("", "0", "false", "no"):
            print(
                f"[zscore.fit] scope={self.scope} x.shape={x.shape} x_sel.shape={x_sel.shape} "
                f"idx={'None' if idx is None else len(list(idx))}",
                flush=True,
            )      
        

        if x_sel.ndim < 2:
            raise ValueError("Expected x to have at least 2 dims [.., D]")

        if self.scope == "global":
            flat = x_sel.reshape(-1, x_sel.shape[-1]).astype(np.float64)
            mean = flat.mean(axis=0)
            std = flat.std(axis=0)
            std = np.where(std < self.eps, 1.0, std)
            self._mean = mean.astype(np.float32)
            self._std = std.astype(np.float32)
        else:
            # x is expected to be [Z, T, D]
            if x_sel.ndim != 3:
                raise ValueError("per_zip zscore expects x with shape [Z, T, D]")
            mean = x_sel.mean(axis=1, keepdims=True).astype(np.float64)  # [Z,1,D]
            std = x_sel.std(axis=1, keepdims=True).astype(np.float64)    # [Z,1,D]
            std = np.where(std < self.eps, 1.0, std)
            self._mean = mean.astype(np.float32)
            self._std = std.astype(np.float32)

    def transform(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("ZScoreTransform must be fitted before transform()")

        if self.scope == "global":
            def _fn(a: np.ndarray) -> np.ndarray:
                return (a - self._mean) / self._std
            return apply_on_last_dim(x, idx, _fn)

        # per_zip: x must be [Z,T,D]
        if x.ndim != 3:
            raise ValueError("per_zip zscore expects x with shape [Z, T, D]")
        if idx is None:
            return (x - self._mean) / self._std

        # idx subset
        idx2 = list(idx)
        y = x.copy()
        y[..., idx2] = (y[..., idx2] - self._mean[..., idx2]) / self._std[..., idx2]
        return y

    def _inverse(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("ZScoreTransform must be fitted before inverse()")

        if self.scope == "global":
            def _fn(a: np.ndarray) -> np.ndarray:
                return a * self._std + self._mean
            return apply_on_last_dim(x, idx, _fn)

        if x.ndim != 3:
            raise ValueError("per_zip zscore expects x with shape [Z, T, D]")
        if idx is None:
            return x * self._std + self._mean

        idx2 = list(idx)
        y = x.copy()
        y[..., idx2] = y[..., idx2] * self._std[..., idx2] + self._mean[..., idx2]
        return y
