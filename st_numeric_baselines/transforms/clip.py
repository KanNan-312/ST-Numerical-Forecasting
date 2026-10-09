from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from .base import Transform, apply_on_last_dim


@dataclass
class ClipTransform(Transform):
    method: Literal["quantile", "sigma", "absolute"] = "quantile"
    lower_q: float = 0.001
    upper_q: float = 0.999
    sigma_k: float = 5.0
    abs_lower: Optional[float] = None
    abs_upper: Optional[float] = None

    # fitted bounds: shape [d_sel]
    _lower: Optional[np.ndarray] = None
    _upper: Optional[np.ndarray] = None

    def __init__(
        self,
        *,
        method: Literal["quantile", "sigma", "absolute"] = "quantile",
        lower_q: float = 0.001,
        upper_q: float = 0.999,
        sigma_k: float = 5.0,
        abs_lower: Optional[float] = None,
        abs_upper: Optional[float] = None,
    ) -> None:
        super().__init__(name="clip")
        self.method = method
        self.lower_q = float(lower_q)
        self.upper_q = float(upper_q)
        self.sigma_k = float(sigma_k)
        self.abs_lower = abs_lower
        self.abs_upper = abs_upper

        if self.method not in ("quantile", "sigma", "absolute"):
            raise ValueError("method must be one of: quantile, sigma, absolute")
        if not (0.0 <= self.lower_q < self.upper_q <= 1.0):
            raise ValueError("Require 0 <= lower_q < upper_q <= 1")
        if self.sigma_k <= 0:
            raise ValueError("sigma_k must be > 0")

    def _fit(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> None:
        # fit on selected features only
        x_sel = x if idx is None else x[..., list(idx)]
        flat = x_sel.reshape(-1, x_sel.shape[-1])

        if self.method == "quantile":
            lo = np.quantile(flat, self.lower_q, axis=0)
            hi = np.quantile(flat, self.upper_q, axis=0)
        elif self.method == "sigma":
            mu = flat.mean(axis=0)
            sd = flat.std(axis=0)
            lo = mu - self.sigma_k * sd
            hi = mu + self.sigma_k * sd
        else:  # absolute
            lo_val = -np.inf if self.abs_lower is None else float(self.abs_lower)
            hi_val = np.inf if self.abs_upper is None else float(self.abs_upper)
            lo = np.full((flat.shape[-1],), lo_val, dtype=np.float64)
            hi = np.full((flat.shape[-1],), hi_val, dtype=np.float64)

        # ensure finite ordering
        lo = np.minimum(lo, hi)
        self._lower = lo.astype(np.float32)
        self._upper = hi.astype(np.float32)

    def transform(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        if self._lower is None or self._upper is None:
            raise RuntimeError("ClipTransform must be fitted before transform()")

        def _fn(a: np.ndarray) -> np.ndarray:
            # broadcast bounds over leading dims
            return np.clip(a, self._lower, self._upper)

        return apply_on_last_dim(x, idx, _fn)

    def _inverse(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        # not invertible; treat as identity
        return x
