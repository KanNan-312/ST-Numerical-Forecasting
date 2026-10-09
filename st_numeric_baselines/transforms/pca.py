from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from sklearn.decomposition import PCA

from .base import Transform


@dataclass
class PCATransform(Transform):
    n_components: int = 16
    whiten: bool = False
    random_state: int = 0

    _pca: Optional[PCA] = None
    _in_dim: Optional[int] = None

    def __init__(self, n_components: int = 16, *, whiten: bool = False, random_state: int = 0) -> None:
        super().__init__(name="pca")
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.random_state = int(random_state)
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")

    def _fit(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> None:
        if idx is not None:
            raise ValueError("PCATransform currently supports idx=None only (apply PCA to all features)")
        if x.ndim != 3:
            raise ValueError("PCATransform expects x with shape [Z, T, D]")

        Z, T, D = x.shape
        self._in_dim = D
        if self.n_components > D:
            raise ValueError(f"n_components={self.n_components} cannot be > D={D}")

        flat = x.reshape(-1, D).astype(np.float64)
        if np.isnan(flat).any() or np.isinf(flat).any():
            raise ValueError("PCATransform fit received NaN/Inf values")

        pca = PCA(n_components=self.n_components, whiten=self.whiten, random_state=self.random_state)
        pca.fit(flat)
        self._pca = pca

    def transform(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        if idx is not None:
            raise ValueError("PCATransform currently supports idx=None only")
        if self._pca is None or self._in_dim is None:
            raise RuntimeError("PCATransform must be fitted before transform()")
        if x.ndim != 3:
            raise ValueError("PCATransform expects x with shape [Z, T, D]")
        if x.shape[-1] != self._in_dim:
            raise ValueError(f"Expected last dim D={self._in_dim}, got {x.shape[-1]}")

        Z, T, D = x.shape
        flat = x.reshape(-1, D).astype(np.float64)
        y = self._pca.transform(flat).astype(np.float32)
        return y.reshape(Z, T, self.n_components)

    def _inverse(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        if idx is not None:
            raise ValueError("PCATransform currently supports idx=None only")
        if self._pca is None or self._in_dim is None:
            raise RuntimeError("PCATransform must be fitted before inverse()")
        if x.ndim != 3:
            raise ValueError("PCATransform expects x with shape [Z, T, K]")
        if x.shape[-1] != self.n_components:
            raise ValueError(f"Expected last dim K={self.n_components}, got {x.shape[-1]}")

        Z, T, K = x.shape
        flat = x.reshape(-1, K).astype(np.float64)
        y = self._pca.inverse_transform(flat).astype(np.float32)
        return y.reshape(Z, T, self._in_dim)

    @property
    def in_dim(self) -> int:
        if self._in_dim is None:
            raise RuntimeError("PCATransform not fitted")
        return self._in_dim

    @property
    def out_dim(self) -> int:
        return self.n_components
