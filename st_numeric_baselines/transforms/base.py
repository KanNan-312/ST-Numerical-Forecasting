from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

import numpy as np


def _unique_sorted(idx: Sequence[int]) -> list[int]:
    out = sorted(set(int(i) for i in idx))
    if any(i < 0 for i in out):
        raise ValueError(f"Feature indices must be >=0, got: {out}")
    return out


def apply_on_last_dim(
    x: np.ndarray,
    idx: Optional[Sequence[int]],
    fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:

    if idx is None:
        return fn(x)

    idx2 = _unique_sorted(idx)
    if x.shape[-1] <= 0:
        raise ValueError("x must have a feature dimension")
    if idx2 and max(idx2) >= x.shape[-1]:
        raise IndexError(f"idx contains {max(idx2)} but x has D={x.shape[-1]}")

    y = x.copy()
    y[..., idx2] = fn(y[..., idx2])
    return y


class Transform(ABC):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> "Transform":
        self._fit(x, idx=idx)
        self._fitted = True
        return self

    def _fit(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> None:
        # default: stateless
        return None

    @abstractmethod
    def transform(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        raise NotImplementedError

    def inverse(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        # default: identity
        return self._inverse(x, idx=idx)

    def _inverse(self, x: np.ndarray, *, idx: Optional[Sequence[int]] = None) -> np.ndarray:
        return x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, fitted={self.fitted})"
