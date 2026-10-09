from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


def fit_affine_calibrator(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    if y_pred.shape != y_true.shape:
        raise ValueError(f"shape mismatch: y_pred={y_pred.shape}, y_true={y_true.shape}")
    if y_pred.ndim != 3:
        raise ValueError(f"expected 3D arrays [N,H,Dy], got ndim={y_pred.ndim}")

    N, H, Dy = y_pred.shape
    scale = np.zeros((H, Dy), dtype=np.float32)
    bias = np.zeros((H, Dy), dtype=np.float32)

    for h in range(H):
        for d in range(Dy):
            p = y_pred[:, h, d].astype(np.float64, copy=False)
            t = y_true[:, h, d].astype(np.float64, copy=False)
            mp = float(p.mean())
            mt = float(t.mean())
            vp = float(((p - mp) ** 2).mean())

            if vp < eps:
                # No variance in the predictor => only fit an intercept.
                scale[h, d] = 0.0
                bias[h, d] = np.float32(mt)
                continue

            cov = float(((p - mp) * (t - mt)).mean())
            a = cov / (vp + eps)
            b = mt - a * mp
            scale[h, d] = np.float32(a)
            bias[h, d] = np.float32(b)

    return scale, bias


@dataclass
class AffineCalibrator:
    scale: torch.Tensor  # [H, Dy]
    bias: torch.Tensor  # [H, Dy]

    @classmethod
    def from_numpy(
        cls,
        scale: np.ndarray,
        bias: np.ndarray,
        *,
        device: torch.device | None = None,
    ) -> "AffineCalibrator":
        s = torch.as_tensor(scale, dtype=torch.float32, device=device)
        b = torch.as_tensor(bias, dtype=torch.float32, device=device)
        return cls(scale=s, bias=b)

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 3:
            raise ValueError(f"expected y as [B,H,Dy], got {tuple(y.shape)}")
        if y.shape[1] != self.scale.shape[0] or y.shape[2] != self.scale.shape[1]:
            raise ValueError(
                f"calibrator expects [H,Dy]={tuple(self.scale.shape)}, got y={tuple(y.shape)}"
            )
        s = self.scale.to(device=y.device)
        b = self.bias.to(device=y.device)
        return y * s.unsqueeze(0) + b.unsqueeze(0)
