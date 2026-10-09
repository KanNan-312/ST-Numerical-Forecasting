from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass

    return np.asarray(x)


def logrmse(y_true_log: Any, y_pred_log: Any) -> float:
    yt = _to_numpy(y_true_log).astype(np.float64)
    yp = _to_numpy(y_pred_log).astype(np.float64)
    mse = np.mean((yp - yt) ** 2)
    return float(np.sqrt(mse))


def mape(y_true: Any, y_pred: Any, *, eps: float = 1e-8) -> float:
    yt = _to_numpy(y_true).astype(np.float64)
    yp = _to_numpy(y_pred).astype(np.float64)

    denom = np.maximum(np.abs(yt), eps)
    val = np.mean(np.abs(yp - yt) / denom)
    return float(val)
