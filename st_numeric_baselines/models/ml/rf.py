from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple
import os

import numpy as np
import torch

from sklearn.ensemble import RandomForestRegressor

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.registry import register
from .lag_features import extract_last_n_torch


def _subsample_indices(
    indices: Sequence[Tuple[int, int]],
    *,
    max_samples: Optional[int],
    random_state: int,
) -> Sequence[Tuple[int, int]]:
    if max_samples is None:
        return indices
    n = len(indices)
    if max_samples >= n:
        return indices
    rng = np.random.RandomState(int(random_state))
    sel = rng.choice(n, size=int(max_samples), replace=False)
    return [indices[i] for i in sel]


def build_flat_lagged_xy_select(
    *,
    values: np.ndarray,                    # [Z, T, D]
    indices: Sequence[Tuple[int, int]],    # (zi, t0)
    seq_len: int,
    pred_len: int,
    x_idx: Sequence[int],
    y_idx: Sequence[int],
    max_samples: Optional[int],
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if values.ndim != 3:
        raise ValueError("values must have shape [Z, T, D]")
    Z, T, D = values.shape
    seq_len = int(seq_len)
    pred_len = int(pred_len)

    x_idx = list(map(int, list(x_idx)))
    y_idx = list(map(int, list(y_idx)))
    Dx = len(x_idx)
    Dy = len(y_idx)
    if Dx <= 0 or Dy <= 0:
        raise ValueError(f"Empty x_idx/y_idx: Dx={Dx}, Dy={Dy}")

    idx_use = _subsample_indices(indices, max_samples=max_samples, random_state=random_state)
    N = len(idx_use)

    X = np.empty((N, seq_len * Dx), dtype=np.float32)
    Y = np.empty((N, pred_len * Dy), dtype=np.float32)

    for i, (zi, t0) in enumerate(idx_use):
        if not (0 <= zi < Z):
            raise ValueError(f"Bad zi={zi} (Z={Z})")
        t0 = int(t0)
        t1 = t0 + seq_len
        t2 = t1 + pred_len
        if t0 < 0 or t2 > T:
            raise ValueError(f"Bad window (t0={t0}, t2={t2}) with T={T}")

        x_win = values[zi, t0:t1, :][:, x_idx]   # [seq_len, Dx]
        y_win = values[zi, t1:t2, :][:, y_idx]   # [pred_len, Dy]
        if not np.isfinite(x_win).all() or not np.isfinite(y_win).all():
            raise ValueError("RF received NaN/Inf in training windows. Check impute/transforms.")

        X[i, :] = x_win.reshape(-1).astype(np.float32, copy=False)
        Y[i, :] = y_win.reshape(-1).astype(np.float32, copy=False)

    return X, Y


@register("rf")
class RandomForestForecaster(BaseForecaster):
    name: str = "rf"

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 0,
        max_train_samples: Optional[int] = None,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.max_train_samples = None if max_train_samples is None else int(max_train_samples)
        self.n_jobs = int(n_jobs)

        self._model: Optional[RandomForestRegressor] = None
        self._dx: Optional[int] = None
        self._dy: Optional[int] = None

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        debug = os.environ.get("HOUSETS_DEBUG_RF", "0") not in ("", "0", "false", "no")

        orig_dim = int(bundle.raw.aligned.values.shape[-1])
        proc_dim = int(bundle.aligned_proc.values.shape[-1])
        if proc_dim != orig_dim:
            raise ValueError(
                "rf expects NO PCA (proc_dim must equal orig_dim). "
                "Disable transforms.pca for rf, or use model rf_pca for PCA baseline."
            )

        seq_len = int(bundle.raw.spec.seq_len)
        pred_len = int(bundle.raw.spec.pred_len)

        ds = bundle.datasets["train"]
        x_idx = getattr(ds, "x_idx", None)
        y_idx = getattr(ds, "y_idx", None)
        if x_idx is None or y_idx is None:
            raise RuntimeError("Train dataset must expose x_idx/y_idx (WindowDataset does).")

        x_idx = list(map(int, list(x_idx)))
        y_idx = list(map(int, list(y_idx)))
        Dx = len(x_idx)
        Dy = len(y_idx)

        if debug:
            print(
                "[rf.fit] (non-PCA) "
                f"window=(seq_len={seq_len}, pred_len={pred_len}) "
                f"orig_dim={orig_dim} proc_dim={proc_dim} Dx={Dx} Dy={Dy} "
                f"x_idx_head={x_idx[:8]} y_idx_head={y_idx[:8]} "
                f"n_train_windows={len(ds)}",
                flush=True,
            )

        X, Y = build_flat_lagged_xy_select(
            values=bundle.aligned_proc.values,
            indices=ds.indices,  # type: ignore[attr-defined]
            seq_len=seq_len,
            pred_len=pred_len,
            x_idx=x_idx,
            y_idx=y_idx,
            max_samples=self.max_train_samples,
            random_state=self.random_state,
        )

        if debug:
            print(f"[rf.fit] X.shape={X.shape} Y.shape={Y.shape}", flush=True)

        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        model.fit(X, Y)

        self._model = model
        self._dx = Dx
        self._dy = Dy

    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self._model is None or self._dx is None or self._dy is None:
            raise RuntimeError("RandomForestForecaster must be fit() before predict_batch()")

        x: torch.Tensor = batch["x"]  # [B, L, Dx]
        x_mask: Optional[torch.Tensor] = batch.get("x_mask", None)

        if device is not None:
            x = x.to(device)
            if x_mask is not None:
                x_mask = x_mask.to(device)

        seq_len = int(bundle.raw.spec.seq_len)
        pred_len = int(bundle.raw.spec.pred_len)
        Dx = int(self._dx)
        Dy = int(self._dy)

        if x.ndim != 3:
            raise ValueError(f"Expected batch['x'] to be [B,L,D], got {tuple(x.shape)}")
        if int(x.shape[-1]) != Dx:
            raise ValueError(f"RF Dx mismatch: batch Dx={int(x.shape[-1])}, trained Dx={Dx}")

        ctx = extract_last_n_torch(x, x_mask=x_mask, n=seq_len)  # [B, seq_len, Dx]
        Xb = ctx.detach().cpu().numpy().astype(np.float32).reshape(ctx.shape[0], -1)  # [B, seq_len*Dx]

        Yh = self._model.predict(Xb).astype(np.float32)  # [B, pred_len*Dy] or [B,]
        if Yh.ndim == 1:
            Yh = Yh.reshape(-1, 1)

        expected = int(pred_len) * int(Dy)
        if int(Yh.shape[1]) != expected:
            raise ValueError(f"RF output dim mismatch: got {Yh.shape[1]}, expected {expected}")

        Yh = Yh.reshape(ctx.shape[0], pred_len, Dy)
        y_hat = torch.from_numpy(Yh)
        if device is not None:
            y_hat = y_hat.to(device)
        return y_hat


@register("rf_pca")
class RandomForestPCAForecaster(BaseForecaster):
    name: str = "rf_pca"

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 0,
        max_train_samples: Optional[int] = None,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.max_train_samples = None if max_train_samples is None else int(max_train_samples)
        self.n_jobs = int(n_jobs)

        self._model: Optional[RandomForestRegressor] = None
        self._k: Optional[int] = None

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        debug = os.environ.get("HOUSETS_DEBUG_RF", "0") not in ("", "0", "false", "no")

        orig_dim = int(bundle.raw.aligned.values.shape[-1])
        proc_dim = int(bundle.aligned_proc.values.shape[-1])
        if proc_dim == orig_dim:
            raise ValueError("rf_pca expects PCA-reduced bundle (enable transforms.pca)")

        seq_len = int(bundle.raw.spec.seq_len)
        pred_len = int(bundle.raw.spec.pred_len)
        K = int(proc_dim)

        ds = bundle.datasets["train"]
        X, Y = build_flat_lagged_xy_select(
            values=bundle.aligned_proc.values,
            indices=ds.indices,  # type: ignore[attr-defined]
            seq_len=seq_len,
            pred_len=pred_len,
            x_idx=list(range(K)),
            y_idx=list(range(K)),
            max_samples=self.max_train_samples,
            random_state=self.random_state,
        )

        if debug:
            print(
                "[rf_pca.fit] "
                f"window=(seq_len={seq_len}, pred_len={pred_len}) K={K} "
                f"X.shape={X.shape} Y.shape={Y.shape}",
                flush=True,
            )

        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        model.fit(X, Y)

        self._model = model
        self._k = K

    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self._model is None or self._k is None:
            raise RuntimeError("RandomForestPCAForecaster must be fit() before predict_batch()")

        x: torch.Tensor = batch["x"]  # [B, L, K]
        x_mask: Optional[torch.Tensor] = batch.get("x_mask", None)

        if device is not None:
            x = x.to(device)
            if x_mask is not None:
                x_mask = x_mask.to(device)

        seq_len = int(bundle.raw.spec.seq_len)
        pred_len = int(bundle.raw.spec.pred_len)
        K = int(self._k)

        ctx = extract_last_n_torch(x, x_mask=x_mask, n=seq_len)  # [B, seq_len, K]
        Xb = ctx.detach().cpu().numpy().astype(np.float32).reshape(ctx.shape[0], -1)

        Yh = self._model.predict(Xb).astype(np.float32)  # [B, pred_len*K] or [B,]
        if Yh.ndim == 1:
            Yh = Yh.reshape(-1, 1)

        expected = int(pred_len) * int(K)
        if int(Yh.shape[1]) != expected:
            raise ValueError(f"rf_pca output dim mismatch: got {Yh.shape[1]}, expected {expected}")

        Yh = Yh.reshape(ctx.shape[0], pred_len, K)
        y_hat = torch.from_numpy(Yh)
        if device is not None:
            y_hat = y_hat.to(device)
        return y_hat
