from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple
import os

import numpy as np
import torch

from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.registry import register
from st_numeric_baselines.models.ml.lag_features import extract_last_n_torch


def _subsample_indices(
    indices: Sequence[Tuple[int, int]],
    *,
    max_samples: Optional[int],
    random_state: int,
) -> Sequence[Tuple[int, int]]:
    if max_samples is None or max_samples <= 0:
        return indices
    n = len(indices)
    if max_samples >= n:
        return indices
    rng = np.random.RandomState(int(random_state))
    sel = rng.choice(n, size=int(max_samples), replace=False)
    return [indices[i] for i in sel]


def build_flat_lagged_xy_select(
    *,
    values: np.ndarray,                    # [Z,T,D]
    indices: Sequence[Tuple[int, int]],    # (zi, t0)
    seq_len: int,
    pred_len: int,
    x_idx: Sequence[int],
    y_idx: Sequence[int],
    max_samples: Optional[int],
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if values.ndim != 3:
        raise ValueError("values must have shape [Z,T,D]")
    Z, T, D = values.shape

    seq_len = int(seq_len)
    pred_len = int(pred_len)
    if seq_len <= 0 or pred_len <= 0:
        raise ValueError("seq_len and pred_len must be positive")

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
            raise ValueError(f"Bad window: (zi={zi}, t0={t0}) with T={T}")

        x_win = values[zi, t0:t1, :][:, x_idx]  # [seq_len, Dx]
        y_win = values[zi, t1:t2, :][:, y_idx]  # [pred_len, Dy]

        if not np.isfinite(x_win).all() or not np.isfinite(y_win).all():
            raise ValueError("ARDL received NaN/Inf. Check impute/transforms.")

        X[i, :] = x_win.reshape(-1).astype(np.float32, copy=False)
        Y[i, :] = y_win.reshape(-1).astype(np.float32, copy=False)

    return X, Y


@register("ardl")
class ARDLForecaster(BaseForecaster):

    name: str = "ardl"

    def __init__(
        self,
        alpha: float = 1.0,             
        fit_intercept: bool = True,
        max_train_samples: Optional[int] = None,
        random_state: int = 0,
        use_pca: bool = False,          
        pca_n_components: int = 64,
        pca_whiten: bool = False,
    ) -> None:
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.max_train_samples = None if max_train_samples is None else int(max_train_samples)
        self.random_state = int(random_state)

        self.use_pca = bool(use_pca)
        self.pca_n_components = int(pca_n_components)
        self.pca_whiten = bool(pca_whiten)

        self._model: Optional[Ridge] = None
        self._pca: Optional[PCA] = None
        self._dx: Optional[int] = None
        self._dy: Optional[int] = None

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        debug = os.environ.get("HOUSETS_DEBUG_ARDL", "0") not in ("", "0", "false", "no")
        orig_dim = int(bundle.raw.aligned.values.shape[-1])
        proc_dim = int(bundle.aligned_proc.values.shape[-1])
        if proc_dim != orig_dim:
            raise ValueError(
                "ARDL expects no global PCA in pipeline (proc_dim must equal orig_dim). "
                "Disable transforms.pca for MS/S when using ardl. "
                "If you want PCA, set ardl.use_pca=true (input-only PCA)."
            )

        seq_len = int(bundle.raw.spec.seq_len)
        pred_len = int(bundle.raw.spec.pred_len)

        ds = bundle.datasets["train"]
        x_idx = getattr(ds, "x_idx", None)
        y_idx = getattr(ds, "y_idx", None)
        if x_idx is None or y_idx is None:
            raise RuntimeError("Train dataset must expose x_idx/y_idx")

        x_idx = list(map(int, list(x_idx)))
        y_idx = list(map(int, list(y_idx)))
        Dx = len(x_idx)
        Dy = len(y_idx)
        out_dim = int(pred_len) * int(Dy)

        if debug:
            print(
                "[ardl.fit] "
                f"window=(seq_len={seq_len}, pred_len={pred_len}) "
                f"Dx={Dx} Dy={Dy} out_dim={out_dim} "
                f"use_pca={self.use_pca} pca_n_components={self.pca_n_components} "
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

        if self.use_pca:
            n_comp = min(int(self.pca_n_components), int(X.shape[1]))
            if n_comp <= 0:
                raise ValueError("pca_n_components must be positive")
            pca = PCA(n_components=n_comp, whiten=bool(self.pca_whiten), random_state=int(self.random_state))
            X2 = pca.fit_transform(X.astype(np.float64)).astype(np.float32)
            self._pca = pca
            X = X2
            if debug:
                print(f"[ardl.fit] PCA(X): {X2.shape} (from {X.shape})", flush=True)

        model = Ridge(alpha=float(self.alpha), fit_intercept=bool(self.fit_intercept))
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
            raise RuntimeError("ARDLForecaster must be fit() before predict_batch()")

        x: torch.Tensor = batch["x"]  # [B,L,Dx]
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
            raise ValueError(f"ARDL Dx mismatch: batch Dx={int(x.shape[-1])}, trained Dx={Dx}")

        ctx = extract_last_n_torch(x, x_mask=x_mask, n=seq_len)  # [B,seq_len,Dx]
        Xb = ctx.detach().cpu().numpy().astype(np.float32).reshape(ctx.shape[0], -1)  # [B, seq_len*Dx]

        if self._pca is not None:
            Xb = self._pca.transform(Xb.astype(np.float64)).astype(np.float32)

        Yh = self._model.predict(Xb).astype(np.float32)  # [B, pred_len*Dy] or [B,]
        if Yh.ndim == 1:
            Yh = Yh.reshape(-1, 1)

        expected = int(pred_len) * int(Dy)
        if int(Yh.shape[1]) != expected:
            raise ValueError(f"ARDL output dim mismatch: got {Yh.shape[1]}, expected {expected}")

        Yh = Yh.reshape(ctx.shape[0], pred_len, Dy)
        y_hat = torch.from_numpy(Yh)
        if device is not None:
            y_hat = y_hat.to(device)
        return y_hat
