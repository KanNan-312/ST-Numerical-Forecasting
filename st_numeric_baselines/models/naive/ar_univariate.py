from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.registry import register


@dataclass
class _ARParams:
    intercept: float
    coefs: np.ndarray  # shape [p], order corresponds to lags [y_{t-1}, ..., y_{t-p}]


def _normalize_zip(z: object) -> str:
    s = str(z)
    if s.isdigit() and len(s) < 5:
        s = s.zfill(5)
    return s


def _extract_last_p(
    x: torch.Tensor,
    *,
    x_mask: Optional[torch.Tensor],
    target_pos: int,
    p: int,
) -> torch.Tensor:
    B, L, _ = x.shape
    if p <= 0:
        raise ValueError("p must be positive")
    if p > L:
        raise ValueError(f"p={p} cannot exceed sequence length L={L}")

    x_t = x[:, :, target_pos]  # [B, L]

    if x_mask is None:
        return x_t[:, -p:]

    pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)  # [B,L]
    masked_pos = pos * (x_mask > 0).to(pos.dtype)
    idx_last = masked_pos.max(dim=1).values.long()  # [B]

    idx_start = idx_last - (p - 1)
    idx_start = torch.clamp(idx_start, min=0, max=L - p)

    gather_idx = idx_start.unsqueeze(1) + torch.arange(p, device=x.device).unsqueeze(0)  # [B,p]
    return x_t.gather(1, gather_idx)  # [B,p]


@register("ar_univariate")
class ARUnivariateForecaster(BaseForecaster):

    name: str = "ar_univariate"

    def __init__(self, p: Optional[int] = None, ridge: float = 0.0) -> None:
        self.p = p
        self.ridge = float(ridge)
        self._params: Dict[str, _ARParams] = {}
        self._target_col: Optional[str] = None

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        # enforce univariate target
        if len(bundle.y_cols) != 1:
            raise ValueError(f"ARUnivariateForecaster expects Dy=1, got y_cols={bundle.y_cols}")
        if bundle.y_cols[0] != bundle.raw_target_col:
            raise ValueError(
                f"ARUnivariateForecaster expects y_cols to be the raw target '{bundle.raw_target_col}', "
                f"got y_cols={bundle.y_cols}"
            )

        self._target_col = bundle.raw_target_col

        # choose order p
        p = int(self.p) if self.p is not None else int(bundle.raw.spec.seq_len)
        if p <= 0:
            raise ValueError("AR order p must be positive")
        self.p = p

        # locate target index in processed aligned tensor
        proc_names = list(bundle.aligned_proc.schema.continuous_cols)
        try:
            target_idx = proc_names.index(bundle.raw_target_col)
        except ValueError as e:
            raise ValueError(
                f"Target '{bundle.raw_target_col}' not found in processed schema continuous_cols. "
                f"Got first cols={proc_names[:10]}"
            ) from e

        t0, t1 = bundle.raw.split.train
        values = bundle.aligned_proc.values  # [Z, T, Dproc]
        zipcodes = bundle.raw.aligned.zipcodes

        ridge = float(getattr(self, "ridge", 0.0) or 0.0)

        params: Dict[str, _ARParams] = {}

        for zi, zc in enumerate(zipcodes):
            key = _normalize_zip(zc)
            y = values[zi, t0:t1, target_idx].astype(np.float64, copy=False)  # [N]
            y = np.asarray(y)
            N = int(y.shape[0])

            # basic sanity
            if N <= 0 or not np.isfinite(y).all():
                params[key] = _ARParams(intercept=0.0, coefs=np.zeros((p,), dtype=np.float64))
                continue

            if N <= p:
                last = float(y[-1])
                params[key] = _ARParams(intercept=last, coefs=np.zeros((p,), dtype=np.float64))
                continue

            # Build lagged design matrix using sliding windows
            # windows shape: (N-p, p+1), row i is y[i : i+p+1] = [y_{t-p},...,y_{t-1},y_t]
            win = np.lib.stride_tricks.sliding_window_view(y, p + 1)
            y_tgt = win[:, -1]                       # y_t
            X = win[:, :-1][:, ::-1].copy()          # [y_{t-1},...,y_{t-p}]  shape (N-p, p)

            # Add intercept column
            X1 = np.concatenate([np.ones((X.shape[0], 1), dtype=np.float64), X], axis=1)  # (N-p, p+1)

            # Ridge solve
            try:
                A = X1.T @ X1
                b = X1.T @ y_tgt
                if ridge > 0:
                    A[1:, 1:] += ridge * np.eye(p, dtype=np.float64)
                beta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                beta, *_ = np.linalg.lstsq(X1, y_tgt, rcond=None)

            intercept = float(beta[0])
            coefs = np.asarray(beta[1:], dtype=np.float64)

            params[key] = _ARParams(intercept=intercept, coefs=coefs)

        self._params = params

    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self.p is None or self._target_col is None:
            raise RuntimeError("ARUnivariateForecaster must be fit() before predict_batch()")

        x: torch.Tensor = batch["x"]
        x_mask: Optional[torch.Tensor] = batch.get("x_mask", None)
        meta = batch.get("meta", None)
        if meta is None:
            raise KeyError("batch missing 'meta' (needed to map ZIP -> fitted AR params)")

        if device is not None:
            x = x.to(device)
            if x_mask is not None:
                x_mask = x_mask.to(device)

        pred_len = int(bundle.raw.spec.pred_len)
        p = int(self.p)

        # locate target inside x
        x_name_to_pos = {name: i for i, name in enumerate(bundle.x_cols)}
        if self._target_col not in x_name_to_pos:
            raise KeyError(
                f"Target '{self._target_col}' not present in x_cols. "
                f"ARUnivariate requires target history in encoder window. "
                f"x_cols head={bundle.x_cols[:10]}"
            )
        target_pos = int(x_name_to_pos[self._target_col])

        # ctx: [B, p] oldest->newest
        ctx = _extract_last_p(x, x_mask=x_mask, target_pos=target_pos, p=p)

        # CPU numpy recursion
        ctx_np = ctx.detach().cpu().numpy().astype(np.float64, copy=False)  # [B,p]
        B = ctx_np.shape[0]
        out = np.zeros((B, pred_len), dtype=np.float64)

        zips = [_normalize_zip(m.get("zipcode")) for m in meta]

        for i in range(B):
            prm = self._params.get(zips[i], None)
            hist = ctx_np[i, :].copy()  # [p]

            if prm is None:
                out[i, :] = float(hist[-1])
                continue

            intercept = float(prm.intercept)
            coefs = prm.coefs  # [p]

            for h in range(pred_len):
                lags = hist[::-1]  # [y_{t-1},...,y_{t-p}]
                y_next = intercept + float(np.dot(coefs, lags))

                # guard against explosion
                if not np.isfinite(y_next):
                    y_next = float(hist[-1])

                out[i, h] = y_next
                hist[:-1] = hist[1:]
                hist[-1] = y_next

        y_hat = torch.from_numpy(out.astype(np.float32)).unsqueeze(-1)  # [B, H, 1]
        if y_hat.shape[1] != pred_len:
            raise RuntimeError(f"AR output horizon mismatch: got H={y_hat.shape[1]} but pred_len={pred_len}")

        if device is not None:
            y_hat = y_hat.to(device)
        return y_hat
