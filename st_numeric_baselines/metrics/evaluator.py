from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.models.base import BaseForecaster


def _to_numpy(x: Any) -> np.ndarray:
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _bundle_target_indices(bundle: ProcBundle):
    proc_names = list(bundle.aligned_proc.schema.continuous_cols)
    name_to_idx = {n: i for i, n in enumerate(proc_names)}
    y_idx_full = [name_to_idx[n] for n in bundle.y_cols]
    d_proc = int(bundle.aligned_proc.values.shape[-1])
    raw_target_index = int(bundle.raw_target_index)
    return y_idx_full, d_proc, raw_target_index


def invert_to_raw(bundle: ProcBundle, y_true_proc: Any, y_pred_proc: Any):
    """Invert processed-space ``[B,H,Dy]`` tensors to raw target-space ``[B,H]`` arrays.

    Pipeline-agnostic: fully un-does whatever transform stages (log/clip/
    zscore/pca) the bundle's pipeline applied, then reads back just the raw
    target column. Shared by :class:`StreamingEvaluator` and
    :func:`per_sample_errors` so both report errors in the same raw units.
    """
    yt = _to_numpy(y_true_proc).astype(np.float32)
    yp = _to_numpy(y_pred_proc).astype(np.float32)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if yt.ndim != 3:
        raise ValueError(f"Expected [B,H,Dy], got {yt.shape}")

    B, H, Dy = yt.shape
    y_idx_full, d_proc, raw_target_index = _bundle_target_indices(bundle)

    full_t = np.zeros((B, H, d_proc), dtype=np.float32)
    full_p = np.zeros((B, H, d_proc), dtype=np.float32)
    full_t[:, :, y_idx_full] = yt
    full_p[:, :, y_idx_full] = yp

    # Fully invert the pipeline → raw price space (pipeline-agnostic)
    t_raw_full = bundle.pipeline.inverse(full_t, keep_log=False)
    p_raw_full = bundle.pipeline.inverse(full_p, keep_log=False)
    t_raw = t_raw_full[:, :, raw_target_index].astype(np.float64)
    p_raw = p_raw_full[:, :, raw_target_index].astype(np.float64)
    return t_raw, p_raw


def per_sample_errors(bundle: ProcBundle, y_true_proc: Any, y_pred_proc: Any):
    """Per-row MAE and RMSE in raw target units, averaged over the horizon axis.

    Returns ``(mae[B], rmse[B])``. Used by the case-library builder, which
    needs one error value per (region, window) instance rather than the
    dataset-wide aggregate :class:`StreamingEvaluator` produces.
    """
    t_raw, p_raw = invert_to_raw(bundle, y_true_proc, y_pred_proc)
    diff = p_raw - t_raw  # [B, H]
    mae = np.mean(np.abs(diff), axis=1)
    rmse = np.sqrt(np.mean(diff * diff, axis=1))
    return mae, rmse


@dataclass
class EvalResult:
    log_rmse: float   # RMSE(log1p(p_raw), log1p(t_raw))
    rmse: float       # RMSE in raw price space
    mape: float       # mean absolute percentage error in raw price space
    mae: float        # MAE in raw price space
    log_mae: float    # MAE(log1p(p_raw), log1p(t_raw))
    n_points: int


class StreamingEvaluator:
    """Accumulates predictions and ground-truth across batches, then computes 5 metrics.

    The pipeline is always **fully inverted** to raw price space first.
    All metrics are then computed on those raw prices (or log1p thereof):

    * ``log_rmse``  – RMSE(log1p(p_raw), log1p(t_raw))
    * ``rmse``      – RMSE(p_raw, t_raw)
    * ``mape``      – mean |p_raw - t_raw| / |t_raw|
    * ``mae``       – mean |p_raw - t_raw|
    * ``log_mae``   – MAE(log1p(p_raw), log1p(t_raw))
    """

    def __init__(self, bundle: ProcBundle, *, eps: float = 1e-8) -> None:
        self.bundle = bundle
        self.eps = float(eps)

        self._sse_log = 0.0
        self._sse_raw = 0.0
        self._sum_ape = 0.0
        self._sum_ae_raw = 0.0
        self._sum_ae_log = 0.0
        self._n = 0

    def update(self, y_true_proc: Any, y_pred_proc: Any) -> None:
        t_raw, p_raw = invert_to_raw(self.bundle, y_true_proc, y_pred_proc)

        # log space (clamp to ≥0 before log1p to guard float precision)
        t_log = np.log1p(np.maximum(t_raw, 0.0))
        p_log = np.log1p(np.maximum(p_raw, 0.0))

        diff_raw = p_raw - t_raw
        diff_log = p_log - t_log

        self._sse_log += float(np.sum(diff_log * diff_log))
        self._sse_raw += float(np.sum(diff_raw * diff_raw))
        self._sum_ae_raw += float(np.sum(np.abs(diff_raw)))
        self._sum_ae_log += float(np.sum(np.abs(diff_log)))
        self._n += int(diff_raw.size)

        denom = np.maximum(np.abs(t_raw), self.eps)
        self._sum_ape += float(np.sum(np.abs(diff_raw) / denom))

    def compute(self) -> EvalResult:
        nan = float("nan")
        if self._n == 0:
            return EvalResult(log_rmse=nan, rmse=nan, mape=nan, mae=nan, log_mae=nan, n_points=0)
        return EvalResult(
            log_rmse=float(np.sqrt(self._sse_log / self._n)),
            rmse=float(np.sqrt(self._sse_raw / self._n)),
            mape=float(self._sum_ape / self._n),
            mae=float(self._sum_ae_raw / self._n),
            log_mae=float(self._sum_ae_log / self._n),
            n_points=self._n,
        )


@torch.no_grad()
def evaluate_forecaster(
    model: BaseForecaster,
    bundle: ProcBundle,
    *,
    split: str = "test",
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> EvalResult:
    """Evaluate a forecaster on the given split.

    Always fully inverts the pipeline to raw price space, then computes
    log_rmse, rmse, mape, mae, and log_mae.
    """
    split = split.lower()
    if split not in ("train", "val", "test"):
        raise ValueError("split must be one of: train/val/test")

    # GNN models store their own graph-structured dataloaders; fall back to
    # the standard per-ZIP dataloader for all other model families.
    _graph_dls = getattr(model, "_graph_dataloaders", None)
    dl = _graph_dls[split] if (_graph_dls and split in _graph_dls) else bundle.dataloaders[split]

    evaluator = StreamingEvaluator(bundle)

    pred_len = int(bundle.raw.spec.pred_len)
    total = min(len(dl), max_batches) if max_batches is not None else len(dl)
    dl_bar = tqdm(dl, desc=f"  eval/{split}", total=total, leave=False, unit="bt")
    for bi, batch in enumerate(dl_bar):
        if max_batches is not None and bi >= max_batches:
            dl_bar.close()
            break

        y_true = batch["y"][:, -pred_len:, :]
        y_pred = model.predict_batch(batch, bundle=bundle, device=device)
        evaluator.update(y_true, y_pred)

    return evaluator.compute()
