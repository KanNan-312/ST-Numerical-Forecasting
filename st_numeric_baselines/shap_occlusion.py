"""Grouped exact-Shapley feature importance for multivariate forecasting models.

Implements the method from arXiv:2604.28149 ("Explainable Load Forecasting with
Covariate-Informed Time Series Foundation Models"): group the input into a small
number of "players" (default: one group per covariate column; optionally also
split the lookback window into contiguous temporal buckets, mirroring the
paper's own scheme of covariate groups + temporal-window groups), then compute
the **exact** Shapley value over all ``2^N`` coalitions of groups —

    SHAP(g) = sum_{S subseteq G\\{g}} [ |S|!(N-|S|-1)! / N! ] * (f(S+g) - f(S))

with ``f(S)`` the model's own forecast (mean over the horizon, raw target units)
when every group *not* in ``S`` is masked. No Monte-Carlo/KernelSHAP
approximation — this matches the paper, and is only tractable because ``N`` is
kept small (grouped, not per-raw-input-cell).

**Masking mechanic**, per the paper's own model-specific distinction:
- Chronos models (``chronos*``): masked cells are set to **NaN**. Chronos's own
  tokenizer already treats NaN as "missing" (the same mechanism this codebase's
  fine-tuning code already relies on, see ``ChronosFullFineTuneForecaster``'s
  ``masked_fill(mask == 0, torch.nan)``), so this is genuine *native* missingness
  rather than a substituted value — matching the paper's "no background sampling
  needed" approach for models that natively tolerate incomplete input.
- Every other model (iTransformer, DL/GNN models, TimesFM — none of which
  reliably treat interior NaNs as "missing" data): masked cells are replaced
  with the **training-history mean** (:func:`st_numeric_baselines.explain._train_feature_means`,
  the same in-distribution fill already used by the neighbor/feature occlusion
  tool) — standard perturbation-based Shapley, the paper's own fallback for
  fixed-shape models.

Only called for one instance at a time (not baked into the case library — see
:mod:`st_numeric_baselines.case_library`).
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from st_numeric_baselines.experiments.run_loader import load_run
from st_numeric_baselines.explain import (
    _build_instance_x,
    _build_instance_x_mark,
    _predict_target_raw,
    _resolve_target_and_time,
    _train_feature_means,
)
from st_numeric_baselines.models.gnn.gnn_forecaster import GNNForecasterBase

_NAN_MASKING_MODELS = {
    "chronos_zero",
    "chronos_ft",
    "chronos_full_ft",
    "chronos2_zero",
    "chronos2_ft",
    "chronos2_full_ft",
}

_MAX_GROUPS = 16  # 2^16 = 65536 forward passes; raise only if you know what you're doing


def _supports_nan_masking(model_name: str) -> bool:
    return model_name in _NAN_MASKING_MODELS


class _GroupSpec:
    __slots__ = ("name", "feature_idxs", "time_slice")

    def __init__(self, name: str, feature_idxs: Optional[List[int]], time_slice: Optional[slice]) -> None:
        self.name = name
        self.feature_idxs = feature_idxs
        self.time_slice = time_slice


def _build_groups(
    x_cols: Sequence[str], seq_len: int, *, groups: Optional[Dict[str, List[str]]], n_temporal_groups: int
) -> List[_GroupSpec]:
    specs: List[_GroupSpec] = []

    if n_temporal_groups and n_temporal_groups > 1:
        bounds = np.linspace(0, seq_len, n_temporal_groups + 1).astype(int)
        for i in range(n_temporal_groups):
            lo, hi = int(bounds[i]), int(bounds[i + 1])
            if hi > lo:
                specs.append(_GroupSpec(f"time[{lo}:{hi}]", None, slice(lo, hi)))

    if groups is not None:
        for name, cols in groups.items():
            idxs = [list(x_cols).index(c) for c in cols]
            specs.append(_GroupSpec(name, idxs, None))
    else:
        for f, col in enumerate(x_cols):
            specs.append(_GroupSpec(col, [f], None))

    if len(specs) > _MAX_GROUPS:
        raise ValueError(
            f"{len(specs)} feature groups requested -> 2^{len(specs)} model evaluations, too expensive. "
            "Pass `groups={...}` to consolidate covariates into fewer groups, or reduce n_temporal_groups."
        )
    if len(specs) == 0:
        raise ValueError("no feature groups to explain")
    return specs


def shapley_feature_importance(
    run_dir: Union[str, Path],
    *,
    region_id: str,
    lookback_start: str,
    groups: Optional[Dict[str, List[str]]] = None,
    n_temporal_groups: int = 1,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Grouped exact-Shapley feature importance for one forecast instance.

    ``groups``: optional ``{group_name: [x_col, ...]}`` to consolidate covariates
    into fewer players (default: one group per column in ``bundle.x_cols``).
    ``n_temporal_groups``: split the lookback window into this many contiguous
    temporal-bucket groups too (default 1 = no temporal grouping, since this
    benchmark's lookback windows are far shorter than the paper's 672-hour
    example — pass a larger value for parity with the paper's full scheme).

    Returns a DataFrame with columns ``group, shap_value, contribution_pct``,
    sorted by contribution descending. ``shap_value`` is signed (raw target
    units, mean over the forecast horizon); ``contribution_pct`` normalizes
    ``|shap_value|`` to 100%.
    """
    model, bundle, cfg = load_run(run_dir, device=device)
    model_name = str((cfg.get("model", {}) or {}).get("name"))
    is_gnn = isinstance(model, GNNForecasterBase)
    use_nan = _supports_nan_masking(model_name)

    target_idx, t0, seq_len = _resolve_target_and_time(bundle, region_id, lookback_start)
    x_base = _build_instance_x(bundle, target_idx, t0, seq_len, is_gnn=is_gnn)
    x_mark = _build_instance_x_mark(bundle, t0, seq_len)
    train_means = _train_feature_means(bundle)  # [N, Dx]

    group_specs = _build_groups(bundle.x_cols, seq_len, groups=groups, n_temporal_groups=n_temporal_groups)
    n_groups = len(group_specs)

    def _fill_value(group: _GroupSpec) -> torch.Tensor:
        if use_nan:
            return torch.tensor(float("nan"), dtype=x_base.dtype)
        if group.feature_idxs is not None:
            return torch.as_tensor(train_means[target_idx, group.feature_idxs], dtype=x_base.dtype)
        return torch.as_tensor(train_means[target_idx], dtype=x_base.dtype)

    def _masked_x(present: frozenset) -> torch.Tensor:
        x_occ = x_base.clone()
        for gi, group in enumerate(group_specs):
            if gi in present:
                continue
            fill = _fill_value(group)
            if is_gnn:
                if group.feature_idxs is not None:
                    x_occ[:, :, target_idx, group.feature_idxs] = fill
                else:
                    x_occ[:, group.time_slice, target_idx, :] = fill
            else:
                if group.feature_idxs is not None:
                    x_occ[:, :, group.feature_idxs] = fill
                else:
                    x_occ[:, group.time_slice, :] = fill
        return x_occ

    f_cache: Dict[frozenset, float] = {}

    def _f_scalar(present: frozenset) -> float:
        if present not in f_cache:
            x_occ = _masked_x(present)
            y = _predict_target_raw(model, bundle, x_occ, target_idx, is_gnn=is_gnn, device=device, x_mark=x_mark)
            f_cache[present] = float(np.mean(y))
        return f_cache[present]

    all_idx = list(range(n_groups))
    n_fact = math.factorial(n_groups)
    shap = np.zeros(n_groups)
    for i in all_idx:
        others = [j for j in all_idx if j != i]
        for r in range(len(others) + 1):
            weight = math.factorial(r) * math.factorial(n_groups - r - 1) / n_fact
            for S in itertools.combinations(others, r):
                S = frozenset(S)
                marg = _f_scalar(S | {i}) - _f_scalar(S)
                shap[i] += weight * marg

    df = pd.DataFrame({"group": [g.name for g in group_specs], "shap_value": shap})
    total = df["shap_value"].abs().sum()
    if total > 0:
        df["contribution_pct"] = 100.0 * df["shap_value"].abs() / total
    else:
        df["contribution_pct"] = 100.0 / max(len(df), 1)
    return df.sort_values("contribution_pct", ascending=False).reset_index(drop=True)
