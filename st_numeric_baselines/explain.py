"""On-demand explainability for one (region, lookback_start) forecast instance.

Two complementary breakdowns, both via model-agnostic **occlusion**
(:func:`occlusion_sensitivity`): replace one "object" (a neighbor node's
whole feature vector, or one feature channel) with its **mean over the
training history** — computed in the model's own processed feature space, so
it's a true in-distribution replacement rather than an arbitrary value —
rerun the forward pass, and measure how much the target region's forecast
moves: ``sensitivity = mean(|y_full - y_occluded|)`` over the horizon, in raw
target units.

- :func:`explain_neighbors` — spatial axis, GNN models only: occlude one
  *node* (all its features, replaced with that node's own per-feature
  training means) at a time. Which neighbor regions did the forecast rely on?
- :func:`explain_features` — variable axis, any registered model: occlude
  one *feature channel* (for the target region only, replaced with the
  target's own training mean for that feature) at a time. Which input
  variables (own price history, homes_sold, inventory, ...) did the forecast
  rely on?

Both only need the public ``model.predict_batch(...)`` API, so they work
identically across every model in the registry without per-architecture
plumbing. Called for one instance at a time (not baked into the case
library, which would need one forward pass per perturbation per instance).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.data.schema import normalize_zipcode
from st_numeric_baselines.experiments.run_loader import load_run
from st_numeric_baselines.graph.loader import load_graph
from st_numeric_baselines.metrics.evaluator import invert_to_raw
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.gnn.gnn_forecaster import GNNForecasterBase


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_raw_horizon(bundle: ProcBundle, y_proc_row: np.ndarray) -> np.ndarray:
    """[H, Dy] processed -> [H] raw target values."""
    arr = y_proc_row[None, :, :]
    _, p_raw = invert_to_raw(bundle, arr, arr)
    return p_raw[0]


def _resolve_target_and_time(bundle: ProcBundle, region_id: str, lookback_start: str) -> Tuple[int, int, int]:
    """Return (target_idx, t0, seq_len) for one (region_id, lookback_start) instance."""
    zipcodes = bundle.raw.aligned.zipcodes
    target_norm = normalize_zipcode(region_id)
    zip_norm = [normalize_zipcode(z) for z in zipcodes]
    if target_norm not in zip_norm:
        raise ValueError(f"region_id={region_id!r} not found in this run's regions")
    target_idx = zip_norm.index(target_norm)

    dates = bundle.raw.aligned.dates
    target_date = pd.Timestamp(lookback_start)
    date_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    if target_date not in date_idx:
        raise ValueError(
            f"lookback_start={lookback_start!r} does not match any date in this dataset "
            f"(range: {dates[0]} .. {dates[-1]})"
        )
    t0 = date_idx[target_date]

    seq_len = int(bundle.raw.spec.seq_len)
    if t0 + seq_len > len(dates):
        raise ValueError(f"lookback_start={lookback_start!r} leaves too few future steps for seq_len={seq_len}")
    return target_idx, t0, seq_len


def _build_instance_x(bundle: ProcBundle, target_idx: int, t0: int, seq_len: int, *, is_gnn: bool) -> torch.Tensor:
    """Single-instance model input: [1, L, N, Dx] for GNN models, [1, L, Dx] otherwise."""
    name_to_idx = {n: i for i, n in enumerate(bundle.aligned_proc.schema.continuous_cols)}
    x_idx = [name_to_idx[c] for c in bundle.x_cols]
    values = bundle.aligned_proc.values  # [N, T, D]

    if is_gnn:
        x_np = values[:, t0 : t0 + seq_len, :][:, :, x_idx]  # [N, L, Dx]
        return torch.tensor(x_np, dtype=torch.float32).permute(1, 0, 2).unsqueeze(0)  # [1, L, N, Dx]

    x_np = values[target_idx, t0 : t0 + seq_len, :][:, x_idx]  # [L, Dx]
    return torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # [1, L, Dx]


def _build_instance_x_mark(bundle: ProcBundle, t0: int, seq_len: int) -> torch.Tensor:
    """Single-instance time marks [1, L, 2] (year, month) — some models (e.g. Chronos-2,
    which builds a real-timestamp DataFrame) need this alongside ``x`` in the batch dict;
    it's the same regardless of node/target, so it's built once per instance and reused
    across every occlusion/masking variant of that instance.
    """
    tm_np = bundle.aligned_proc.time_marks[t0 : t0 + seq_len, :]  # [L, 2]
    return torch.tensor(tm_np, dtype=torch.float32).unsqueeze(0)  # [1, L, 2]


def _predict_target_raw(
    model: BaseForecaster,
    bundle: ProcBundle,
    x: torch.Tensor,
    target_idx: int,
    *,
    is_gnn: bool,
    device: Optional[torch.device],
    x_mark: Optional[torch.Tensor] = None,
) -> np.ndarray:
    batch: Dict[str, torch.Tensor] = {"x": x}
    if x_mark is not None:
        batch["x_mark"] = x_mark
    y = model.predict_batch(batch, bundle=bundle, device=device)
    y_target = y[target_idx] if is_gnn else y[0]  # [pred_len, Dy]
    return _to_raw_horizon(bundle, _to_numpy(y_target))


def _train_feature_means(bundle: ProcBundle) -> np.ndarray:
    """Per-node, per-feature mean over the training split, in the model's own
    processed feature space (matching ``bundle.x_cols`` order) — ``[N, Dx]``.
    The occlusion fill value: an actual in-distribution average, not an
    arbitrary placeholder like zero.
    """
    name_to_idx = {n: i for i, n in enumerate(bundle.aligned_proc.schema.continuous_cols)}
    x_idx = [name_to_idx[c] for c in bundle.x_cols]
    train_start, train_end = bundle.raw.split.train
    values = bundle.aligned_proc.values  # [N, T, D]
    return values[:, train_start:train_end, :][:, :, x_idx].mean(axis=1)  # [N, Dx]


def occlusion_sensitivity(y_full_raw: np.ndarray, y_occluded_raw: np.ndarray) -> float:
    """Sensitivity of a forecast to one occluded object.

    ``sensitivity = mean(|y_full - y_occluded|)`` over the forecast horizon,
    in raw target units — ``y_full`` is the model's own unperturbed forecast
    (not the ground truth), ``y_occluded`` is its forecast with one object
    (a neighbor node or a feature channel) replaced by its training-history
    mean.
    """
    return float(np.mean(np.abs(np.asarray(y_full_raw) - np.asarray(y_occluded_raw))))


def _normalize_contributions(df: pd.DataFrame) -> pd.DataFrame:
    total = df["sensitivity"].sum()
    if total > 0:
        df["contribution_pct"] = 100.0 * df["sensitivity"] / total
    else:
        df["contribution_pct"] = 100.0 / max(len(df), 1)
    return df


def explain_neighbors(
    run_dir: Union[str, Path],
    *,
    region_id: str,
    lookback_start: str,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Spatial breakdown: which neighbor regions contributed to this forecast.

    GNN models only (there is no neighbor structure otherwise). Returns a
    DataFrame with columns ``neighbor_region_id, contribution_pct,
    sensitivity, is_self``, sorted by contribution descending.
    """
    model, bundle, cfg = load_run(run_dir, device=device)

    if not isinstance(model, GNNForecasterBase):
        raise ValueError(
            f"explain_neighbors only applies to spatiotemporal/GNN models "
            f"(got model={cfg.get('model', {}).get('name')!r}, which is not a GNNForecasterBase)"
        )

    graph_cfg = bundle.raw.graph
    if not graph_cfg.path:
        raise ValueError("This run's dataset config has no graph.path set — cannot resolve neighbors")

    zipcodes = bundle.raw.aligned.zipcodes
    geo = load_graph(graph_cfg.path, zipcodes)
    target_idx, t0, seq_len = _resolve_target_and_time(bundle, region_id, lookback_start)

    # neighbor set: any node connected to target_idx in either direction, excluding self-loops
    src, dst = np.asarray(geo.edge_index[0]), np.asarray(geo.edge_index[1])
    neighbor_idxs = sorted(
        (set(dst[src == target_idx].tolist()) | set(src[dst == target_idx].tolist())) - {target_idx}
    )

    x_base = _build_instance_x(bundle, target_idx, t0, seq_len, is_gnn=True)
    x_mark = _build_instance_x_mark(bundle, t0, seq_len)
    y_full = _predict_target_raw(model, bundle, x_base, target_idx, is_gnn=True, device=device, x_mark=x_mark)
    train_means = _train_feature_means(bundle)  # [N, Dx]

    rows = []
    for node_idx, label, is_self in [(target_idx, "self", True)] + [
        (n, zipcodes[n], False) for n in neighbor_idxs
    ]:
        x_occ = x_base.clone()
        fill = torch.as_tensor(train_means[node_idx], dtype=x_occ.dtype)  # [Dx]
        x_occ[:, :, node_idx, :] = fill
        y_occluded = _predict_target_raw(model, bundle, x_occ, target_idx, is_gnn=True, device=device, x_mark=x_mark)
        sensitivity = occlusion_sensitivity(y_full, y_occluded)
        rows.append({"neighbor_region_id": label, "sensitivity": sensitivity, "is_self": is_self})

    df = _normalize_contributions(pd.DataFrame(rows))
    return df[["neighbor_region_id", "contribution_pct", "sensitivity", "is_self"]].sort_values(
        "contribution_pct", ascending=False
    ).reset_index(drop=True)


def explain_features(
    run_dir: Union[str, Path],
    *,
    region_id: str,
    lookback_start: str,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """Variable breakdown: which input features contributed to this forecast.

    Works for any registered model (GNN, DL, foundation, statistical/ML —
    anything implementing ``predict_batch``), not just spatiotemporal ones.
    Occludes one feature channel at a time for the target region only
    (replaced with the target's own training-history mean for that feature,
    across the whole lookback window), leaving every other region/feature
    untouched. Returns a DataFrame with columns ``feature, contribution_pct,
    sensitivity``, sorted by contribution descending.
    """
    model, bundle, _cfg = load_run(run_dir, device=device)
    is_gnn = isinstance(model, GNNForecasterBase)

    target_idx, t0, seq_len = _resolve_target_and_time(bundle, region_id, lookback_start)
    x_base = _build_instance_x(bundle, target_idx, t0, seq_len, is_gnn=is_gnn)
    x_mark = _build_instance_x_mark(bundle, t0, seq_len)
    y_full = _predict_target_raw(model, bundle, x_base, target_idx, is_gnn=is_gnn, device=device, x_mark=x_mark)
    train_means = _train_feature_means(bundle)  # [N, Dx]

    rows = []
    for f, col_name in enumerate(bundle.x_cols):
        x_occ = x_base.clone()
        fill = float(train_means[target_idx, f])
        if is_gnn:
            x_occ[:, :, target_idx, f] = fill
        else:
            x_occ[:, :, f] = fill
        y_occluded = _predict_target_raw(model, bundle, x_occ, target_idx, is_gnn=is_gnn, device=device, x_mark=x_mark)
        sensitivity = occlusion_sensitivity(y_full, y_occluded)
        rows.append({"feature": col_name, "sensitivity": sensitivity})

    df = _normalize_contributions(pd.DataFrame(rows))
    return df[["feature", "contribution_pct", "sensitivity"]].sort_values(
        "contribution_pct", ascending=False
    ).reset_index(drop=True)
