"""Oracle / ensemble upper-bound analysis for per-instance model selection.

Given a set of already-trained model runs (same dataset/window/split, per
:func:`st_numeric_baselines.case_library.build_case_library`'s consistency check),
answers three questions that motivate (or don't) building an efficient
per-instance model-selection policy — e.g. an LLM that picks which model to
trust for a given region/window — instead of always using one fixed model.
By default this compares on the held-out **test set only** (pass ``splits``
to widen it) — "oracle accuracy" should answer "how much upside is there on
unseen data", not on data the models were fit on:

1. **Oracle**: if you always picked the best-scoring model for each
   individual instance (by its own lowest MAE), what accuracy would you get?
   This is a strict upper bound: an average of pointwise minimums is always
   <= any single model's average, so the oracle *always* beats the best
   single model in aggregate, by construction — the real question is *how
   much* (the "selection gap": the upside available to a perfect per-instance
   selector, and thus the ceiling a real selector could ever reach).
2. **Ensemble**: does simply averaging (or taking the median of) every
   model's prediction, with no selection at all, already capture most of
   that upside — or does it actually underperform the single best model?
   Unlike the oracle, this is *not* guaranteed to help (correlated biases
   across models can make an average worse than the best individual model),
   so it's measured, not assumed.
3. **Win distribution**: how often does the oracle pick each model? A
   heavily skewed distribution (one model wins almost everywhere) means
   per-instance selection has little upside over just deploying that model;
   an even distribution across several models is exactly the signal that a
   selector has real value to add.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from st_numeric_baselines.case_library import _VALUE_SEP, build_case_library, model_category

_INSTANCE_KEY_COLUMNS = ["region_id", "lookback_start", "forecast_start", "forecast_end"]


def _parse_values(s: str) -> np.ndarray:
    return np.array([float(v) for v in s.split(_VALUE_SEP)], dtype=np.float64)


def _error_stats(prefix: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mse = float(np.mean(np.square(diff)))
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_rmse": float(np.sqrt(mse)),
    }


def build_oracle_report(
    run_dirs: List[Union[str, Path]],
    *,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    ensemble_method: str = "mean",
    splits: Sequence[str] = ("test",),
    test_stride: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Score every run, then compare each model, the oracle, and an ensemble.

    ``splits``: which split(s) to compare on — defaults to the held-out
    **test set only**, since "oracle accuracy" is meant to answer "how much
    upside is there on unseen data", not on data the models were fit on.
    Pass e.g. ``("train", "val", "test")`` to widen it, matching
    :func:`st_numeric_baselines.case_library.build_case_library`'s own default.

    ``test_stride``: defaults to ``None`` — i.e. each run's *own* configured
    ``window.test_stride`` is used, so the instance count here matches your
    actual test set size (e.g. 147 instances, not an artificially densified
    441 from forcing stride 1 the way ``build_case_library`` does for its own
    exhaustive-analysis purpose). Pass ``1`` to force dense sampling instead,
    matching ``build_case_library``'s default.

    Returns ``(instance_df, summary_df, win_counts_df)``:
      - ``instance_df``: one row per instance — the oracle's chosen model and
        its error, the ensemble's error, and every individual model's own MAE
        for that instance (wide format — e.g. feed this to an LLM/classifier
        to learn *when* each model wins).
      - ``summary_df``: one row per method (each model, ``oracle``,
        ``ensemble_<method>``) — aggregate mse/mae/rmse over the instance set
        every model scored in common, sorted best-to-worst by mae.
      - ``win_counts_df``: how often the oracle picks each model.

    Only instances scored by *every* model are compared (an instance a model
    dropped — e.g. via the NaN-window filtering described in
    ``experiments/explainable_library.py``'s ``_nan_dropped_window_counts``
    — can't be fairly attributed to a "best" model, so it's excluded and
    counted in the printed skip count).
    """
    if ensemble_method not in ("mean", "median"):
        raise ValueError(f"ensemble_method must be 'mean' or 'median', got {ensemble_method!r}")

    _case_library_df, detail_df = build_case_library(
        run_dirs,
        device=device,
        max_batches=max_batches,
        splits=splits,
        test_stride=test_stride,
        verbose=verbose,
    )
    if detail_df.empty:
        raise ValueError(f"no scored instances found across the given runs for splits={tuple(splits)!r}")

    models = sorted(detail_df["model"].unique())
    if len(models) < 2:
        raise ValueError(f"oracle/ensemble analysis needs >= 2 models to compare, got {models}")

    if verbose:
        print(f"[oracle] comparing {len(models)} models on splits={tuple(splits)}: {models}")

    instance_records: List[Dict[str, Any]] = []
    n_skipped = 0

    for key, group in detail_df.groupby(_INSTANCE_KEY_COLUMNS, sort=False):
        by_model = {row["model"]: row for _, row in group.iterrows()}
        if set(models) - set(by_model):
            n_skipped += 1
            continue  # not scored by every model -- can't attribute a fair winner

        region_id, lookback_start, forecast_start, forecast_end = key
        y_true = _parse_values(next(iter(by_model.values()))["y_true_raw"])
        preds = {m: _parse_values(by_model[m]["y_pred_raw"]) for m in models}
        maes = {m: float(by_model[m]["mae"]) for m in models}

        oracle_model = min(maes, key=maes.get)
        stacked = np.stack(list(preds.values()))
        ens_pred = np.median(stacked, axis=0) if ensemble_method == "median" else np.mean(stacked, axis=0)

        rec: Dict[str, Any] = {
            "region_id": region_id,
            "lookback_start": lookback_start,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end,
            "oracle_model": oracle_model,
        }
        rec.update(_error_stats("oracle", y_true, preds[oracle_model]))
        rec.update(_error_stats("ensemble", y_true, ens_pred))
        for m in models:
            rec[f"{m}_mae"] = maes[m]
        instance_records.append(rec)

    if verbose and n_skipped:
        print(f"[oracle]   skipped {n_skipped} instance(s) not scored by every model")

    instance_df = pd.DataFrame.from_records(instance_records)
    if instance_df.empty:
        raise ValueError("no instance was scored by every model -- nothing to compare")

    summary_rows: List[Dict[str, Any]] = []
    for m in models:
        sub = detail_df[detail_df["model"] == m].merge(
            instance_df[_INSTANCE_KEY_COLUMNS], on=_INSTANCE_KEY_COLUMNS, how="inner"
        )
        mse = float(sub["mse"].mean())
        summary_rows.append(
            {
                "method": m,
                "category": model_category(m),
                "n_instances": len(sub),
                "mse": mse,
                "mae": float(sub["mae"].mean()),
                "rmse": float(np.sqrt(mse)),
            }
        )

    ensemble_label = f"ensemble_{ensemble_method}"
    for prefix, label, category in [("oracle", "oracle", "oracle"), ("ensemble", ensemble_label, "ensemble")]:
        mse = float(instance_df[f"{prefix}_mse"].mean())
        summary_rows.append(
            {
                "method": label,
                "category": category,
                "n_instances": len(instance_df),
                "mse": mse,
                "mae": float(instance_df[f"{prefix}_mae"].mean()),
                "rmse": float(np.sqrt(mse)),
            }
        )

    summary_df = pd.DataFrame.from_records(summary_rows).sort_values("mae").reset_index(drop=True)

    win_counts = instance_df["oracle_model"].value_counts()
    win_counts_df = pd.DataFrame(
        {
            "model": win_counts.index,
            "n_wins": win_counts.values,
            "win_pct": 100.0 * win_counts.values / len(instance_df),
        }
    ).sort_values("n_wins", ascending=False).reset_index(drop=True)

    if verbose:
        best_single = summary_df[summary_df["method"].isin(models)].sort_values("mae").iloc[0]
        oracle_row = summary_df[summary_df["method"] == "oracle"].iloc[0]
        ensemble_row = summary_df[summary_df["method"] == ensemble_label].iloc[0]
        gap = best_single["mae"] - oracle_row["mae"]
        gap_pct = 100.0 * gap / best_single["mae"] if best_single["mae"] > 0 else 0.0
        print(f"[oracle] best single model: {best_single['method']} (mae={best_single['mae']:.4f})")
        print(
            f"[oracle] oracle (perfect per-instance selection): mae={oracle_row['mae']:.4f} "
            f"({gap_pct:.1f}% lower than the best single model — the selection upside)"
        )
        beats_best = "beats" if ensemble_row["mae"] < best_single["mae"] else "does not beat"
        beats_oracle = "beats" if ensemble_row["mae"] < oracle_row["mae"] else "does not beat"
        print(
            f"[oracle] {ensemble_label}: mae={ensemble_row['mae']:.4f} "
            f"({beats_best} the best single model; {beats_oracle} the oracle)"
        )
        print("[oracle] win distribution (how often the oracle picks each model):")
        print(win_counts_df.to_string(index=False))

    return instance_df, summary_df, win_counts_df
