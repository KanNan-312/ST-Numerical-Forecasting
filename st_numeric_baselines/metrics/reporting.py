from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


def load_metrics_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(run_root: str | Path) -> pd.DataFrame:
    root = Path(run_root)
    rows: List[Dict[str, Any]] = []
    for metrics_path in root.rglob("metrics.json"):
        try:
            m = load_metrics_json(metrics_path)
        except Exception:
            continue
        fc = m.get("forecast_config", {}) or {}
        row_base: Dict[str, Any] = {
            "run_dir": str(metrics_path.parent),
            "dataset": m.get("dataset"),
            "model": m.get("model"),
            "task": m.get("task"),
            "window": m.get("window"),
            "seq_len": fc.get("seq_len"),
            "label_len": fc.get("label_len"),
            "pred_len": fc.get("pred_len"),
            "test_stride": fc.get("test_stride"),
            "features_mode": fc.get("features_mode"),
            "split_train": m.get("n_train"),
            "split_val": m.get("n_val"),
            "split_test": m.get("n_test"),
            "pipeline": m.get("pipeline"),
        }
        for split in ("val", "test"):
            if split not in m:
                continue
            s = m[split] or {}
            row = dict(row_base)
            row["split"] = split
            row["log_rmse"] = s.get("log_rmse")
            row["rmse"] = s.get("rmse")
            row["mape"] = s.get("mape")
            row["mae"] = s.get("mae")
            row["log_mae"] = s.get("log_mae")
            row["n_points"] = s.get("n_points")
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def pivot_metric(
    df: pd.DataFrame,
    *,
    task: str,
    split: str,
    metric: str,
    dataset: Optional[str] = None,
    window_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """One row per model, one column per forecast-config ``window`` label.

    ``dataset``: restrict to one dataset's runs (recommended — two
    different datasets' same model+window combo would otherwise look
    identical in one combined table, since ``window`` alone doesn't carry
    dataset identity). Pass ``None`` only if you've already filtered
    ``df`` to a single dataset yourself, or genuinely want everything
    combined.
    """
    mask = (df["task"] == task) & (df["split"] == split)
    if dataset is not None:
        mask &= df["dataset"] == dataset
    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    piv = sub.pivot_table(index="model", columns="window", values=metric, aggfunc="first")
    if window_order is not None:
        cols = [c for c in window_order if c in piv.columns]
        piv = piv.reindex(columns=cols)
    piv = piv.sort_index()
    return piv
