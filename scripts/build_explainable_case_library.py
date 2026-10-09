"""Build the 3-model-family explainable case library for one dataset.

Different from scripts/build_case_library.py (which ranks many models against
each other): this always uses exactly one model per family — a univariate
model, a multivariate model, and STExplainer — reusing their already-trained
checkpoints (no retraining), and produces forecasts plus per-instance
explanations (which node/feature/time mattered):

  - forecasts_detail.csv        — point forecasts from all 3 runs, every instance
  - feature_importance.csv      — grouped-Shapley feature importance (the
                                   multivariate model only): a sampled subset of
                                   --feature-importance-split instances by default,
                                   or every instance in that split if
                                   --explain-n-instances <= 0 (2^N model evals
                                   per instance — check the printed cost estimate
                                   before running with "every instance")
  - stexplainer_explanation.csv — STExplainer's native node/time/feature
                                   explanation, every instance (cheap: one extra
                                   forward pass per time window)

Usage
-----
    python scripts/build_explainable_case_library.py \
      --runs-root runs/dc_house \
      --univariate-model timesfm_zero \
      --multivariate-model chronos2_zero \
      --stexplainer-model stexplainer \
      --explain-n-instances 20 \
      --out-dir runs/dc_house/explainable_case_library
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from st_numeric_baselines.experiments.explainable_library import build_explainable_case_library
from st_numeric_baselines.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=str, required=True,
                   help="directory containing one subdirectory per run (config.yaml + checkpoint.pt)")
    p.add_argument("--univariate-model", type=str, default="timesfm_zero", help="e.g. patchtst, timesfm_zero")
    p.add_argument("--multivariate-model", type=str, default="chronos2_zero", help="e.g. chronos2_zero, itransformer")
    p.add_argument("--stexplainer-model", type=str, default="stexplainer")
    p.add_argument("--device", type=str, default=None, help="e.g. cuda, cpu (default: each run's own config)")
    p.add_argument("--max-eval-batches", type=int, default=None, help="cap batches per split per run (debugging)")
    p.add_argument("--explain-n-instances", type=int, default=20,
                   help="number of instances to run grouped-SHAP feature importance on "
                   "(2^N_groups model evaluations each — keep this modest). Pass 0 or a "
                   "negative number to run every instance in --feature-importance-split instead of a sample")
    p.add_argument("--feature-importance-split", type=str, default="test", choices=["train", "val", "test"],
                   help="which split's instances grouped-SHAP feature importance is computed over (default: test)")
    p.add_argument("--n-temporal-groups", type=int, default=1,
                   help="split the lookback window into this many temporal-bucket SHAP groups too "
                   "(default 1 = feature-only grouping)")
    p.add_argument("--out-dir", type=str, required=True,
                   help="output folder — forecasts_detail.csv, feature_importance.csv, "
                   "stexplainer_explanation.csv are written here")
    return p.parse_args()


def _find_run(runs_root: Path, model_name: str) -> Path:
    candidates = [
        child for child in sorted(runs_root.iterdir())
        if child.is_dir() and (child / "config.yaml").exists()
    ]
    matches = []
    for run_dir in candidates:
        cfg = load_yaml(run_dir / "config.yaml")
        if str((cfg.get("model", {}) or {}).get("name", "")) == model_name:
            matches.append(run_dir)
    if not matches:
        raise SystemExit(f"No run found under {runs_root} for model {model_name!r}")
    if len(matches) > 1:
        raise SystemExit(
            f"Multiple runs found under {runs_root} for model {model_name!r}: {matches} — "
            "point --runs-root at a directory with exactly one run per model, or rename the extras"
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        raise SystemExit(f"--runs-root not found or not a directory: {runs_root}")

    univariate_run = _find_run(runs_root, args.univariate_model)
    multivariate_run = _find_run(runs_root, args.multivariate_model)
    stexplainer_run = _find_run(runs_root, args.stexplainer_model)
    print("Runs selected:")
    print(f"  univariate:   {univariate_run}")
    print(f"  multivariate: {multivariate_run}")
    print(f"  stexplainer:  {stexplainer_run}")

    device = torch.device(args.device) if args.device else None
    forecasts_detail_df, feature_importance_df, stexplainer_explanation_df = build_explainable_case_library(
        univariate_run_dir=univariate_run,
        multivariate_run_dir=multivariate_run,
        stexplainer_run_dir=stexplainer_run,
        device=device,
        max_batches=args.max_eval_batches,
        explain_n_instances=args.explain_n_instances,
        feature_importance_split=args.feature_importance_split,
        n_temporal_groups=args.n_temporal_groups,
        verbose=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    forecasts_path = out_dir / "forecasts_detail.csv"
    forecasts_detail_df.to_csv(forecasts_path, index=False)
    print(f"Saved forecasts ({len(forecasts_detail_df)} rows) -> {forecasts_path}")

    fi_path = out_dir / "feature_importance.csv"
    feature_importance_df.to_csv(fi_path, index=False)
    print(f"Saved feature importance ({len(feature_importance_df)} rows) -> {fi_path}")

    exp_path = out_dir / "stexplainer_explanation.csv"
    stexplainer_explanation_df.to_csv(exp_path, index=False)
    print(f"Saved STExplainer explanation ({len(stexplainer_explanation_df)} rows) -> {exp_path}")


if __name__ == "__main__":
    main()
