"""Oracle / ensemble upper-bound analysis for efficient per-instance model selection.

Given a set of already-trained model runs (same dataset/window/split),
compared on the held-out **test set only** by default (``--splits`` widens
this), using each run's own configured ``test_stride`` (``--test-stride``
overrides it) so the instance count matches your actual test set size —
"oracle accuracy" should answer "how much upside is there on unseen data",
not on an artificially densified re-sampling of it. Answers:
  1. Oracle: if you always used whichever model scored best on each individual
     instance, how much better is that than the best single model overall?
     (the "selection gap" — the upside a per-instance selector, e.g. an
     LLM-driven one, could capture; this is a strict upper bound, guaranteed
     to beat every single model in aggregate by construction)
  2. Ensemble: does simply averaging every model's prediction already capture
     most of that upside, or does it underperform the best single model?
     (not guaranteed either way — measured, not assumed)
  3. Win distribution: how often does each model turn out to be the
     per-instance winner — a skewed distribution means selection barely
     matters, an even one means it matters a lot.

Three files are written under --out-dir:
  - oracle_instance_detail.csv — one row per instance: the oracle's pick and
    its error, the ensemble's error, and every model's own MAE for that
    instance (wide format, e.g. for training/prompting an LLM selector)
  - oracle_summary.csv         — one row per method (each model, oracle,
    ensemble): aggregate mse/mae/rmse over the shared instance set
  - oracle_win_counts.csv      — how often the oracle picks each model

Usage
-----
    python scripts/build_oracle_report.py \
      --runs-root runs/dc_house \
      --models dlinear patchtst gcn_tcn stgformer stid agcrn mtgnn \
      --ensemble-method mean \
      --out-dir runs/dc_house/oracle_report

Omit --models to include every run found directly under --runs-root.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]  # requires: pip install -e . (see README Quick start)

import torch

from st_numeric_baselines.oracle_selection import build_oracle_report
from st_numeric_baselines.utils.config import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=str, required=True,
                   help="directory containing one subdirectory per run (config.yaml + checkpoint.pt)")
    p.add_argument("--models", nargs="+", default=None,
                   help="model names to include (matches config.yaml's model.name); omit for all runs found")
    p.add_argument("--device", type=str, default=None, help="e.g. cuda, cpu (default: each run's own config)")
    p.add_argument("--max-eval-batches", type=int, default=None, help="cap batches per split per run (debugging)")
    p.add_argument("--ensemble-method", type=str, default="mean", choices=["mean", "median"],
                   help="how to combine models' predictions for the ensemble baseline (default: mean)")
    p.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"],
                   help="which split(s) to compare on (default: test only -- oracle accuracy should "
                   "answer 'how much upside is there on unseen data')")
    p.add_argument("--test-stride", type=int, default=None,
                   help="force this test_stride onto every run before scoring (default: None -- use "
                   "each run's own configured test_stride, so the instance count matches your actual "
                   "test set size; pass 1 for dense/exhaustive sampling instead)")
    p.add_argument("--out-dir", type=str, required=True,
                   help="output folder — oracle_instance_detail.csv, oracle_summary.csv, "
                   "oracle_win_counts.csv are written here")
    return p.parse_args()


def _discover_run_dirs(runs_root: Path, models: Optional[List[str]]) -> List[Path]:
    # Only config.yaml is required here — checkpoint.pt may legitimately be
    # missing for checkpoint_optional models (e.g. timesfm_zero, chronos2_zero),
    # which load_run() handles by running fit() directly instead of erroring.
    candidates = [
        child for child in sorted(runs_root.iterdir())
        if child.is_dir() and (child / "config.yaml").exists()
    ]
    if not candidates:
        raise SystemExit(f"No run directories with config.yaml found under {runs_root}")

    if models is None:
        return candidates

    wanted = set(models)
    selected = []
    for run_dir in candidates:
        cfg = load_yaml(run_dir / "config.yaml")
        name = str((cfg.get("model", {}) or {}).get("name", ""))
        if name in wanted:
            selected.append(run_dir)

    found_names = {str((load_yaml(r / "config.yaml").get("model", {}) or {}).get("name", "")) for r in candidates}
    missing = wanted - found_names
    if missing:
        raise SystemExit(f"No run found under {runs_root} for model(s): {sorted(missing)}")
    return selected


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        raise SystemExit(f"--runs-root not found or not a directory: {runs_root}")

    run_dirs = _discover_run_dirs(runs_root, args.models)
    print(f"Found {len(run_dirs)} run(s):")
    for r in run_dirs:
        print(f"  {r}")

    device = torch.device(args.device) if args.device else None
    instance_df, summary_df, win_counts_df = build_oracle_report(
        run_dirs,
        device=device,
        max_batches=args.max_eval_batches,
        ensemble_method=args.ensemble_method,
        splits=tuple(args.splits),
        test_stride=args.test_stride,
        verbose=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instance_path = out_dir / "oracle_instance_detail.csv"
    instance_df.to_csv(instance_path, index=False)
    print(f"Saved instance detail ({len(instance_df)} rows) -> {instance_path}")

    summary_path = out_dir / "oracle_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary ({len(summary_df)} methods) -> {summary_path}")

    win_path = out_dir / "oracle_win_counts.csv"
    win_counts_df.to_csv(win_path, index=False)
    print(f"Saved win counts ({len(win_counts_df)} models) -> {win_path}")


if __name__ == "__main__":
    main()
