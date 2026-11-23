"""Build a per-instance case library from a set of already-trained runs.

For each model listed (or every run found under --runs-root), reload its
checkpoint, score every (region, lookback/forecast window) instance across
the full train+val+test span in raw target units, and record which model
wins each instance.

One file is written to --out-dir:
  - case_library.csv — one row per (model, instance): mse/mae/rmse over the
    whole forecast window, plus the raw forecasted and true values for that
    window (each pipe-joined into a single cell), plus an `is_best_model`
    flag marking the row with the lowest mae within its instance group. This
    one table has everything -- group by the four instance-key columns and
    filter `is_best_model` to recover "the winning model per instance" if you
    only want that view; there's no need for a second, derived file.

All runs must share the same dataset, window shape, and split boundaries --
scores from different window configs (e.g. w12_h12 vs w6_h6) aren't
comparable instance-for-instance, so mixing them is refused rather than
silently producing a nonsensical table (see `_validate_run_consistency`).
`scripts/run_one.py` writes runs to `<out_root>/<dataset>/<window>/<model>__<task>/`
precisely so a dataset's run folder is never an ambiguous mix of window
configs -- pass --dataset/--window (e.g. w12_h12) below to point at one such
folder directly, instead of spelling out --runs-root by hand.

Usage
-----
    python scripts/build_case_library.py \\
      --dataset dc_house --window w12_h12 \\
      --models dlinear patchtst gcn_tcn stgformer \\
      --out-dir runs/dc_house/w12_h12/case_library

Omit --models to include every run found directly under --runs-root (each
subdirectory containing a config.yaml, and a checkpoint.pt unless the model
is checkpoint_optional, e.g. timesfm_zero/chronos2_zero).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from st_numeric_baselines.case_library import build_case_library
from st_numeric_baselines.experiments.run_discovery import add_runs_root_args, discover_run_dirs, resolve_runs_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_runs_root_args(p, default_out_root=str(REPO_ROOT / "runs"))
    p.add_argument("--models", nargs="+", default=None,
                   help="model names to include (matches config.yaml's model.name); omit for all runs found")
    p.add_argument("--device", type=str, default=None, help="e.g. cuda, cpu (default: each run's own config)")
    p.add_argument("--max-eval-batches", type=int, default=None, help="cap batches per split per run (debugging)")
    p.add_argument("--out-dir", type=str, required=True, help="output folder — case_library.csv is written here")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = resolve_runs_root(args)
    if not runs_root.is_dir():
        raise SystemExit(f"runs-root not found or not a directory: {runs_root}")

    run_dirs = discover_run_dirs(runs_root, args.models)
    print(f"Found {len(run_dirs)} run(s) under {runs_root}:")
    for r in run_dirs:
        print(f"  {r}")

    device = torch.device(args.device) if args.device else None
    case_library_df, detail_df = build_case_library(
        run_dirs, device=device, max_batches=args.max_eval_batches, verbose=True
    )

    group_cols = ["region_id", "lookback_start", "forecast_start", "forecast_end"]
    detail_df = detail_df.merge(
        case_library_df[group_cols + ["model_best"]], on=group_cols, how="left"
    )
    detail_df["is_best_model"] = detail_df["model"] == detail_df["model_best"]
    detail_df = detail_df.drop(columns=["model_best"])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "case_library.csv"
    detail_df.to_csv(out_path, index=False)
    print(f"Saved case library ({len(detail_df)} rows: one per model per instance, "
          f"{int(detail_df['is_best_model'].sum())} marked is_best_model) -> {out_path}")


if __name__ == "__main__":
    main()
