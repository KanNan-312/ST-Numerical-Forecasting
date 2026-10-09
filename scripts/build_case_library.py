"""Build a per-instance case library from a set of already-trained runs.

For each model listed (or every run found under --runs-root), reload its
checkpoint, score every (region, lookback/forecast window) instance across
the full train+val+test span in raw target units, and record which model
wins each instance.

Two files are written under --out-dir:
  - case_library_detail.csv — one row per (model, instance): mse/mae/rmse
    over the whole forecast window, plus the raw forecasted and true values
    for that window, each pipe-joined into a single cell.
  - case_library.csv — one row per instance: the winning model only
    (derived from the detail table).

Usage
-----
    python scripts/build_case_library.py \
      --runs-root runs/dc_house \
      --models dlinear patchtst gcn_tcn stgformer \
      --out-dir runs/dc_house/case_library

Omit --models to include every run found directly under --runs-root (each
subdirectory containing a config.yaml, and a checkpoint.pt unless the model
is checkpoint_optional, e.g. timesfm_zero/chronos2_zero).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from st_numeric_baselines.case_library import build_case_library
from st_numeric_baselines.utils.config import load_yaml

MODELS = ["timesfm_zero", "graph_wavenet"]
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=str, required=True,
                   help="directory containing one subdirectory per run (config.yaml + checkpoint.pt)")
    p.add_argument("--models", nargs="+", default=None,
                   help="model names to include (matches config.yaml's model.name); omit for all runs found")
    p.add_argument("--device", type=str, default=None, help="e.g. cuda, cpu (default: each run's own config)")
    p.add_argument("--max-eval-batches", type=int, default=None, help="cap batches per split per run (debugging)")
    p.add_argument("--out-dir", type=str, required=True,
                   help="output folder — case_library.csv and case_library_detail.csv are written here")
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

    # run_dirs = _discover_run_dirs(runs_root, args.models)
    run_dirs = _discover_run_dirs(runs_root, MODELS)
    print(f"Found {len(run_dirs)} run(s):")
    for r in run_dirs:
        print(f"  {r}")

    device = torch.device(args.device) if args.device else None
    case_library_df, detail_df = build_case_library(
        run_dirs, device=device, max_batches=args.max_eval_batches, verbose=True
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / "case_library_detail.csv"
    detail_df.to_csv(detail_path, index=False)
    print(f"Saved detail table ({len(detail_df)} rows: one per model per instance) -> {detail_path}")

    summary_path = out_dir / "case_library.csv"
    case_library_df.to_csv(summary_path, index=False)
    print(f"Saved case library summary ({len(case_library_df)} instances) -> {summary_path}")


if __name__ == "__main__":
    main()
