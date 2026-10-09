"""Build result tables from a `runs/` folder -- one row per model, one
column per forecast-config window, grouped explicitly per dataset x task
(not combined across datasets -- two different datasets' same
model+window combo would otherwise look identical, since the window
label alone never carried dataset identity before this field existed).

This is the single canonical reporting entry point (it absorbs what the
now-archived root-level `collate_metrics.py` did, plus the multi-task/
multi-metric sweep `make_report.py` already did) -- see
`archive/README.md` for why that duplicate was retired.

Usage
-----
    python scripts/make_report.py --runs runs --out runs/reports
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from st_numeric_baselines.metrics.reporting import collect_runs, pivot_metric

METRICS = ("rmse", "mae", "log_rmse", "log_mae", "mape")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=str, default=str(REPO_ROOT / "runs"),
                   help="root directory to scan for metrics.json (recursively) -- default: runs/")
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "runs" / "reports"))
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_runs(run_root)
    if df.empty:
        raise SystemExit(f"No runs found under {run_root}")

    df.to_csv(out_dir / "metrics_long.csv", index=False)
    print(f"Wrote {len(df)} rows -> {out_dir / 'metrics_long.csv'}")

    n_legacy = int(df["dataset"].isna().sum())
    if n_legacy:
        # Runs from before metrics.json carried a "dataset" field -- group
        # them under an explicit, clearly-labeled bucket instead of
        # silently dropping them from the report.
        df["dataset"] = df["dataset"].fillna("_unknown_pre_dataset_field")
        print(f"Note: {n_legacy} row(s) predate the 'dataset' field -- "
              "grouped under dataset='_unknown_pre_dataset_field'; re-run them to get a real label.")
    datasets = sorted(df["dataset"].dropna().unique().tolist())

    n_written = 0
    for dataset in datasets:
        sub_ds = df if dataset is None else df[df["dataset"] == dataset]
        tasks = sorted(sub_ds["task"].dropna().unique().tolist())
        for task in tasks:
            for metric in METRICS:
                piv = pivot_metric(sub_ds, task=task, split=args.split, metric=metric, dataset=dataset)
                if piv.empty:
                    continue
                label = f"{dataset or 'all'}__{task}__{metric}"
                piv.to_csv(out_dir / f"pivot_{args.split}_{label}.csv")
                try:
                    md = piv.to_markdown()
                except ImportError:
                    md = None  # optional `tabulate` dep not installed -- CSV above is authoritative either way
                if md is not None:
                    with open(out_dir / f"pivot_{args.split}_{label}.md", "w", encoding="utf-8") as f:
                        f.write(md)
                n_written += 1

    print(f"Wrote {n_written} pivot table(s) (csv + md) for {len(datasets)} dataset(s) -> {out_dir}")


if __name__ == "__main__":
    main()
