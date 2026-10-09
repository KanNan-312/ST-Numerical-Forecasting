"""Explain one instance's forecast: which input features it relied on (any
model), and — for spatiotemporal (GNN) models — which neighbor regions it
relied on.

Usage
-----
    python scripts/explain_instance.py \
      --run-dir runs/dc_house/stgformer__multivariate__w6_h3 \
      --region 20001 --lookback-start 2021-06-01 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch

from st_numeric_baselines.explain import explain_features, explain_neighbors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=str, required=True, help="path to a run directory")
    p.add_argument("--region", type=str, required=True, help="target region id (e.g. a zipcode)")
    p.add_argument("--lookback-start", type=str, required=True, help="lookback window start date, e.g. 2021-06-01")
    p.add_argument("--device", type=str, default=None, help="e.g. cuda, cpu (default: the run's own config)")
    return p.parse_args()


def _print(df: pd.DataFrame) -> None:
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(df.to_string(index=False))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else None

    print("=== Feature (variable) contribution ===")
    feat_df = explain_features(
        args.run_dir, region_id=args.region, lookback_start=args.lookback_start, device=device
    )
    _print(feat_df)

    print("\n=== Neighbor (spatial) contribution ===")
    try:
        nbr_df = explain_neighbors(
            args.run_dir, region_id=args.region, lookback_start=args.lookback_start, device=device
        )
        _print(nbr_df)
    except ValueError as e:
        print(f"(skipped: {e})")


if __name__ == "__main__":
    main()
