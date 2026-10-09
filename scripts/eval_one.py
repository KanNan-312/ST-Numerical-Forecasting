"""Evaluate a previously trained model without re-training.

Usage
-----
    python scripts/eval_one.py --run-dir runs/dlinear__multivariate__w6_h3
    python scripts/eval_one.py --run-dir runs/... --splits test --device cuda
    python scripts/eval_one.py --run-dir runs/... --splits val test --max-eval-batches 50

The script reads config.yaml and checkpoint.pt from the run directory, rebuilds
the data bundle, loads the checkpoint, runs evaluation on the requested splits,
and writes eval_metrics.json back into the same run directory.
"""
from __future__ import annotations

import argparse
import sys
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from st_numeric_baselines.experiments.artifacts import save_json
from st_numeric_baselines.experiments.run_loader import load_run
from st_numeric_baselines.data.windowing import window_label
from st_numeric_baselines.metrics.evaluator import evaluate_forecaster


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
    p.add_argument("--run-dir", type=str, required=True,
                   help="Path to a run directory containing config.yaml and checkpoint.pt")
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   choices=["train", "val", "test"],
                   help="Which splits to evaluate (default: val test)")
    p.add_argument("--device", type=str, default=None,
                   help="Override device (e.g. cuda, cpu)")
    p.add_argument("--max-eval-batches", type=int, default=None,
                   help="Cap number of eval batches (0 = all)")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path (default: <run-dir>/eval_metrics.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    checkpoint_path = run_dir / "checkpoint.pt"

    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
    else:
        print(f"[warn] No checkpoint.pt found in {run_dir}. Evaluating without loading weights.")

    device_str = args.device
    dev = torch.device(device_str) if device_str else None

    model, bundle, cfg = load_run(run_dir, device=dev, require_checkpoint=False)
    dev = dev if dev is not None else torch.device(str((cfg.get("run", {}) or {}).get("device", "cpu")))

    max_eval = args.max_eval_batches
    if max_eval is not None and max_eval <= 0:
        max_eval = None

    model_name = str((cfg.get("model", {}) or {}).get("name"))

    # evaluate
    results: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "model": model_name,
        "task": (cfg.get("task", {}) or {}).get("name"),
        "window": window_label(bundle.raw.spec),
        "pipeline": bundle.pipeline.summary(),
    }

    for split in args.splits:
        print(f"Evaluating {split} ...")
        res = evaluate_forecaster(model, bundle, split=split, device=dev, max_batches=max_eval)
        results[split] = asdict(res)

    out_path = Path(args.out) if args.out else run_dir / "eval_metrics.json"
    save_json(out_path, results)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
