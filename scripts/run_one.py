from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]  # requires: pip install -e . (see README Quick start)

from st_numeric_baselines.data.windowing import make_window_spec, window_label
from st_numeric_baselines.experiments.artifacts import collect_env, make_run_dir, save_json, save_yaml
from st_numeric_baselines.experiments.sweep import run_one_cfg
from st_numeric_baselines.utils.config import deep_update, load_yaml, pop_cli_overrides, resolve_relpaths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--config-dir", type=str, default=str(REPO_ROOT / "configs"))
    p.add_argument(
        "--dataset",
        type=str,
        default="dc_house",
        help="dataset config name under configs/dataset (target/feature cols, path, graph settings, ...)",
    )
    p.add_argument("--task", type=str, default="multivariate", choices=["univariate", "multivariate"])
    p.add_argument(
        "--model",
        type=str,
        default="stsgcn",
        help="model config name under configs/models",
    )

    # common overrides
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--n-zip", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max-eval-batches", type=int, default=None)

    # window/split overrides — all default to None so the merged YAML config
    # (default.yaml, optionally overridden per-dataset) is the source of truth
    # unless the flag is explicitly passed on the command line.
    p.add_argument("--seq-len", type=int, default=None, help="override window.seq_len (lookback length)")
    p.add_argument("--label-len", type=int, default=None, help="override window.label_len (decoder label length)")
    p.add_argument("--pred-len", type=int, default=None, help="override window.pred_len (forecast horizon)")
    p.add_argument(
        "--test-stride",
        type=int,
        default=None,
        help="stride between consecutive test windows (>1 strides through the test set instead of "
        "sliding one step at a time, to cut evaluation cost); falls back to window.test_stride in config",
    )
    p.add_argument(
        "--train-ratio",
        type=float,
        default=None,
        help="override split.train_ratio (fraction of the pre-test span used for training)",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="override split.val_ratio (fraction of the pre-test span used for validation)",
    )
    p.add_argument(
        "--test-cutoff-date",
        type=str,
        default=None,
        help="exact date (e.g. 2022-01-01) at which the test split begins; overrides ratio-based "
        "splitting for the test boundary (train/val before it are still split by train_ratio/val_ratio)",
    )

    # artifacts
    p.add_argument("--out-root", type=str, default=str(REPO_ROOT / "runs"))
    p.add_argument("--run-name", type=str, default=None, help="optional run dir name (otherwise auto)")

    # arbitrary overrides: --set a.b=1 --set x=true
    p.add_argument("--set", dest="overrides", action="append", default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = Path(args.config_dir)

    cfg: Dict[str, Any] = {}
    deep_update(cfg, load_yaml(cfg_dir / "default.yaml"))
    deep_update(cfg, load_yaml(cfg_dir / "dataset" / f"{args.dataset}.yaml"))
    deep_update(cfg, load_yaml(cfg_dir / "task" / f"{args.task}.yaml"))
    deep_update(cfg, load_yaml(cfg_dir / "models" / f"{args.model}.yaml"))

    # apply CLI scalar overrides — only touch cfg when the flag was actually
    # passed; otherwise the merged YAML config (default.yaml, optionally
    # overridden per-dataset) stands as-is.
    if args.data is not None:
        cfg.setdefault("data", {})["path"] = args.data
    if args.n_zip is not None:
        cfg.setdefault("data", {})["n_zip"] = int(args.n_zip)
    if args.device is not None:
        cfg.setdefault("run", {})["device"] = args.device
    if args.max_eval_batches is not None:
        cfg.setdefault("run", {})["max_eval_batches"] = int(args.max_eval_batches)
    if args.seq_len is not None:
        cfg.setdefault("window", {})["seq_len"] = int(args.seq_len)
    if args.label_len is not None:
        cfg.setdefault("window", {})["label_len"] = int(args.label_len)
    if args.pred_len is not None:
        cfg.setdefault("window", {})["pred_len"] = int(args.pred_len)
    if args.test_stride is not None:
        cfg.setdefault("window", {})["test_stride"] = int(args.test_stride)
    if args.train_ratio is not None:
        cfg.setdefault("split", {})["train_ratio"] = float(args.train_ratio)
    if args.val_ratio is not None:
        cfg.setdefault("split", {})["val_ratio"] = float(args.val_ratio)
    if args.test_cutoff_date is not None:
        cfg.setdefault("split", {})["test_start_date"] = args.test_cutoff_date

    # apply --set overrides
    deep_update(cfg, pop_cli_overrides(args.overrides))

    # resolve relative paths
    resolve_relpaths(cfg, root=REPO_ROOT, keys=["data.path", "graph.path"])

    # pre-compute run dir so checkpoint can be saved during the run
    _dataset_name = str((cfg.get("dataset", {}) or {}).get("name", args.dataset))
    _model_name = str((cfg.get("model", {}) or {}).get("name", "unknown"))
    _task_name = str((cfg.get("task", {}) or {}).get("name", "unknown"))
    _window_cfg = cfg.get("window", {}) or {}
    _window_spec = make_window_spec(
        seq_len=int(_window_cfg.get("seq_len", 6)),
        pred_len=int(_window_cfg.get("pred_len", 3)),
        label_len=int(_window_cfg.get("label_len", 3)),
        test_stride=int(_window_cfg.get("test_stride", 1)),
    )
    _window_name = window_label(_window_spec)
    run_name = args.run_name or f"{_dataset_name}/{_model_name}__{_task_name}__{_window_name}"
    paths = make_run_dir(root=args.out_root, name=run_name, exist_ok=True)
    save_yaml(paths.config_path, cfg)

    # run
    result = run_one_cfg(cfg=cfg, device=cfg.get("run", {}).get("device", None), out_dir=paths.run_dir)

    save_json(paths.metrics_path, result)
    save_json(paths.env_path, collect_env())

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
