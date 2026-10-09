from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import time
import copy
import json
import torch
import st_numeric_baselines.models

from st_numeric_baselines.bundles import build_proc_bundle
from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.data.io import AlignedData, load_aligned_from_cfg
from st_numeric_baselines.data.split import make_split
from st_numeric_baselines.data.windowing import make_window_spec, window_label, WindowSpec
from st_numeric_baselines.graph.loader import GraphConfig
from st_numeric_baselines.metrics.evaluator import evaluate_forecaster
from st_numeric_baselines.metrics.loss import evaluate_mse_loss, extract_train_history, sync_device
from st_numeric_baselines.models.hparams import apply_hparams  # used below; also re-exported for run_loader.py
from st_numeric_baselines.models.registry import get as get_model
from st_numeric_baselines.transforms import ClipTransform, LogTransform, PCATransform, StageSpec, TransformPipeline, ZScoreTransform


def _log_dataset_summary(aligned: AlignedData, bundle: ProcBundle) -> None:
    raw = bundle.raw
    split = raw.split
    dates = raw.aligned.dates

    def _dr(start: int, end: int) -> str:
        if not dates or end <= start:
            return "?"
        s = dates[start].strftime("%Y-%m")
        e = dates[min(end - 1, len(dates) - 1)].strftime("%Y-%m")
        return f"{s} → {e}"

    date_start = dates[0].strftime("%Y-%m") if dates else "?"
    date_end = dates[-1].strftime("%Y-%m") if dates else "?"
    n_train = len(bundle.datasets["train"])
    n_val = len(bundle.datasets["val"])
    n_test = len(bundle.datasets["test"])
    t_train = split.train[1] - split.train[0]
    t_val = split.val[1] - split.val[0]
    t_test = split.test[1] - split.test[0]
    print("=" * 60)
    print("Dataset summary")
    print(f"  ZIPs: {aligned.n_zip}  |  time steps: {aligned.n_time}  ({date_start} → {date_end})  |  features: {aligned.n_features}")
    def _warn(n: int, name: str) -> str:
        return "  *** EMPTY — no valid windows ***" if n == 0 else ""

    print(f"  train: {_dr(*split.train)}  ({t_train} months,  {n_train} samples){_warn(n_train, 'train')}")
    print(f"  val:   {_dr(*split.val)}  ({t_val} months,  {n_val} samples){_warn(n_val, 'val')}")
    print(f"  test:  {_dr(*split.test)}  ({t_test} months,  {n_test} samples){_warn(n_test, 'test')}")
    print(f"  window:  seq_len={raw.spec.seq_len}  pred_len={raw.spec.pred_len}  label_len={raw.spec.label_len}  test_stride={raw.spec.test_stride}")
    print(f"  features_mode={raw.features_mode}  |  x_cols={len(bundle.x_cols)}  y_cols={len(bundle.y_cols)}")
    print(f"  graph: path={raw.graph.path}")
    print(f"  pipeline: {bundle.pipeline.summary()}")
    print("=" * 60)


def build_pipeline_from_cfg(*, schema, cfg: Dict[str, Any]) -> TransformPipeline:
    """Build TransformPipeline from config dict following order [log, clip, zscore, pca]."""
    transforms_cfg = cfg.get("transforms", {}) or {}
    order = transforms_cfg.get("order", ["log", "clip", "zscore", "pca"])

    cont_names = list(schema.continuous_cols)
    target = schema.target_col
    if target not in cont_names:
        raise ValueError(f"target_col {target!r} not found in continuous_cols")
    target_idx = cont_names.index(target)
    all_idx = list(range(len(cont_names)))

    def _idx_from_apply_to(apply_to: Dict[str, Any]) -> Optional[List[int]]:
        tgt = bool((apply_to or {}).get("target", False))
        cont = bool((apply_to or {}).get("continuous_features", False))
        if cont and tgt:
            return None  # all
        if cont and not tgt:
            return [i for i in all_idx if i != target_idx]
        if (not cont) and tgt:
            return [target_idx]
        return []  

    stages: List[StageSpec] = []

    for name in order:
        if name == "log":
            c = transforms_cfg.get("log", {}) or {}
            if not c.get("enabled", True):
                continue
            mode = str(c.get("mode", "log1p"))
            idx = _idx_from_apply_to(c.get("apply_to", {"target": True, "continuous_features": True}))
            if idx == []:
                continue
            stages.append(StageSpec(LogTransform(mode=mode), idx=None if idx is None else idx))
            continue

        if name == "clip":
            c = transforms_cfg.get("clip", {}) or {}
            if not c.get("enabled", False):
                continue
            method = str(c.get("method", "quantile"))
            idx = _idx_from_apply_to(c.get("apply_to", {"target": False, "continuous_features": True}))
            if idx == []:
                continue

            if method == "quantile":
                qcfg = c.get("quantile", {}) or {}
                lower = float(qcfg.get("lower", 0.001))
                upper = float(qcfg.get("upper", 0.999))
                tr = ClipTransform(method="quantile", lower_q=lower, upper_q=upper)

            elif method == "sigma":
                scfg = c.get("sigma", {}) or {}
                k = float(scfg.get("k", 5.0))
                tr = ClipTransform(method="sigma", sigma_k=k)

            elif method == "absolute":
                acfg = c.get("absolute", {}) or {}
                lower = acfg.get("lower", None)
                upper = acfg.get("upper", None)
                tr = ClipTransform(method="absolute", abs_lower=lower, abs_upper=upper)

            else:
                raise ValueError(f"Unknown clip.method={method!r}")

            stages.append(StageSpec(tr, idx=None if idx is None else idx))
            continue

        if name == "zscore":
            c = transforms_cfg.get("zscore", {}) or {}
            if not c.get("enabled", False):
                continue
            scope = str(c.get("scope", "global"))
            eps = float(c.get("eps", 1e-6))
            idx = _idx_from_apply_to(c.get("apply_to", {"target": True, "continuous_features": True}))
            if idx == []:
                continue
            stages.append(StageSpec(ZScoreTransform(scope=scope, eps=eps), idx=None if idx is None else idx))
            continue

        if name == "pca":
            c = transforms_cfg.get("pca", {}) or {}
            if not c.get("enabled", False):
                continue
            n_components = int(c.get("n_components", 16))
            stages.append(StageSpec(PCATransform(n_components=n_components), idx=None))
            continue

        raise ValueError(f"Unknown transform stage {name!r}")

    return TransformPipeline(stages)


def build_bundle_from_cfg(
    *,
    aligned: AlignedData,
    cfg: Dict[str, Any],
) -> ProcBundle:
    split_cfg = cfg.get("split", {}) or {}
    split = make_split(
        aligned.n_time,
        aligned.dates,
        train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        val_ratio=float(split_cfg.get("val_ratio", 0.1)),
        test_start_date=split_cfg.get("test_start_date"),
    )

    window_cfg = cfg.get("window", {}) or {}
    spec = make_window_spec(
        seq_len=int(window_cfg.get("seq_len", 6)),
        pred_len=int(window_cfg.get("pred_len", 3)),
        label_len=int(window_cfg.get("label_len", 3)),
        test_stride=int(window_cfg.get("test_stride", 1)),
    )

    # features mode
    features_mode = str(((cfg.get("task", {}) or {}).get("features_mode", cfg.get("features_mode", "MS")))).upper()

    # pipeline
    pipeline = build_pipeline_from_cfg(schema=aligned.schema, cfg=cfg)

    # dataloader
    dl_cfg = cfg.get("dataloader", {}) or {}
    batch_size = int(dl_cfg.get("batch_size", 64))
    num_workers = int(dl_cfg.get("num_workers", 0))
    pad_to = int(dl_cfg.get("pad_to", 0))
    pad_to_val: Optional[int] = None if pad_to <= 0 else pad_to

    graph_yaml = cfg.get("graph", {}) or {}
    graph_cfg = GraphConfig(path=graph_yaml.get("path"))

    bundle = build_proc_bundle(
        aligned,
        split=split,
        spec=spec,
        features_mode=features_mode,
        pipeline=pipeline,
        batch_size=batch_size,
        num_workers=num_workers,
        pad_to=pad_to_val,
        graph_cfg=graph_cfg,
    )
    return bundle


def run_one_cfg(
    *,
    cfg: Dict[str, Any],
    aligned: Optional[AlignedData] = None,
    device: Optional[str] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    t0_total = time.perf_counter()

    if aligned is None:
        aligned = load_aligned_from_cfg(cfg)
    bundle = build_bundle_from_cfg(aligned=aligned, cfg=cfg)
    _log_dataset_summary(aligned, bundle)

    model_cfg = cfg.get("model", {}) or {}
    model_name = str(model_cfg.get("name"))
    model = get_model(model_name)

    # apply hparams to model instance
    hparams = model_cfg.get("hparams", {}) or {}
    apply_hparams(model, hparams)

    run_cfg = cfg.get("run", {}) or {}
    dev = torch.device(device or run_cfg.get("device", "cpu"))

    max_eval = run_cfg.get("max_eval_batches", None)
    if max_eval is not None:
        max_eval = int(max_eval)
        if max_eval <= 0:
            max_eval = None

    # fit / eval
    sync_device(dev)
    t0_fit = time.perf_counter()
    model.fit(bundle, device=dev)
    sync_device(dev)
    fit_sec = time.perf_counter() - t0_fit

    if out_dir is not None:
        model.save_checkpoint(Path(out_dir) / "checkpoint.pt")

    sync_device(dev)
    t0_val = time.perf_counter()
    val = evaluate_forecaster(model, bundle, split="val", device=dev, max_batches=max_eval)
    sync_device(dev)
    val_eval_sec = time.perf_counter() - t0_val

    sync_device(dev)
    t0_test = time.perf_counter()
    test = evaluate_forecaster(model, bundle, split="test", device=dev, max_batches=max_eval)
    sync_device(dev)
    test_eval_sec = time.perf_counter() - t0_test

    def _safe_loss(split: str) -> Dict[str, object]:
        try:
            return evaluate_mse_loss(model, bundle, split=split, device=dev, max_batches=max_eval)
        except Exception as e:
            raise(e)
            return {"error": f"{type(e).__name__}: {e}"}

    loss_train = _safe_loss("train")
    loss_val = _safe_loss("val")
    loss_test = _safe_loss("test")

    total_sec = time.perf_counter() - t0_total

    spec = bundle.raw.spec
    out: Dict[str, Any] = {
        "dataset": (cfg.get("dataset", {}) or {}).get("name"),
        "model": model_name,
        "task": (cfg.get("task", {}) or {}).get("name"),
        "window": window_label(spec),
        # explicit forecast-config fields, alongside the packed "window" label
        # above (kept for run-dir naming) -- lets a report group/filter by
        # dataset and by individual config fields instead of only by the
        # opaque combined string.
        "forecast_config": {
            "seq_len": int(spec.seq_len),
            "label_len": int(spec.label_len),
            "pred_len": int(spec.pred_len),
            "test_stride": int(spec.test_stride),
            "features_mode": bundle.raw.features_mode,
        },
        "pipeline": bundle.pipeline.summary(),
        "val": asdict(val),
        "test": asdict(test),
        "n_train": len(bundle.datasets["train"]),
        "n_val": len(bundle.datasets["val"]),
        "n_test": len(bundle.datasets["test"]),
        "timing": {
            "fit_sec": float(fit_sec),
            "val_eval_sec": float(val_eval_sec),
            "test_eval_sec": float(test_eval_sec),
            "total_sec": float(total_sec),
        },
        "loss": {
            "space": "processed",
            "metric": "mse",
            "train": loss_train,
            "val": loss_val,
            "test": loss_test,
        },
    }

    hist = extract_train_history(model)
    if hist is not None:
        out["train_history"] = hist
    return out
