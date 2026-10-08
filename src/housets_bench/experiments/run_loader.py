"""Shared "load a finished run" sequence: config.yaml + checkpoint.pt -> (model, bundle, cfg).

Used by scripts/eval_one.py and by the case-library / explain tooling
(housets_bench.case_library, housets_bench.explain), which all need the exact
same reconstruction of a trained run from its output directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

import housets_bench.models  # noqa: F401 — populates the model registry
from housets_bench.bundles.datatypes import ProcBundle
from housets_bench.data.io import load_aligned_from_cfg
from housets_bench.experiments.sweep import apply_hparams, build_bundle_from_cfg
from housets_bench.models.base import BaseForecaster
from housets_bench.models.registry import get as get_model
from housets_bench.utils.config import deep_update, load_yaml, resolve_relpaths

REPO_ROOT = Path(__file__).resolve().parents[3]


def load_run(
    run_dir: Union[str, Path],
    *,
    device: Optional[torch.device] = None,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    require_checkpoint: bool = True,
) -> Tuple[BaseForecaster, ProcBundle, Dict[str, Any]]:
    """Rebuild a trained model + its data bundle from a run directory.

    ``cfg_overrides`` is deep-merged into the loaded config before the bundle
    is built (e.g. forcing ``window.test_stride`` to 1 for exhaustive
    per-instance analysis, which the normal benchmark stride would skip).
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    checkpoint_path = run_dir / "checkpoint.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    cfg: Dict[str, Any] = load_yaml(config_path)
    resolve_relpaths(cfg, root=REPO_ROOT, keys=["data.path", "graph.path"])
    if cfg_overrides:
        deep_update(cfg, cfg_overrides)

    dev = device if device is not None else torch.device(str((cfg.get("run", {}) or {}).get("device", "cpu")))

    aligned = load_aligned_from_cfg(cfg)

    bundle = build_bundle_from_cfg(aligned=aligned, cfg=cfg)

    model_cfg = cfg.get("model", {}) or {}
    model_name = str(model_cfg.get("name"))
    model = get_model(model_name)
    apply_hparams(model, model_cfg.get("hparams", {}) or {})

    if checkpoint_path.exists():
        model.load_checkpoint(checkpoint_path, device=dev)
        if hasattr(model, "setup_graph_dataloaders"):
            model.setup_graph_dataloaders(bundle)
    elif getattr(model, "checkpoint_optional", False):
        # No train-data-dependent fitted state (e.g. pure zero-shot foundation
        # models) — fit() is just "load the pretrained weights", so there's
        # nothing a missing checkpoint actually loses; run it directly instead
        # of treating the missing file as an error.
        model.fit(bundle, device=dev)
    elif require_checkpoint:
        raise FileNotFoundError(f"checkpoint.pt not found in {run_dir}")

    return model, bundle, cfg
