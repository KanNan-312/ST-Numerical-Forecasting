"""Apply a config's ``hparams`` dict onto a model instance.

Lives at this leaf (no dependency on ``st_numeric_baselines.models``'s own
registration imports) so it can be imported both by
``experiments.sweep`` (the normal config-driven entry point) and by any
model that needs to instantiate *other* registered models itself (e.g.
``ensemble_st``, ``gc_moe``) without a circular import through
``st_numeric_baselines.models.__init__`` -> that model -> back to
``experiments.sweep`` -> ``st_numeric_baselines.models``.
"""
from __future__ import annotations

from typing import Any, Dict


def _maybe_set(obj: object, name: str, value: Any) -> None:
    if hasattr(obj, name):
        try:
            setattr(obj, name, value)
        except Exception:
            pass


def apply_hparams(model: object, hparams: Dict[str, Any]) -> None:
    hparams = hparams or {}

    # Compatibility aliases
    if "mode_select" in hparams and "mode_select_method" not in hparams:
        hparams = dict(hparams)
        hparams["mode_select_method"] = hparams["mode_select"]

    for k, v in hparams.items():
        # Chronos stores point in a tiny config dataclass
        if k == "point" and hasattr(model, "point_cfg"):
            try:
                model.point_cfg.point = str(v)
            except Exception:
                pass
            continue

        _maybe_set(model, k, v)
