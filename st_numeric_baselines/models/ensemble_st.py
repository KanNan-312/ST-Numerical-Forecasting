"""EnsembleST — a fixed, equal-weight ensemble over a configurable list of
this registry's own models (mostly spatiotemporal/GNN ones).

This is deliberately the simplest possible combination rule — plain
averaging, zero extra trainable parameters — matching the "zero-parameter
ensemble" baseline that's standard in this line of literature (e.g. GC-MoE's
own paper, ``models/gc_moe.py``, compares against exactly this). Appears as
one registered model (``ensemble_st``): its own ``fit()`` trains every
selected sub-model itself on the same data, so it's usable directly via
``run_one.py --model ensemble_st`` like any other model, with no separate
orchestration script and no dependency on other runs already existing.

Sub-models are expected to be GNN models (this registry's ``GraphWindowDataset``
convention — one batch item is *all* N nodes for one time window), since
that's what "mostly ST-based" means here and it's what lets every sub-model
share the exact same batch shape without per-model special-casing. A
non-GNN sub-model (one with no ``_graph_dataloaders`` after its own
``fit()``) is rejected with a clear error rather than silently mishandled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.hparams import apply_hparams
from st_numeric_baselines.models.registry import get as get_model
from st_numeric_baselines.models.registry import register


@register("ensemble_st")
class EnsembleSTForecaster(BaseForecaster):
    """Fixed equal-weight ensemble over a configurable list of GNN sub-models."""

    name: str = "ensemble_st"

    def __init__(self) -> None:
        # Instance (not class) attributes -- sub_models/sub_model_hparams are
        # mutable, so a shared class-level default would leak edits across
        # every EnsembleSTForecaster instance.
        self.sub_models: List[str] = ["gcn_tcn", "stgformer", "dcrnn"]
        self.sub_model_hparams: Dict[str, Dict[str, Any]] = {}
        self._fitted: List[BaseForecaster] = []
        self._fitted_names: List[str] = []
        self._graph_dataloaders: Optional[Dict[str, Any]] = None  # duck-typed "graph mode" signal
        # for evaluate_forecaster/evaluate_mse_loss (metrics/evaluator.py,
        # metrics/loss.py) -- both check getattr(model, "_graph_dataloaders",
        # None), not isinstance(GNNForecasterBase), so just having this
        # attribute set is enough to route evaluation through graph mode.

    def _instantiate(self, name: str) -> BaseForecaster:
        sub = get_model(name)
        apply_hparams(sub, (self.sub_model_hparams or {}).get(name, {}))
        return sub

    def _sync_graph_dataloaders(self) -> None:
        for sub in self._fitted:
            if getattr(sub, "_graph_dataloaders", None) is not None:
                self._graph_dataloaders = sub._graph_dataloaders
                return
        raise ValueError(
            f"{self.name} requires every sub_model to be a GNN model (expose "
            f"_graph_dataloaders after fit/setup) -- got sub_models={list(self.sub_models)}, "
            "none of which did. ensemble_st is for 'mostly ST-based' models only."
        )

    # ── fit / predict ────────────────────────────────────────────────────────

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        if not self.sub_models:
            raise ValueError(f"{self.name}.sub_models must be a non-empty list of registered model names")

        dev = device if device is not None else torch.device("cpu")
        self._fitted = []
        self._fitted_names = list(self.sub_models)
        for name in self._fitted_names:
            sub = self._instantiate(name)
            sub.fit(bundle, device=dev)
            self._fitted.append(sub)
        self._sync_graph_dataloaders()

    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError(f"{self.name} must be fit() before predict_batch()")
        preds = [sub.predict_batch(batch, bundle=bundle, device=device) for sub in self._fitted]
        return torch.stack(preds, dim=0).mean(dim=0)

    # ── checkpoint: delegate to each sub-model's own save/load ─────────────────

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if not self._fitted:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sub_model_names": self._fitted_names}, p)
        for name, sub in zip(self._fitted_names, self._fitted):
            sub.save_checkpoint(p.parent / f"{p.stem}__{name}.pt")

    def load_checkpoint(self, path: Union[str, Path], *, device: Optional[torch.device] = None) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        payload = torch.load(p, map_location=device or "cpu", weights_only=False)
        names = list(payload["sub_model_names"])

        fitted: List[BaseForecaster] = []
        for name in names:
            sub = self._instantiate(name)
            sub.load_checkpoint(p.parent / f"{p.stem}__{name}.pt", device=device)
            fitted.append(sub)
        self._fitted = fitted
        self._fitted_names = names

    def setup_graph_dataloaders(self, bundle: ProcBundle) -> None:
        """Rebuild every sub-model's graph dataloaders after load_checkpoint()
        (mirrors GNNForecasterBase.setup_graph_dataloaders — called by
        experiments.run_loader.load_run the same way)."""
        for sub in self._fitted:
            if hasattr(sub, "setup_graph_dataloaders"):
                sub.setup_graph_dataloaders(bundle)
        self._sync_graph_dataloaders()
