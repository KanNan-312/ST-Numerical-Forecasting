"""XGBStack — a learned (gradient-boosted-trees) meta-learner that stacks
predictions from a configurable list of this registry's own models.

Unlike ``ensemble_st``'s fixed equal-weight average, this *learns* a
(possibly nonlinear) combination rule:

1. ``fit()`` trains every sub-model itself on the training split, exactly
   like ``ensemble_st`` does.
2. It then runs every (now-fitted) sub-model over the **validation split**
   and collects their predictions there — a genuine out-of-sample forecast
   for every sub-model, since none of them ever trained on val rows — via
   ``ensemble_utils.collect_stacking_features``.
3. One ``XGBRegressor`` per target channel is fit on
   ``[sub-model predictions at a point] -> [true value at that point]``
   using those out-of-sample val predictions as its only training data.

At inference time (``predict_batch``, called by the usual
evaluate_forecaster/case_library machinery on the test split) the same
learned combiner is applied to the sub-models' test-split predictions —
so the combiner only ever sees genuinely out-of-sample base forecasts,
both when it was trained and when it's deployed.

Sub-models may mix GNN models (``GraphWindowDataset`` convention — one
batch item is all N nodes for one time window) and plain per-node models
(``WindowDataset`` convention): see ``ensemble_st.py``'s module docstring
for exactly the same batch-shape adaptation reused here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from xgboost import XGBRegressor

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.ensemble_utils import adapt_batch_for_submodel, collect_stacking_features, is_graph_model
from st_numeric_baselines.models.hparams import apply_hparams
from st_numeric_baselines.models.registry import get as get_model
from st_numeric_baselines.models.registry import register


@register("xgb_stack")
class XGBStackForecaster(BaseForecaster):
    """Learned XGBoost meta-learner over a configurable list of this registry's own models."""

    name: str = "xgb_stack"

    def __init__(self) -> None:
        # Instance (not class) attributes -- see ensemble_st.py for why.
        self.sub_models: List[str] = [
            "dcrnn", "stid", "graph_wavenet", "staeformer",
            "dlinear", "itransformer", "patchtst", "timemixer",
        ]
        self.sub_model_hparams: Dict[str, Dict[str, Any]] = {}

        # Meta-learner (XGBRegressor) hyperparameters -- same defaults as
        # models/ml/xgb.py's own XGBForecaster.
        self.n_estimators: int = 300
        self.learning_rate: float = 0.05
        self.max_depth: int = 6
        self.subsample: float = 0.8
        self.colsample_bytree: float = 0.8
        self.reg_lambda: float = 1.0
        self.random_state: int = 0

        self._fitted: List[BaseForecaster] = []
        self._fitted_names: List[str] = []
        self._meta_models: List[XGBRegressor] = []
        self._graph_dataloaders: Optional[Dict[str, Any]] = None  # duck-typed "graph mode" signal

    def _instantiate(self, name: str) -> BaseForecaster:
        sub = get_model(name)
        apply_hparams(sub, (self.sub_model_hparams or {}).get(name, {}))
        return sub

    def _sync_graph_dataloaders(self) -> None:
        for sub in self._fitted:
            if is_graph_model(sub):
                self._graph_dataloaders = sub._graph_dataloaders
                return
        raise ValueError(
            f"{self.name} requires at least one sub_model to be a GNN model (expose "
            f"_graph_dataloaders after fit) -- got sub_models={list(self.sub_models)}, none of which did."
        )

    def _make_regressor(self) -> XGBRegressor:
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
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

        X_val, Y_val = collect_stacking_features(
            self._fitted, self._fitted_names, bundle, split="val", device=dev,
        )
        n, H, M, Dy = X_val.shape
        X_flat = X_val.reshape(n * H, M, Dy)
        Y_flat = Y_val.reshape(n * H, Dy)

        self._meta_models = []
        for d in range(Dy):
            reg = self._make_regressor()
            reg.fit(X_flat[:, :, d], Y_flat[:, d])
            self._meta_models.append(reg)

    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if not self._fitted or not self._meta_models:
            raise RuntimeError(f"{self.name} must be fit() before predict_batch()")

        preds = [
            sub.predict_batch(adapt_batch_for_submodel(sub, batch), bundle=bundle, device=device)
            for sub in self._fitted
        ]
        stacked = torch.stack(preds, dim=0).detach().cpu().numpy()  # [M, B, H, Dy]
        M, B, H, Dy = stacked.shape
        if Dy != len(self._meta_models):
            raise ValueError(f"{self.name} Dy mismatch: batch has Dy={Dy}, trained for Dy={len(self._meta_models)}")

        out = np.empty((B, H, Dy), dtype=np.float32)
        for d in range(Dy):
            feat = stacked[:, :, :, d].transpose(1, 2, 0).reshape(B * H, M)  # [B*H, M]
            out[:, :, d] = self._meta_models[d].predict(feat).reshape(B, H)

        y_hat = torch.from_numpy(out)
        if device is not None:
            y_hat = y_hat.to(device)
        return y_hat

    # ── checkpoint: delegate sub-models to their own save/load, pickle the meta-models ─────

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        if not self._fitted:
            return
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sub_model_names": self._fitted_names, "meta_models": self._meta_models}, p)
        for name, sub in zip(self._fitted_names, self._fitted):
            sub.save_checkpoint(p.parent / f"{p.stem}__{name}.pt")

    def load_checkpoint(self, path: Union[str, Path], *, device: Optional[torch.device] = None) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        payload = torch.load(p, map_location=device or "cpu", weights_only=False)
        names = list(payload["sub_model_names"])
        self._meta_models = list(payload["meta_models"])

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
