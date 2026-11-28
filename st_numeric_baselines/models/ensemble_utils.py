"""Shared helpers for ensemble/meta-learner models (``ensemble_st``,
``xgb_stack``) that combine a mix of GNN sub-models (this registry's
``GraphWindowDataset`` convention: one batch item = all N nodes for one
time window, batch["x"] is [B,L,N,Dx]) and plain per-node sub-models
(``WindowDataset`` convention: one batch item = one node's one window,
batch["x"] is [B,L,Dx]).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.data.graph_dataset import GraphWindowDataset, graph_collate
from st_numeric_baselines.models.base import BaseForecaster

# Batch size used only for the ad hoc val-split feature collection in
# collect_stacking_features -- unrelated to any sub-model's own training
# batch size, just a tradeoff between loop overhead and memory here.
_FEATURE_COLLECTION_BATCH_SIZE = 8


def is_graph_model(model: BaseForecaster) -> bool:
    """True if `model` exposes a populated `_graph_dataloaders` (set by
    GNNForecasterBase.fit/setup_graph_dataloaders) -- the same duck-typed
    signal metrics/evaluator.py and case_library.py already key off of."""
    return getattr(model, "_graph_dataloaders", None) is not None


def adapt_batch_for_submodel(sub: BaseForecaster, batch: Dict[str, Any]) -> Dict[str, Any]:
    """Reshape a graph-shaped batch (`x`: [B,L,N,Dx], `y`: [B*N,H,Dy]) into
    the flat per-row view a non-graph sub-model's `predict_batch` expects
    (`x`: [B*N,L,Dx]), when `sub` isn't itself a GNN model. Row order is
    b*N+n, matching graph_collate's own y-flattening exactly, so `y` (and
    the node axis of `x_mark`, if present) line up unchanged / broadcast
    correctly. No-op when `sub` is a GNN model, or `batch["x"]` is already
    flat (3D, e.g. every sub_model in the ensemble is non-graph)."""
    if is_graph_model(sub):
        return batch
    x = batch["x"]
    if x.dim() != 4:
        return batch
    B, L, N, Dx = x.shape
    flat: Dict[str, Any] = {"x": x.permute(0, 2, 1, 3).reshape(B * N, L, Dx)}
    y = batch.get("y")
    if y is not None:
        flat["y"] = y
    x_mark = batch.get("x_mark")
    if x_mark is not None:
        Lm, Dm = x_mark.shape[1], x_mark.shape[2]
        flat["x_mark"] = x_mark.unsqueeze(1).expand(B, N, Lm, Dm).reshape(B * N, Lm, Dm)
    return flat


@torch.no_grad()
def collect_stacking_features(
    sub_models: Sequence[BaseForecaster],
    sub_names: Sequence[str],
    bundle: ProcBundle,
    *,
    split: str,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run every (already-fitted) sub-model over `split` and align their
    per-point predictions by (zipcode, t_pred_start), independent of
    whether each sub-model uses the graph dataloader (one batch = all N
    nodes) or the flat per-node dataloader (one batch = one node's one
    window) -- both ultimately slice the same bundle.aligned_proc.values
    for the same split, so this key is enough to match rows across the two
    conventions (and is robust to either dataset silently dropping a
    (zipcode, t0) pair, e.g. WindowDataset's own NaN filtering: only keys
    every sub_model actually produced end up in the output).

    Returns ``(X[n, H, M, Dy], Y[n, H, Dy])`` where ``M = len(sub_models)``.
    """
    pred_len = int(bundle.raw.spec.pred_len)
    seq_len = int(bundle.raw.spec.seq_len)
    zipcodes = list(bundle.aligned_proc.zipcodes)
    N = len(zipcodes)

    preds_by_model: Dict[str, Dict[Tuple[str, int], np.ndarray]] = {n: {} for n in sub_names}
    true_by_key: Dict[Tuple[str, int], np.ndarray] = {}

    for name, sub in zip(sub_names, sub_models):
        if is_graph_model(sub):
            ds = GraphWindowDataset(bundle, split=split)
            dl = DataLoader(ds, batch_size=_FEATURE_COLLECTION_BATCH_SIZE, shuffle=False, collate_fn=graph_collate)
            cursor = 0
            for batch in dl:
                y_true = batch["y"][:, -pred_len:, :]  # [Bc*N, H, Dy]
                y_pred = sub.predict_batch(batch, bundle=bundle, device=device)
                Bc = y_true.shape[0] // N
                H = y_true.shape[1]
                Dy = y_true.shape[2]
                t0s = ds.time_anchors[cursor : cursor + Bc]
                cursor += Bc
                yt = y_true.detach().cpu().numpy().reshape(Bc, N, H, Dy)
                yp = y_pred.detach().cpu().numpy().reshape(Bc, N, H, Dy)
                for bi, t0 in enumerate(t0s):
                    t_pred_start = int(t0) + seq_len
                    for ni, zc in enumerate(zipcodes):
                        key = (zc, t_pred_start)
                        preds_by_model[name][key] = yp[bi, ni]
                        true_by_key[key] = yt[bi, ni]
        else:
            dl = bundle.dataloaders[split]
            for batch in dl:
                y_true = batch["y"][:, -pred_len:, :]
                y_pred = sub.predict_batch(batch, bundle=bundle, device=device)
                yt = y_true.detach().cpu().numpy()
                yp = y_pred.detach().cpu().numpy()
                for bi, m in enumerate(batch["meta"]):
                    key = (m["zipcode"], int(m["t_pred_start"]))
                    preds_by_model[name][key] = yp[bi]
                    true_by_key[key] = yt[bi]

    common_keys = set(true_by_key.keys())
    for name in sub_names:
        common_keys &= set(preds_by_model[name].keys())
    if not common_keys:
        raise RuntimeError(
            "collect_stacking_features: no overlapping (zipcode, t_pred_start) instances "
            f"across sub_models={list(sub_names)} on split={split!r} -- cannot build stacking features"
        )
    keys_sorted = sorted(common_keys)

    H, Dy = true_by_key[keys_sorted[0]].shape
    M = len(sub_names)
    n = len(keys_sorted)
    X = np.empty((n, H, M, Dy), dtype=np.float32)
    Y = np.empty((n, H, Dy), dtype=np.float32)
    for i, key in enumerate(keys_sorted):
        Y[i] = true_by_key[key]
        for j, name in enumerate(sub_names):
            X[i, :, j, :] = preds_by_model[name][key]
    return X, Y
