from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional

import torch


def sync_device(device: Optional[torch.device]) -> None:
    if device is None:
        return
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except Exception:
        # timing should never crash an experiment
        pass


def _get_pred_len(bundle: Any) -> int:
    try:
        return int(bundle.raw.spec.pred_len)
    except Exception:
        pass

    # fallback layouts
    try:
        return int(bundle.spec.pred_len)
    except Exception:
        pass

    raise AttributeError(
        "Could not determine pred_len from bundle (expected bundle.raw.spec.pred_len or bundle.spec.pred_len)."
    )


def extract_train_history(model: Any) -> Optional[Any]:
    for attr in (
        "train_history",
        "training_history",
        "train_log",
        "_train_log",
        "history",
        "_history",
    ):
        if hasattr(model, attr):
            hist = getattr(model, attr)
            if hist is not None:
                return hist
    return None


def evaluate_mse_loss(
    model: Any,
    bundle: Any,
    *,
    split: str,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:

    if not hasattr(bundle, "dataloaders") or split not in bundle.dataloaders:
        raise KeyError(f"bundle.dataloaders missing split={split!r}")

    _graph_dls = getattr(model, "_graph_dataloaders", None)
    dl = _graph_dls[split] if (_graph_dls and split in _graph_dls) else bundle.dataloaders[split]
    pred_len = _get_pred_len(bundle)

    sse = 0.0
    n = 0
    batches = 0

    t0 = time.perf_counter()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            if max_batches is not None and bi >= int(max_batches):
                break

            if "y" not in batch:
                raise KeyError("batch missing key 'y'")

            y_hat = model.predict_batch(batch, bundle=bundle, device=device)
            y_true = batch["y"][:, -pred_len:, :]

            # tolerate univariate [B,H]
            if torch.is_tensor(y_hat) and y_hat.ndim == 2:
                y_hat = y_hat.unsqueeze(-1)

            if not torch.is_tensor(y_hat):
                raise TypeError(f"predict_batch must return a torch.Tensor, got {type(y_hat)!r}")
            if y_hat.ndim != 3:
                raise ValueError(
                    f"predict_batch must return [B,H,Dy], got shape={tuple(y_hat.shape)}"
                )

            # Accumulate on CPU for stability and to avoid GPU memory growth.
            y_hat_cpu = y_hat.detach().to(dtype=torch.float32, device=torch.device("cpu"))
            y_true_cpu = y_true.detach().to(dtype=torch.float32, device=torch.device("cpu"))

            if y_hat_cpu.shape != y_true_cpu.shape:
                raise ValueError(
                    f"shape mismatch: y_hat={tuple(y_hat_cpu.shape)} vs y_true={tuple(y_true_cpu.shape)}"
                )

            diff = y_hat_cpu - y_true_cpu
            sse += float((diff * diff).sum().item())
            n += int(diff.numel())
            batches += 1

    time_sec = time.perf_counter() - t0
    mse = sse / max(n, 1)
    rmse = math.sqrt(mse) if mse >= 0 else float("nan")

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "n": int(n),
        "batches": int(batches),
        "time_sec": float(time_sec),
    }