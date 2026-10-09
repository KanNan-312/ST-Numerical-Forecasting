from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.registry import register

from .calibration import AffineCalibrator, fit_affine_calibrator


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise ImportError(
            "Chronos-2 DataFrame API requires pandas. Install with:\n"
            "  pip install pandas\n"
        ) from e
    return pd


def _load_chronos_pipeline(
    *,
    model_id: str,
    device: Optional[torch.device],
    torch_dtype: Optional[torch.dtype] = None,
) -> object:
    """Lazy import + load for Chronos (chronos-forecasting).
    """
    try:
        from chronos import BaseChronosPipeline  # type: ignore
    except Exception as e:
        raise ImportError(
            "Chronos dependency not found. Install with:\n"
            "  pip install chronos-forecasting\n"
        ) from e

    devmap = "cpu"
    if device is not None and device.type == "cuda":
        devmap = "cuda"

    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if devmap == "cuda" else torch.float32

    return BaseChronosPipeline.from_pretrained(
        model_id,
        device_map=devmap,
        torch_dtype=torch_dtype,
    )


def _load_chronos2_pipeline(
    *,
    model_id: str,
    device: Optional[torch.device],
    torch_dtype: Optional[torch.dtype] = None,
) -> object:
    """Lazy import + load for Chronos-2 (chronos-forecasting).
    """
    try:
        # Newer versions export Chronos2Pipeline at top-level.
        from chronos import Chronos2Pipeline  # type: ignore
    except Exception:
        try:
            from chronos.chronos2.pipeline import Chronos2Pipeline  # type: ignore
        except Exception as e:
            raise ImportError(
                "Chronos-2 dependency not found. Install with:\n"
                "  pip install chronos-forecasting\n"
            ) from e

    devmap = "cpu"
    if device is not None and device.type == "cuda":
        devmap = "cuda"

    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if devmap == "cuda" else torch.float32

    # HF-style: model_id can be a hub repo id or a local directory.
    return Chronos2Pipeline.from_pretrained(
        str(model_id),
        device_map=devmap,
        torch_dtype=torch_dtype,
    )


def _get_pipe_components(pipe: object) -> Tuple[torch.nn.Module, object]:

    def _first_attr(obj: object, names: List[str]) -> Optional[object]:
        for n in names:
            if hasattr(obj, n):
                return getattr(obj, n)
        return None

    model = _first_attr(pipe, ["model", "_model"])
    tok = _first_attr(pipe, ["tokenizer", "_tokenizer"])

    if model is None or tok is None:
        raise RuntimeError(
            "Could not extract model/tokenizer from Chronos pipeline. "
            "Expected attributes like .model and .tokenizer."
        )

    if not isinstance(model, torch.nn.Module):
        raise RuntimeError(f"Chronos pipeline model is not a torch.nn.Module (got {type(model)!r})")

    return model, tok


@dataclass
class _ChronosPointConfig:
    point: str = "median"  # "median" or "mean"


@dataclass
class _ChronosFTConfig:

    lr: float = 1e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # Training control
    epochs: int = 1
    max_train_batches: Optional[int] = 200
    max_train_steps: Optional[int] = None

    # Save/load
    load_path: Optional[str] = None
    save_path: Optional[str] = None


class _ChronosBase(BaseForecaster):
    name: str = "chronos"

    def __init__(
        self,
        *,
        model_id: str = "amazon/chronos-t5-small",
        infer_batch_size: int = 128,
        point: str = "median",
        calibrate: bool = False,
        max_calib_batches: Optional[int] = 200,
    ) -> None:
        self.model_id = str(model_id)
        self.infer_batch_size = int(infer_batch_size)
        self.point_cfg = _ChronosPointConfig(point=str(point))
        self.calibrate = bool(calibrate)
        self.max_calib_batches = None if max_calib_batches is None else int(max_calib_batches)

        # set during fit
        self._pipe: Optional[object] = None
        self._pred_len: Optional[int] = None
        self._seq_len: Optional[int] = None
        self._y2x_idx: Optional[List[int]] = None
        self._calibrator: Optional[AffineCalibrator] = None
        self._device: Optional[torch.device] = None

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        self._device = device
        self._pred_len = int(bundle.raw.spec.pred_len)
        self._seq_len = int(bundle.raw.spec.seq_len)

        x_cols = list(bundle.x_cols)
        y_cols = list(bundle.y_cols)
        self._y2x_idx = [x_cols.index(c) for c in y_cols]

        self._pipe = _load_chronos_pipeline(model_id=self.model_id, device=device)

        if self.calibrate:
            self._fit_calibrator(bundle)

    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self._pipe is None or self._pred_len is None or self._seq_len is None or self._y2x_idx is None:
            raise RuntimeError("Chronos forecaster is not fitted")

        x = batch["x"]  # [B,L,Dx]
        if x.ndim != 3:
            raise ValueError(f"expected x as [B,L,D], got {tuple(x.shape)}")

        B = int(x.shape[0])
        Dy = len(self._y2x_idx)
        seq_len = int(self._seq_len)
        ctx = x[:, -seq_len:, :]
        ctx_y = ctx[:, :, self._y2x_idx]  # [B, seq_len, Dy]
        ctx_flat = ctx_y.permute(0, 2, 1).contiguous().view(B * Dy, seq_len)
        ctx_flat = ctx_flat.to(dtype=torch.float32, device=torch.device("cpu"))

        preds: List[torch.Tensor] = []
        for s in range(0, ctx_flat.shape[0], self.infer_batch_size):
            chunk = ctx_flat[s : s + self.infer_batch_size]
            q, mean = self._pipe.predict_quantiles(
                chunk,
                prediction_length=self._pred_len,
                quantile_levels=[0.5],
            )

            if self.point_cfg.point.lower() == "mean":
                p = mean
            else:
                p = q[..., 0]  # median (0.5 quantile)

            if p.ndim != 2:
                raise ValueError(f"Unexpected Chronos point forecast shape: {tuple(p.shape)}")
            preds.append(p.to(dtype=torch.float32))

        pred = torch.cat(preds, dim=0)  # [B*Dy, H]
        pred = pred.view(B, Dy, self._pred_len).permute(0, 2, 1).contiguous()  # [B,H,Dy]

        y_hat = pred
        if self._calibrator is not None:
            y_hat = self._calibrator(y_hat)
        return y_hat

    def _fit_calibrator(self, bundle: ProcBundle) -> None:
        assert self._pred_len is not None

        train_loader = bundle.dataloaders["train"]
        yhat_list: List[np.ndarray] = []
        ytrue_list: List[np.ndarray] = []

        for bi, batch in enumerate(train_loader):
            if self.max_calib_batches is not None and bi >= self.max_calib_batches:
                break

            y_hat = self.predict_batch(batch, bundle=bundle, device=self._device).detach().cpu().numpy()
            y_true = batch["y"][:, -self._pred_len :, :].detach().cpu().numpy()
            yhat_list.append(y_hat)
            ytrue_list.append(y_true)

        if not yhat_list:
            raise RuntimeError("No training batches available for calibration")

        yhat_all = np.concatenate(yhat_list, axis=0)
        ytrue_all = np.concatenate(ytrue_list, axis=0)

        scale, bias = fit_affine_calibrator(yhat_all, ytrue_all)
        self._calibrator = AffineCalibrator.from_numpy(scale, bias, device=self._device)

    # ── checkpointing ────────────────────────────────────────────────────────
    # Chronos itself is a large frozen pretrained model re-downloaded from
    # `model_id` rather than duplicated into checkpoint.pt; only the small
    # fitted state (calibrator + the bits needed to reconstruct predict_batch
    # without re-running fit()) is persisted. BaseForecaster's default
    # save/load only handles a populated `self._net`, which zero-shot and
    # calibration-only forecasters never set — so without this override
    # checkpoint.pt was silently never written at all.

    def _checkpoint_state(self) -> Dict[str, Any]:
        return {
            "pred_len": self._pred_len,
            "seq_len": self._seq_len,
            "y2x_idx": self._y2x_idx,
            "calibrator": self._calibrator,
        }

    def _restore_checkpoint_state(self, state: Dict[str, Any]) -> None:
        self._pred_len = state.get("pred_len")
        self._seq_len = state.get("seq_len")
        self._y2x_idx = state.get("y2x_idx")
        self._calibrator = state.get("calibrator")

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_state(), p)

    def load_checkpoint(self, path: Union[str, Path], *, device: Optional[torch.device] = None) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        state = torch.load(p, map_location=device or "cpu", weights_only=False)
        self._restore_checkpoint_state(state)
        self._device = device
        self._pipe = _load_chronos_pipeline(model_id=self.model_id, device=device)


@register("chronos_zero")
class ChronosZeroForecaster(_ChronosBase):
    """Chronos zero-shot."""

    name: str = "chronos_zero"
    checkpoint_optional: bool = True

    def __init__(
        self,
        *,
        model_id: str = "amazon/chronos-t5-small",
        infer_batch_size: int = 128,
        point: str = "median",
    ) -> None:
        super().__init__(
            model_id=model_id,
            infer_batch_size=infer_batch_size,
            point=point,
            calibrate=False,
        )


@register("chronos_ft")
class ChronosCalibratedForecaster(_ChronosBase):
    """Chronos + lightweight affine calibration head.
    """

    name: str = "chronos_ft"

    def __init__(
        self,
        *,
        model_id: str = "amazon/chronos-t5-small",
        infer_batch_size: int = 128,
        point: str = "median",
        max_calib_batches: Optional[int] = 200,
    ) -> None:
        super().__init__(
            model_id=model_id,
            infer_batch_size=infer_batch_size,
            point=point,
            calibrate=True,
            max_calib_batches=max_calib_batches,
        )


@register("chronos_full_ft")
class ChronosFullFineTuneForecaster(_ChronosBase):
    """Chronos full fine-tuning.
    """

    name: str = "chronos_full_ft"

    # This variant gradient-fine-tunes real weights (unlike zero-shot/calibration
    # variants), so _ChronosBase's small calibrator-only checkpoint would be
    # actively misleading here — it would silently omit the fine-tuned weights.
    # Persisting the actual weights into the standard checkpoint.pt is a
    # separate follow-up (today they're only saved if `ft_save_path` is set,
    # via a side file outside the normal checkpoint mechanism); until then,
    # keep the base no-op behavior rather than writing an incomplete file.
    save_checkpoint = BaseForecaster.save_checkpoint
    load_checkpoint = BaseForecaster.load_checkpoint

    def __init__(
        self,
        *,
        model_id: str = "amazon/chronos-t5-small",
        infer_batch_size: int = 128,
        point: str = "median",
        # fine-tune hparams
        ft_lr: float = 1e-5,
        ft_weight_decay: float = 0.0,
        ft_grad_clip: float = 1.0,
        ft_epochs: int = 1,
        ft_max_train_batches: Optional[int] = 200,
        ft_max_train_steps: Optional[int] = None,
        ft_load_path: Optional[str] = None,
        ft_save_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            infer_batch_size=infer_batch_size,
            point=point,
            calibrate=False,
        )
        self.ft_cfg = _ChronosFTConfig(
            lr=float(ft_lr),
            weight_decay=float(ft_weight_decay),
            grad_clip=float(ft_grad_clip),
            epochs=int(ft_epochs),
            max_train_batches=None if ft_max_train_batches is None else int(ft_max_train_batches),
            max_train_steps=None if ft_max_train_steps is None else int(ft_max_train_steps),
            load_path=None if ft_load_path is None else str(ft_load_path),
            save_path=None if ft_save_path is None else str(ft_save_path),
        )

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        super().fit(bundle, device=device)
        self._full_finetune(bundle)

    def _full_finetune(self, bundle: ProcBundle) -> None:
        if self._pipe is None or self._pred_len is None or self._seq_len is None or self._y2x_idx is None:
            raise RuntimeError("ChronosFullFineTuneForecaster is not initialized")

        model, tok = _get_pipe_components(self._pipe)

        # Optional: load checkpoint.
        if self.ft_cfg.load_path is not None:
            ckpt_path = Path(self.ft_cfg.load_path)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Chronos fine-tune checkpoint not found: {ckpt_path}")
            state = torch.load(str(ckpt_path), map_location="cpu")
            model.load_state_dict(state)

        model_device = next(model.parameters()).device
        model.train()

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.ft_cfg.lr),
            weight_decay=float(self.ft_cfg.weight_decay),
        )

        train_loader = bundle.dataloaders["train"]
        seq_len = int(self._seq_len)
        pred_len = int(self._pred_len)
        Dy = len(self._y2x_idx)

        # Determine how many steps we will run (for early stop).
        max_batches_per_epoch = len(train_loader)
        if self.ft_cfg.max_train_batches is not None:
            max_batches_per_epoch = min(max_batches_per_epoch, int(self.ft_cfg.max_train_batches))

        max_steps: Optional[int] = None
        if self.ft_cfg.max_train_steps is not None:
            max_steps = int(self.ft_cfg.max_train_steps)

        step = 0
        for _epoch in range(int(self.ft_cfg.epochs)):
            for bi, batch in enumerate(train_loader):
                if bi >= max_batches_per_epoch:
                    break

                x = batch["x"]  # [B,L,Dx]
                y = batch["y"]  # [B,label+pred,Dy]
                x_mask = batch.get("x_mask", None)  # [B,L]

                B = int(x.shape[0])

                # Context: rightmost seq_len real steps.
                ctx = x[:, -seq_len:, :]
                ctx_y = ctx[:, :, self._y2x_idx]  # [B, seq_len, Dy]
                ctx_flat = ctx_y.permute(0, 2, 1).contiguous().view(B * Dy, seq_len)

                if x_mask is not None:
                    m = x_mask[:, -seq_len:]  # [B, seq_len]
                    m = m.unsqueeze(1).repeat(1, Dy, 1).contiguous().view(B * Dy, seq_len)
                    ctx_flat = ctx_flat.masked_fill(m == 0, torch.nan)

                # Target horizon: last pred_len steps.
                y_future = y[:, -pred_len:, :]  # [B, pred_len, Dy]
                y_flat = y_future.permute(0, 2, 1).contiguous().view(B * Dy, pred_len)

                # Tokenization MUST happen on CPU.
                ctx_cpu = ctx_flat.detach().to(dtype=torch.float32, device=torch.device("cpu"))
                y_cpu = y_flat.detach().to(dtype=torch.float32, device=torch.device("cpu"))

                # Tokenize context and labels.
                input_ids, attention_mask, scale = tok.context_input_transform(ctx_cpu)
                labels, labels_mask = tok.label_input_transform(y_cpu, scale)
                labels[labels_mask == 0] = -100

                # Move tokens to model device.
                input_ids = input_ids.to(model_device)
                attention_mask = attention_mask.to(model_device)
                labels = labels.to(model_device)

                opt.zero_grad(set_to_none=True)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                if loss is None:
                    raise RuntimeError("Chronos HF model did not return loss; cannot fine-tune")

                loss.backward()
                if self.ft_cfg.grad_clip and self.ft_cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.ft_cfg.grad_clip))
                opt.step()

                step += 1
                if max_steps is not None and step >= max_steps:
                    break

            if max_steps is not None and step >= max_steps:
                break

        model.eval()

        # Optional: save checkpoint.
        if self.ft_cfg.save_path is not None:
            out_path = Path(self.ft_cfg.save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(out_path))


# ============================================================
# Chronos-2
# ============================================================


@dataclass
class _Chronos2FTConfig:
    """Chronos-2 fine-tuning hyperparameters."""

    finetune_mode: str = "lora"  # "full" or "lora"
    lr: float = 1e-5
    num_steps: int = 500
    batch_size: int = 32
    logging_steps: int = 100

    load_path: Optional[str] = None
    save_path: Optional[str] = None


def _marks_to_timestamps(marks_2col: np.ndarray):
    """Convert [L,2] year/month marks into pandas timestamps."""
    pd = _require_pandas()
    years = marks_2col[:, 0].astype(int)
    months = marks_2col[:, 1].astype(int)
    return pd.to_datetime({"year": years, "month": months, "day": np.ones_like(years)})


def _pick_value_col(pred_df: Any, *, point: str) -> Any:
    """Pick the point-forecast column from Chronos2 predict_df output."""
    cols = list(pred_df.columns)

    def _has(name: str) -> bool:
        return name in pred_df.columns

    # Mean preference
    if str(point).lower() == "mean":
        for c in ("mean", "prediction", "predictions", "yhat", "forecast", "value"):
            if _has(c):
                return c

    # Median / 0.5 quantile
    # Column name might be float(0.5) or string "0.5".
    for c in cols:
        if isinstance(c, (float, int)) and abs(float(c) - 0.5) < 1e-12:
            return c
    for s in ("0.5", "0.50", "0.500", "p50", "median"):
        if _has(s):
            return s

    # Last resort: common generic columns
    for c in ("prediction", "predictions", "yhat", "forecast", "value"):
        if _has(c):
            return c

    raise RuntimeError(f"Could not find a forecast column in predict_df output: {cols}")


class _Chronos2Base(BaseForecaster):
    name: str = "chronos2"

    def __init__(
        self,
        *,
        model_id: str = "amazon/chronos-2",
        infer_batch_size: int = 100,
        point: str = "median",
        cross_learning: bool = False,
        calibrate: bool = False,
        max_calib_batches: Optional[int] = 200,
    ) -> None:
        self.model_id = str(model_id)
        self.infer_batch_size = int(infer_batch_size)
        self.point_cfg = _ChronosPointConfig(point=str(point))
        self.cross_learning = bool(cross_learning)
        self.calibrate = bool(calibrate)
        self.max_calib_batches = None if max_calib_batches is None else int(max_calib_batches)

        # set during fit
        self._pipe: Optional[object] = None
        self._pred_len: Optional[int] = None
        self._seq_len: Optional[int] = None
        self._x_cols: Optional[List[str]] = None
        self._y_cols: Optional[List[str]] = None
        self._target_col: Optional[str] = None
        self._calibrator: Optional[AffineCalibrator] = None
        self._device: Optional[torch.device] = None

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        self._device = device
        self._pred_len = int(bundle.raw.spec.pred_len)
        self._seq_len = int(bundle.raw.spec.seq_len)
        self._x_cols = list(bundle.x_cols)
        self._y_cols = list(bundle.y_cols)

        # Current benchmark task: multiple variables -> one target
        if len(self._y_cols) != 1:
            raise NotImplementedError(
                f"Chronos-2 wrapper currently supports Dy==1 only (got Dy={len(self._y_cols)}). "
                "Use features_mode='S' or 'MS'."
            )
        self._target_col = str(self._y_cols[0])

        if self._target_col not in self._x_cols:
            raise ValueError(
                f"target {self._target_col!r} must be present in x_cols for Chronos-2 DataFrame API"
            )

        self._pipe = _load_chronos2_pipeline(model_id=self.model_id, device=device)

        if self.calibrate:
            self._fit_calibrator(bundle)

    def _pipe_predict_df(self, df: Any, *, pred_len: int) -> Any:
        """Call Chronos2Pipeline.predict_df in a version-tolerant way."""
        import inspect

        if self._pipe is None or self._target_col is None:
            raise RuntimeError("Chronos2 pipeline not loaded")

        fn = getattr(self._pipe, "predict_df", None)
        if fn is None:
            raise RuntimeError(f"Chronos2 pipeline object has no predict_df: {type(self._pipe)!r}")

        try:
            sig = inspect.signature(fn)
            params = sig.parameters
        except Exception:
            params = {}

        kwargs: Dict[str, Any] = {}

        # prediction length param name
        if "prediction_length" in params:
            kwargs["prediction_length"] = int(pred_len)
        elif "horizon" in params:
            kwargs["horizon"] = int(pred_len)
        elif "pred_len" in params:
            kwargs["pred_len"] = int(pred_len)
        elif "horizon_len" in params:
            kwargs["horizon_len"] = int(pred_len)

        # target param name
        for nm in ("target", "target_col", "target_column", "target_name"):
            if nm in params:
                kwargs[nm] = str(self._target_col)
                break

        # quantiles
        q = [0.5]
        if "quantile_levels" in params:
            kwargs["quantile_levels"] = q
        elif "quantiles" in params:
            kwargs["quantiles"] = q
        elif "quantile_probs" in params:
            kwargs["quantile_probs"] = q

        # inference batch size
        for nm in ("batch_size", "inference_batch_size", "infer_batch_size"):
            if nm in params:
                kwargs[nm] = int(self.infer_batch_size)
                break

        # Some versions allow specifying id/time column names
        for nm in ("id_column", "item_id_column", "series_id_column"):
            if nm in params:
                kwargs[nm] = "item_id"
                break
        for nm in ("timestamp_column", "time_column", "ds_column"):
            if nm in params:
                kwargs[nm] = "timestamp"
                break

        for nm in ("freq", "frequency", "freq_str"):
            if nm in params:
                kwargs[nm] = "MS"
                break

        if "cross_learning" in params:
            kwargs["cross_learning"] = bool(self.cross_learning)

        if "validate_inputs" in params:
            kwargs["validate_inputs"] = True
        if "validate" in params:
            kwargs["validate"] = True

        try:
            return fn(df, **kwargs)
        except TypeError:
            # Some versions might want pred_len as second positional.
            try:
                return fn(df, int(pred_len), **kwargs)
            except TypeError:
                # Drop optional kwargs progressively.
                drop_order = [
                    "validate_inputs",
                    "validate",
                    "cross_learning",
                    "batch_size",
                    "inference_batch_size",
                    "infer_batch_size",
                    "quantile_levels",
                    "quantiles",
                    "quantile_probs",
                    "freq",
                    "frequency",
                    "freq_str",
                    "id_column",
                    "item_id_column",
                    "series_id_column",
                    "timestamp_column",
                    "time_column",
                    "ds_column",
                ]
                for d in drop_order:
                    if d in kwargs:
                        kwargs2 = dict(kwargs)
                        kwargs2.pop(d, None)
                        try:
                            return fn(df, **kwargs2)
                        except TypeError:
                            continue
                # Final minimal call.
                return fn(df)


    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            self._pipe is None
            or self._pred_len is None
            or self._seq_len is None
            or self._x_cols is None
            or self._target_col is None
        ):
            raise RuntimeError("Chronos2 forecaster is not fitted")

        pd = _require_pandas()

        x: torch.Tensor = batch["x"]  # [B,L,Dx]
        x_mark: torch.Tensor = batch["x_mark"]  # [B,L,2]

        if x.ndim != 3 or x_mark.ndim != 3:
            raise ValueError(f"expected x/x_mark as [B,L,*], got {tuple(x.shape)} / {tuple(x_mark.shape)}")

        B = int(x.shape[0])
        seq_len = int(self._seq_len)
        pred_len = int(self._pred_len)

        # Use the rightmost seq_len real steps.
        ctx = x[:, -seq_len:, :].detach().cpu().numpy()  # [B,seq_len,Dx]
        ctx_mark = x_mark[:, -seq_len:, :].detach().cpu().numpy()  # [B,seq_len,2]

        metas = batch.get("meta", [])
        ids: List[str] = []
        for i in range(B):
            try:
                m = metas[i]
                z = m.get("zipcode", "")
                t = m.get("t_pred_start", m.get("t0", i))
                ids.append(f"{z}__{t}")
            except Exception:
                ids.append(f"{i:06d}")

        dfs: List[Any] = []
        for i in range(B):
            ts = _marks_to_timestamps(ctx_mark[i])
            df_i = pd.DataFrame(ctx[i], columns=self._x_cols)

            # Chronos-2 expects at least: item_id, timestamp, and the target column.
            df_i.insert(0, "timestamp", ts)
            df_i.insert(0, "item_id", ids[i])

            dfs.append(df_i)

        context_df = pd.concat(dfs, ignore_index=True)
        context_df = context_df.sort_values(["item_id", "timestamp"]).reset_index(drop=True)

        try:
            pred_df = self._pipe_predict_df(context_df, pred_len=pred_len)
        except ValueError as e:
            msg = str(e)
            if ("Missing columns" in msg or "Missing column" in msg) and "target" in msg and self._target_col in context_df.columns:
                context_df2 = context_df.copy()
                if "target" not in context_df2.columns:
                    context_df2["target"] = context_df2[self._target_col]
                pred_df = self._pipe_predict_df(context_df2, pred_len=pred_len)
            else:
                raise


        if not hasattr(pred_df, "columns"):
            raise RuntimeError(f"Chronos2Pipeline.predict_df returned unexpected type: {type(pred_df)!r}")

        cols_set = set(pred_df.columns)

        id_col: Optional[str] = None
        for c in ("item_id", "id", "series_id", "unique_id"):
            if c in cols_set:
                id_col = c
                break
        if id_col is None:
            raise RuntimeError(f"predict_df output missing an id column (have {sorted(cols_set)})")

        time_col: Optional[str] = None
        for c in ("timestamp", "ds", "date", "time"):
            if c in cols_set:
                time_col = c
                break

        value_col = _pick_value_col(pred_df, point=self.point_cfg.point)

        for tgt_name_col in ("target", "target_name", "var", "variable"):
            if tgt_name_col in cols_set:
                try:
                    pred_df = pred_df[pred_df[tgt_name_col] == self._target_col]
                    cols_set = set(pred_df.columns)
                except Exception:
                    pass
                break

        # Normalize id values to string for safe matching.
        pred_df = pred_df.copy()
        pred_df["_id_str"] = pred_df[id_col].astype(str)

        if time_col is not None:
            pred_df = pred_df.sort_values(["_id_str", time_col]).reset_index(drop=True)
        else:
            pred_df = pred_df.sort_values(["_id_str"]).reset_index(drop=True)

        yhat_np = np.zeros((B, pred_len), dtype=np.float32)
        for i, sid in enumerate(ids):
            g = pred_df[pred_df["_id_str"] == sid]
            if len(g) < pred_len:
                raise RuntimeError(f"predict_df output has {len(g)} rows for item_id={sid!r}, expected >= {pred_len}")
            vals = g[value_col].to_numpy(dtype=np.float32, copy=False)[:pred_len]
            yhat_np[i, :] = vals

        y_hat = torch.tensor(yhat_np, dtype=torch.float32, device=x.device).unsqueeze(-1)  # [B,H,1]
        if self._calibrator is not None:
            y_hat = self._calibrator(y_hat)
        return y_hat

    def _fit_calibrator(self, bundle: ProcBundle) -> None:
        assert self._pred_len is not None

        train_loader = bundle.dataloaders["train"]
        yhat_list: List[np.ndarray] = []
        ytrue_list: List[np.ndarray] = []

        for bi, batch in enumerate(train_loader):
            if self.max_calib_batches is not None and bi >= self.max_calib_batches:
                break

            y_hat = self.predict_batch(batch, bundle=bundle, device=self._device).detach().cpu().numpy()
            y_true = batch["y"][:, -self._pred_len :, :].detach().cpu().numpy()
            yhat_list.append(y_hat)
            ytrue_list.append(y_true)

        if not yhat_list:
            raise RuntimeError("No training batches available for Chronos-2 calibration")

        yhat_all = np.concatenate(yhat_list, axis=0)
        ytrue_all = np.concatenate(ytrue_list, axis=0)
        scale, bias = fit_affine_calibrator(yhat_all, ytrue_all)
        self._calibrator = AffineCalibrator.from_numpy(scale, bias, device=self._device)

    # ── checkpointing ────────────────────────────────────────────────────────
    # Same rationale as _ChronosBase: Chronos-2 is a large frozen pretrained
    # pipeline re-downloaded from `model_id`, so only the small fitted state
    # is persisted rather than duplicating it into checkpoint.pt.

    def _checkpoint_state(self) -> Dict[str, Any]:
        return {
            "pred_len": self._pred_len,
            "seq_len": self._seq_len,
            "x_cols": self._x_cols,
            "y_cols": self._y_cols,
            "target_col": self._target_col,
            "calibrator": self._calibrator,
        }

    def _restore_checkpoint_state(self, state: Dict[str, Any]) -> None:
        self._pred_len = state.get("pred_len")
        self._seq_len = state.get("seq_len")
        self._x_cols = state.get("x_cols")
        self._y_cols = state.get("y_cols")
        self._target_col = state.get("target_col")
        self._calibrator = state.get("calibrator")

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_state(), p)

    def load_checkpoint(self, path: Union[str, Path], *, device: Optional[torch.device] = None) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        state = torch.load(p, map_location=device or "cpu", weights_only=False)
        self._restore_checkpoint_state(state)
        self._device = device
        self._pipe = _load_chronos2_pipeline(model_id=self.model_id, device=device)


@register("chronos2_zero")
class Chronos2ZeroForecaster(_Chronos2Base):
    """Chronos-2 zero-shot."""

    name: str = "chronos2_zero"
    checkpoint_optional: bool = True

    def __init__(
        self,
        *,
        model_id: str = "amazon/chronos-2",
        infer_batch_size: int = 100,
        point: str = "median",
        cross_learning: bool = False,
    ) -> None:
        super().__init__(
            model_id=model_id,
            infer_batch_size=infer_batch_size,
            point=point,
            cross_learning=cross_learning,
            calibrate=False,
        )


@register("chronos2_ft")
class Chronos2CalibratedForecaster(_Chronos2Base):
    """Chronos-2 + lightweight affine calibration head."""

    name: str = "chronos2_ft"

    def __init__(
        self,
        *,
        model_id: str = "amazon/chronos-2",
        infer_batch_size: int = 100,
        point: str = "median",
        cross_learning: bool = False,
        max_calib_batches: Optional[int] = 200,
    ) -> None:
        super().__init__(
            model_id=model_id,
            infer_batch_size=infer_batch_size,
            point=point,
            cross_learning=cross_learning,
            calibrate=True,
            max_calib_batches=max_calib_batches,
        )


@register("chronos2_full_ft")
class Chronos2FullFineTuneForecaster(_Chronos2Base):
    """Chronos-2 fine-tuning using Chronos2Pipeline.fit."""

    name: str = "chronos2_full_ft"

    # This variant fine-tunes the pipeline itself (LoRA or full), unlike
    # zero-shot/calibration variants, so _Chronos2Base's small calibrator-only
    # checkpoint would be actively misleading here. The fine-tuned pipeline is
    # only persisted if `ft_save` is set (via Chronos2Pipeline.save_pretrained,
    # a directory rather than a single file — a structural mismatch with the
    # standard single-file checkpoint.pt that needs its own follow-up); until
    # then, keep the base no-op behavior rather than writing an incomplete file.
    save_checkpoint = BaseForecaster.save_checkpoint
    load_checkpoint = BaseForecaster.load_checkpoint

    def __init__(
        self,
        *,
        model_id: str = "amazon/chronos-2",
        infer_batch_size: int = 100,
        point: str = "median",
        cross_learning: bool = False,
        # fit(...) hyperparameters
        ft_mode: str = "lora",
        ft_lr: float = 1e-5,
        ft_steps: int = 500,
        ft_batch_size: int = 32,
        ft_logging_steps: int = 100,
        ft_load: Optional[str] = None,
        ft_save: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            infer_batch_size=infer_batch_size,
            point=point,
            cross_learning=cross_learning,
            calibrate=False,
        )
        self.ft_cfg = _Chronos2FTConfig(
            finetune_mode=str(ft_mode),
            lr=float(ft_lr),
            num_steps=int(ft_steps),
            batch_size=int(ft_batch_size),
            logging_steps=int(ft_logging_steps),
            load_path=None if ft_load is None else str(ft_load),
            save_path=None if ft_save is None else str(ft_save),
        )

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        # Load pipeline as usual.
        super().fit(bundle, device=device)
        self._full_finetune(bundle)

    def _full_finetune(self, bundle: ProcBundle) -> None:
        import inspect

        if self._pred_len is None or self._x_cols is None or self._target_col is None:
            raise RuntimeError("Chronos2FullFineTuneForecaster not initialized")

        # Optional: load an already fine-tuned pipeline directory.
        if self.ft_cfg.load_path:
            self._pipe = _load_chronos2_pipeline(model_id=self.ft_cfg.load_path, device=self._device)
            return

        if self._pipe is None:
            raise RuntimeError("Chronos-2 pipeline not loaded")

        values = bundle.aligned_proc.values  # [Z,T,D]
        cols_all = list(bundle.aligned_proc.schema.continuous_cols)

        if self._target_col not in cols_all:
            raise ValueError(f"target {self._target_col!r} not found in aligned_proc columns")

        tgt_i = cols_all.index(self._target_col)
        cov_cols = [c for c in self._x_cols if c != self._target_col]
        cov_i = {c: cols_all.index(c) for c in cov_cols if c in cols_all}

        t0, t1 = bundle.raw.split.train

        # Build per-ZIP full-series training inputs (Chronos-2 will sample windows internally).
        train_inputs: List[Dict[str, Any]] = []
        for zi in range(values.shape[0]):
            target = values[zi, t0:t1, tgt_i].astype(np.float32, copy=False)
            item: Dict[str, Any] = {"target": target, "item_id": str(bundle.raw.aligned.zipcodes[zi])}
            if cov_i:
                item["past_covariates"] = {
                    c: values[zi, t0:t1, cov_i[c]].astype(np.float32, copy=False) for c in cov_cols if c in cov_i
                }
            train_inputs.append(item)

        # Prepare kwargs for Chronos2Pipeline.fit
        fit_kwargs: Dict[str, Any] = dict(
            inputs=train_inputs,
            prediction_length=int(self._pred_len),
            num_steps=int(self.ft_cfg.num_steps),
            learning_rate=float(self.ft_cfg.lr),
            batch_size=int(self.ft_cfg.batch_size),
            logging_steps=int(self.ft_cfg.logging_steps),
            finetune_mode=str(self.ft_cfg.finetune_mode),
        )

        # Filter to accepted args.
        try:
            sig = inspect.signature(self._pipe.fit)
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                allowed = set(sig.parameters.keys())
                fit_kwargs = {k: v for k, v in fit_kwargs.items() if k in allowed}
        except Exception:
            pass

        out = self._pipe.fit(**fit_kwargs)
        if out is not None:
            self._pipe = out

        # Optional: save fine-tuned pipeline directory.
        if self.ft_cfg.save_path:
            out_dir = Path(self.ft_cfg.save_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            if hasattr(self._pipe, "save_pretrained"):
                self._pipe.save_pretrained(str(out_dir))
            else:
                raise RuntimeError(
                    "Chronos2Pipeline does not implement save_pretrained() in this version; "
                    "cannot save fine-tuned weights."
                )
