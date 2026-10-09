from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from st_numeric_baselines.bundles.datatypes import ProcBundle
from st_numeric_baselines.models.base import BaseForecaster
from st_numeric_baselines.models.registry import register

from .calibration import AffineCalibrator, fit_affine_calibrator


@dataclass
class _TimesFMForecastFn:
    api: str
    obj: object

    def forecast(self, inputs: List[np.ndarray], *, horizon: int) -> np.ndarray:
        if self.api == "v2p5":
            # TimesFM 2.5 API (repo master)
            point, _q = self.obj.forecast(horizon=horizon, inputs=inputs)
            point = np.asarray(point, dtype=np.float32)
            if point.ndim != 2 or point.shape[1] != horizon:
                raise ValueError(f"Unexpected TimesFM forecast shape: {point.shape}")
            return point

        if self.api == "v1":
            # TimesFM 1.x/2.0 PyPI API
            point, _q = self.obj.forecast(inputs, freq=[1] * len(inputs))  # monthly=1
            point = np.asarray(point, dtype=np.float32)
            if point.ndim != 2:
                raise ValueError(f"Unexpected TimesFM forecast shape: {point.shape}")
            if point.shape[1] < horizon:
                raise ValueError(
                    f"TimesFM returned horizon {point.shape[1]} < requested {horizon}. "
                    "Set hparams.horizon_len >= pred_len."
                )
            return point[:, :horizon]

        raise ValueError(f"Unknown TimesFM api={self.api!r}")

    # -------------------------
    # TimesFM-XReg adapter
    # -------------------------
    def supports_xreg(self) -> bool:
        return hasattr(self.obj, "forecast_with_covariates")

    def forecast_with_covariates(
        self,
        *,
        inputs: List[np.ndarray],
        horizon: int,
        dynamic_numerical_covariates: Dict[str, List[np.ndarray]],
        dynamic_categorical_covariates: Optional[Dict[str, List[np.ndarray]]] = None,
        static_numerical_covariates: Optional[Dict[str, List[np.ndarray]]] = None,
        static_categorical_covariates: Optional[Dict[str, List[np.ndarray]]] = None,
        freq: Optional[List[int]] = None,
        forecast_context_len: Optional[int] = None,
        xreg_mode: str = "xreg + timesfm",
        ridge: float = 0.0,
        force_on_cpu: bool = False,
        normalize_xreg_target_per_input: bool = True,
    ) -> np.ndarray:
        """Return XReg point forecast as float32 array [N, horizon].
        """
        if not self.supports_xreg():
            raise RuntimeError(
                "Loaded TimesFM object does not support forecast_with_covariates (XReg). "
                "Install XReg deps, e.g. `pip install 'timesfm[xreg]'` (and ensure jax/jaxlib)."
            )

        if dynamic_categorical_covariates is None:
            dynamic_categorical_covariates = {}
        if static_numerical_covariates is None:
            static_numerical_covariates = {}
        if static_categorical_covariates is None:
            static_categorical_covariates = {}

        call_kwargs: Dict[str, Any] = {
            "inputs": inputs,
            "dynamic_numerical_covariates": dynamic_numerical_covariates,
            "dynamic_categorical_covariates": dynamic_categorical_covariates,
            "static_numerical_covariates": static_numerical_covariates,
            "static_categorical_covariates": static_categorical_covariates,
            "freq": freq,
            "forecast_context_len": forecast_context_len,
            "xreg_mode": str(xreg_mode),
            "ridge": float(ridge),
            "force_on_cpu": bool(force_on_cpu),
            "normalize_xreg_target_per_input": bool(normalize_xreg_target_per_input),
            "horizon": int(horizon),
        }
        call_kwargs = {k: v for k, v in call_kwargs.items() if v is not None}
        try:
            import inspect as _inspect

            sig = _inspect.signature(self.obj.forecast_with_covariates)
            has_varkw = any(
                p.kind == _inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
            if not has_varkw:
                allowed = set(sig.parameters.keys())
                call_kwargs = {k: v for k, v in call_kwargs.items() if k in allowed}
        except Exception:
            pass

        out = self.obj.forecast_with_covariates(**call_kwargs)
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            point = out[0]
        else:
            point = out

        point = np.asarray(point, dtype=np.float32)
        if point.ndim == 1:
            point = point[None, :]
        if point.ndim != 2:
            raise ValueError(f"Unexpected TimesFM XReg forecast shape: {point.shape}")
        if point.shape[1] < horizon:
            raise ValueError(
                f"TimesFM XReg returned horizon {point.shape[1]} < requested {horizon}. "
                "Set hparams.horizon_len >= pred_len."
            )
        return point[:, :horizon]


def _load_timesfm(
    *,
    repo_id: str,
    pred_len: int,
    device: Optional[torch.device],
    infer_batch_size: int,
) -> _TimesFMForecastFn:
    """Load TimesFM lazily.

    Supports:
    - TimesFM 2.5 : TimesFM_2p5_200M_torch
    - TimesFM <= 1.3.0 PyPI: TimesFm + TimesFmHparams + TimesFmCheckpoint
    """
    try:
        import timesfm  # type: ignore
    except Exception as e:
        raise ImportError(
            "TimesFM dependency not found. Install one of:\n"
            "  - pip install timesfm[torch]   (PyPI TimesFM 1.x/2.0 API)\n"
            "  - install from source: https://github.com/google-research/timesfm (TimesFM 2.5)\n"
        ) from e

    # Prefer the newer 2.5 API if present.
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(repo_id)

    cfg = timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=max(16, int(pred_len)),
        normalize_inputs=True,
        use_continuous_quantile_head=False,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )

    try:
        model.compile(cfg)
    except Exception:
        pass

    if device is not None and hasattr(model, "to"):
        try:
            model = model.to(device)
        except Exception:
            pass

    return _TimesFMForecastFn(api="v2p5", obj=model)


    raise RuntimeError(
        "Unsupported timesfm package: could not find TimesFM_2p5_200M_torch or TimesFm in timesfm module"
    )


class _TimesFMBase(BaseForecaster):
    """Shared implementation for TimesFM zero-shot and light fine-tune."""

    name: str = "timesfm"

    def __init__(
        self,
        *,
        repo_id: str = "google/timesfm-2.5-200m-pytorch",
        infer_batch_size: int = 64,
        calibrate: bool = False,
        max_calib_batches: Optional[int] = 200,
    ) -> None:
        self.repo_id = str(repo_id)
        self.infer_batch_size = int(infer_batch_size)
        self.calibrate = bool(calibrate)
        self.max_calib_batches = None if max_calib_batches is None else int(max_calib_batches)

        # Set during fit
        self._pred_len: Optional[int] = None
        self._y2x_idx: Optional[List[int]] = None
        self._tfm: Optional[_TimesFMForecastFn] = None
        self._calibrator: Optional[AffineCalibrator] = None
        self._device: Optional[torch.device] = None

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        self._device = device
        self._pred_len = int(bundle.raw.spec.pred_len)

        x_cols = list(bundle.x_cols)
        y_cols = list(bundle.y_cols)
        self._y2x_idx = [x_cols.index(c) for c in y_cols]

        self._tfm = _load_timesfm(
            repo_id=self.repo_id,
            pred_len=self._pred_len,
            device=device,
            infer_batch_size=self.infer_batch_size,
        )

        if self.calibrate:
            self._fit_calibrator(bundle)

    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self._tfm is None or self._pred_len is None or self._y2x_idx is None:
            raise RuntimeError("TimesFM forecaster is not fitted")

        x = batch["x"]  # [B, L, Dx] (may be left-padded if pad_to is set)
        x_mask = batch.get("x_mask", None)
        if x.ndim != 3:
            raise ValueError(f"expected x as [B,L,D], got {tuple(x.shape)}")

        B = int(x.shape[0])
        Dy = len(self._y2x_idx)
        L = int(x.shape[1])
        if x_mask is None:
            ctx_y = x[:, :, self._y2x_idx]  # [B, L, Dy]
            ctx_flat = ctx_y.permute(0, 2, 1).contiguous().view(B * Dy, L)  # [B*Dy, L]
            ctx_np = ctx_flat.detach().cpu().numpy().astype(np.float32, copy=False)
            inputs: List[np.ndarray] = [ctx_np[i] for i in range(ctx_np.shape[0])]
        else:
            x_mask_cpu = x_mask.detach().cpu()
            ctx_y_cpu = x[:, :, self._y2x_idx].detach().cpu()  # [B, L, Dy]
            inputs = []
            for i in range(B):
                valid_len = int(float(x_mask_cpu[i].sum().item()))
                valid_len = max(1, min(valid_len, L))
                start = L - valid_len
                for j in range(Dy):
                    arr = ctx_y_cpu[i, start:, j].numpy().astype(np.float32, copy=False)
                    inputs.append(arr)

        preds: List[np.ndarray] = []
        for s in range(0, len(inputs), self.infer_batch_size):
            chunk = inputs[s : s + self.infer_batch_size]
            pred = self._tfm.forecast(chunk, horizon=self._pred_len)
            preds.append(pred)

        pred_np = np.concatenate(preds, axis=0)  # [B*Dy, H]
        pred_np = pred_np.reshape(B, Dy, self._pred_len).transpose(0, 2, 1)  # [B, H, Dy]

        y_hat = torch.as_tensor(pred_np, dtype=torch.float32, device=x.device)
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
    # TimesFM itself is a large frozen pretrained model re-downloaded from
    # `repo_id` rather than duplicated into checkpoint.pt; only the small
    # fitted state (calibrator + the bits needed to reconstruct predict_batch
    # without re-running fit()) is persisted. BaseForecaster's default
    # save/load only handles a populated `self._net`, which zero-shot and
    # calibration-only forecasters never set — so without this override
    # checkpoint.pt was silently never written at all.

    def _checkpoint_state(self) -> Dict[str, Any]:
        return {
            "pred_len": self._pred_len,
            "y2x_idx": self._y2x_idx,
            "calibrator": self._calibrator,
        }

    def _restore_checkpoint_state(self, state: Dict[str, Any]) -> None:
        self._pred_len = state.get("pred_len")
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
        if self._pred_len is not None:
            self._tfm = _load_timesfm(
                repo_id=self.repo_id,
                pred_len=self._pred_len,
                device=device,
                infer_batch_size=self.infer_batch_size,
            )


class _TimesFMXRegBase(_TimesFMBase):
    """TimesFM-XReg wrapper .
    """

    name: str = "timesfm_xreg"

    def __init__(
        self,
        *,
        repo_id: str = "google/timesfm-1.0-200m-pytorch",
        infer_batch_size: int = 64,
        calibrate: bool = False,
        max_calib_batches: Optional[int] = 200,
        # xreg knobs
        xreg_mode: str = "xreg + timesfm",
        ridge: float = 0.0,
        normalize_xreg_target_per_input: bool = True,
        force_on_cpu: bool = False,
        covar_horizon_fill: str = "last",  # last|zero
    ) -> None:
        super().__init__(
            repo_id=repo_id,
            infer_batch_size=infer_batch_size,
            calibrate=calibrate,
            max_calib_batches=max_calib_batches,
        )
        self.xreg_mode = str(xreg_mode)
        self.ridge = float(ridge)
        self.normalize_xreg_target_per_input = bool(normalize_xreg_target_per_input)
        self.force_on_cpu = bool(force_on_cpu)
        self.covar_horizon_fill = str(covar_horizon_fill).strip().lower()

        # Set during fit
        self._cov_idx: Optional[List[int]] = None
        self._cov_names: Optional[List[str]] = None

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        super().fit(bundle, device=device)

        x_cols = list(bundle.x_cols)
        y_cols = list(bundle.y_cols)

        # Use all non-target x columns as covariates (typical MS setup).
        y_set = set(y_cols)
        self._cov_idx = [i for i, c in enumerate(x_cols) if c not in y_set]
        self._cov_names = [x_cols[i] for i in self._cov_idx]

        # Only require XReg support when we actually have covariates to use.
        if self._cov_idx:
            assert self._tfm is not None
            if not self._tfm.supports_xreg():
                raise RuntimeError(
                    "TimesFM-XReg requested but your `timesfm` package has no forecast_with_covariates(). "
                    "Install XReg deps, e.g. `pip install 'timesfm[xreg]'` (and ensure jax/jaxlib)."
                )

    def _checkpoint_state(self) -> Dict[str, Any]:
        state = super()._checkpoint_state()
        state["cov_idx"] = self._cov_idx
        state["cov_names"] = self._cov_names
        return state

    def _restore_checkpoint_state(self, state: Dict[str, Any]) -> None:
        super()._restore_checkpoint_state(state)
        self._cov_idx = state.get("cov_idx")
        self._cov_names = state.get("cov_names")

    def _extend_covar(self, ctx: np.ndarray, horizon: int) -> np.ndarray:
        ctx = np.asarray(ctx, dtype=np.float32).reshape(-1)
        if horizon <= 0:
            return ctx
        if self.covar_horizon_fill == "zero":
            fut = np.zeros((horizon,), dtype=np.float32)
        else:
            last = float(ctx[-1]) if ctx.size > 0 else 0.0
            fut = np.full((horizon,), last, dtype=np.float32)
        return np.concatenate([ctx, fut], axis=0)

    def predict_batch(
        self,
        batch: Dict[str, Any],
        *,
        bundle: ProcBundle,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self._tfm is None or self._pred_len is None or self._y2x_idx is None:
            raise RuntimeError("TimesFM-XReg forecaster is not fitted")

        x = batch["x"]  # [B, L, Dx] (may be left-padded if pad_to is set)
        x_mask = batch.get("x_mask", None)
        if x.ndim != 3:
            raise ValueError(f"expected x as [B,L,D], got {tuple(x.shape)}")

        B = int(x.shape[0])
        Dy = len(self._y2x_idx)
        L = int(x.shape[1])
        H = int(self._pred_len)

        cov_idx = self._cov_idx or []
        cov_names = self._cov_names or []

        # If no covariates available, fall back to vanilla TimesFM.
        if not cov_idx:
            return super().predict_batch(batch, bundle=bundle, device=device)

        x_cpu = x.detach().cpu()

        # Build per-series inputs + per-covariate lists.
        inputs: List[np.ndarray] = []
        dyn_num: Dict[str, List[np.ndarray]] = {name: [] for name in cov_names}

        if x_mask is None:
            for i in range(B):
                start = 0
                for j in range(Dy):
                    y_arr = x_cpu[i, start:, self._y2x_idx[j]].numpy().astype(np.float32, copy=False)
                    inputs.append(y_arr)
                    for name, ci in zip(cov_names, cov_idx):
                        c_arr = x_cpu[i, start:, ci].numpy().astype(np.float32, copy=False)
                        dyn_num[name].append(self._extend_covar(c_arr, H))
        else:
            x_mask_cpu = x_mask.detach().cpu()
            for i in range(B):
                valid_len = int(float(x_mask_cpu[i].sum().item()))
                valid_len = max(1, min(valid_len, L))
                start = L - valid_len
                for j in range(Dy):
                    y_arr = x_cpu[i, start:, self._y2x_idx[j]].numpy().astype(np.float32, copy=False)
                    inputs.append(y_arr)
                    for name, ci in zip(cov_names, cov_idx):
                        c_arr = x_cpu[i, start:, ci].numpy().astype(np.float32, copy=False)
                        dyn_num[name].append(self._extend_covar(c_arr, H))

        # monthly series
        freq = [1] * len(inputs)

        preds: List[np.ndarray] = []
        for s in range(0, len(inputs), self.infer_batch_size):
            chunk_inputs = inputs[s : s + self.infer_batch_size]
            chunk_dyn = {k: v[s : s + self.infer_batch_size] for k, v in dyn_num.items()}
            chunk_freq = freq[s : s + self.infer_batch_size]

            pred = self._tfm.forecast_with_covariates(
                inputs=chunk_inputs,
                horizon=H,
                dynamic_numerical_covariates=chunk_dyn,
                dynamic_categorical_covariates={},
                static_numerical_covariates={},
                static_categorical_covariates={},
                freq=chunk_freq,
                xreg_mode=self.xreg_mode,
                ridge=self.ridge,
                force_on_cpu=self.force_on_cpu,
                normalize_xreg_target_per_input=self.normalize_xreg_target_per_input,
            )
            preds.append(pred)

        pred_np = np.concatenate(preds, axis=0)  # [B*Dy, H]
        pred_np = pred_np.reshape(B, Dy, H).transpose(0, 2, 1)  # [B, H, Dy]

        y_hat = torch.as_tensor(pred_np, dtype=torch.float32, device=x.device)
        if self._calibrator is not None:
            y_hat = self._calibrator(y_hat)
        return y_hat


@register("timesfm_zero")
class TimesFMZeroForecaster(_TimesFMBase):
    """TimesFM zero-shot."""

    name: str = "timesfm_zero"
    checkpoint_optional: bool = True

    def __init__(
        self,
        *,
        repo_id: str = "google/timesfm-1.0-200m-pytorch",
        infer_batch_size: int = 64,
    ) -> None:
        super().__init__(repo_id=repo_id, infer_batch_size=infer_batch_size, calibrate=False)


@register("timesfm_ft")
class TimesFMCalibratedForecaster(_TimesFMBase):
    """TimesFM + lightweight affine calibration head .
    """

    name: str = "timesfm_ft"

    def __init__(
        self,
        *,
        repo_id: str = "google/timesfm-1.0-200m-pytorch",
        infer_batch_size: int = 64,
        max_calib_batches: Optional[int] = 200,
    ) -> None:
        super().__init__(
            repo_id=repo_id,
            infer_batch_size=infer_batch_size,
            calibrate=True,
            max_calib_batches=max_calib_batches,
        )


@register("timesfm_xreg_zero")
class TimesFMXRegZeroForecaster(_TimesFMXRegBase):
    """TimesFM-XReg zero-shot (MS uses covariates)."""

    name: str = "timesfm_xreg_zero"
    checkpoint_optional: bool = True

    def __init__(
        self,
        *,
        repo_id: str = "google/timesfm-1.0-200m-pytorch",
        infer_batch_size: int = 64,
        xreg_mode: str = "xreg + timesfm",
        ridge: float = 0.0,
        covar_horizon_fill: str = "last",
    ) -> None:
        super().__init__(
            repo_id=repo_id,
            infer_batch_size=infer_batch_size,
            calibrate=False,
            xreg_mode=xreg_mode,
            ridge=ridge,
            covar_horizon_fill=covar_horizon_fill,
        )


@register("timesfm_xreg_ft")
class TimesFMXRegCalibratedForecaster(_TimesFMXRegBase):
    """TimesFM-XReg + lightweight affine calibration."""

    name: str = "timesfm_xreg_ft"

    def __init__(
        self,
        *,
        repo_id: str = "google/timesfm-1.0-200m-pytorch",
        infer_batch_size: int = 64,
        max_calib_batches: Optional[int] = 200,
        xreg_mode: str = "xreg + timesfm",
        ridge: float = 0.0,
        covar_horizon_fill: str = "last",
    ) -> None:
        super().__init__(
            repo_id=repo_id,
            infer_batch_size=infer_batch_size,
            calibrate=True,
            max_calib_batches=max_calib_batches,
            xreg_mode=xreg_mode,
            ridge=ridge,
            covar_horizon_fill=covar_horizon_fill,
        )


@dataclass
class _TimesFMFTConfig:
    """Full fine-tuning hyperparameters for TimesFM."""

    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    epochs: int = 1
    max_train_batches: Optional[int] = 200
    max_train_steps: Optional[int] = None

    # TimesFM uses a categorical frequency indicator in {0,1,2}.
    # 1 is recommended for weekly/monthly series.
    freq_type: int = 1

    load_path: Optional[str] = None
    save_path: Optional[str] = None


def _extract_timesfm_torch_model(tfm: _TimesFMForecastFn) -> torch.nn.Module:
    """Extract a trainable torch.nn.Module from a loaded TimesFM object."""

    if tfm.api == "v2p5":
        model = tfm.obj
        if not isinstance(model, torch.nn.Module):
            raise RuntimeError(f"TimesFM v2.5 object is not a torch.nn.Module (got {type(model)!r})")
        return model

    if tfm.api == "v1":
        # PyPI TimesFm wrapper keeps the core torch model in a private attribute.
        for attr in ["_model", "model", "torch_model", "_torch_model", "core_model", "_core_model"]:
            if hasattr(tfm.obj, attr):
                cand = getattr(tfm.obj, attr)
                if isinstance(cand, torch.nn.Module):
                    return cand
        raise RuntimeError(
            "Could not locate the underlying torch model inside timesfm.TimesFm. "
            "Expected an attribute like ._model holding a PatchedTimeSeriesDecoder."
        )

    raise RuntimeError(f"Unsupported TimesFM api={tfm.api!r}")


def _timesfm_forward_to_point(
    out: Any,
    *,
    pred_len: int,
) -> torch.Tensor:
    """Convert TimesFM torch forward output into a point-forecast tensor [N, H].
    """

    if isinstance(out, (tuple, list)):
        out = out[0]

    # Some variants might return a dict-like.
    if isinstance(out, dict):
        for k in ("mean_predictions", "point_forecast", "predictions", "mean", "forecast"):
            if k in out:
                out = out[k]
                break

    if not torch.is_tensor(out):
        raise RuntimeError(f"TimesFM forward returned non-tensor output: {type(out)!r}")

    pred = out

    # [N, P, H, K] -> take last patch -> [N, H, K]
    if pred.ndim == 4:
        pred = pred[:, -1, :, :]

    if pred.ndim == 3:
        if pred.shape[-1] <= 32 and pred.shape[-2] > 32:
            # [N, H, K] -> take mean channel 0 -> [N, H]
            pred = pred[..., 0]
        elif pred.shape[-1] > 32 and pred.shape[1] <= 32:
            # [N, P, H] -> take last patch -> [N, H]
            pred = pred[:, -1, :]
        elif pred.shape[-1] == 1:
            # [N, H, 1] -> squeeze
            pred = pred[..., 0]
        elif pred.shape[1] == 1:
            # [N, 1, H] -> squeeze patch dim
            pred = pred[:, 0, :]
        else:
            raise RuntimeError(f"Unexpected TimesFM forward 3D output shape: {tuple(pred.shape)}")

    if pred.ndim != 2:
        raise RuntimeError(f"Unexpected TimesFM forward output shape: {tuple(pred.shape)}")

    # Sanity: horizon dimension must cover pred_len.
    if pred.shape[1] < pred_len:
        raise RuntimeError(
            f"TimesFM forward returned horizon {pred.shape[1]} < required pred_len {pred_len}."
        )
    return pred


@register("timesfm_full_ft")
class TimesFMFullFineTuneForecaster(_TimesFMBase):

    name: str = "timesfm_full_ft"

    # This variant gradient-fine-tunes real weights (unlike zero-shot/calibration
    # variants), so _TimesFMBase's small calibrator-only checkpoint would be
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
        repo_id: str = "google/timesfm-1.0-200m-pytorch",
        infer_batch_size: int = 64,
        # fine-tune hparams
        ft_lr: float = 1e-4,
        ft_weight_decay: float = 0.0,
        ft_grad_clip: float = 1.0,
        ft_epochs: int = 1,
        ft_max_train_batches: Optional[int] = 200,
        ft_max_train_steps: Optional[int] = None,
        ft_freq_type: int = 1,
        ft_load_path: Optional[str] = None,
        ft_save_path: Optional[str] = None,
    ) -> None:
        super().__init__(repo_id=repo_id, infer_batch_size=infer_batch_size, calibrate=False)
        self.ft_cfg = _TimesFMFTConfig(
            lr=float(ft_lr),
            weight_decay=float(ft_weight_decay),
            grad_clip=float(ft_grad_clip),
            epochs=int(ft_epochs),
            max_train_batches=None if ft_max_train_batches is None else int(ft_max_train_batches),
            max_train_steps=None if ft_max_train_steps is None else int(ft_max_train_steps),
            freq_type=int(ft_freq_type),
            load_path=None if ft_load_path is None else str(ft_load_path),
            save_path=None if ft_save_path is None else str(ft_save_path),
        )

    def fit(self, bundle: ProcBundle, *, device: Optional[torch.device] = None) -> None:
        super().fit(bundle, device=device)
        self._full_finetune(bundle, device=device)


def _full_finetune(self, bundle: ProcBundle, *, device: Optional[torch.device]) -> None:
    if self._tfm is None or self._pred_len is None or self._y2x_idx is None:
        raise RuntimeError("TimesFMFullFineTuneForecaster is not initialized")

    torch_model = _extract_timesfm_torch_model(self._tfm)

    # Optional load.
    if self.ft_cfg.load_path is not None:
        ckpt_path = Path(self.ft_cfg.load_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"TimesFM fine-tune checkpoint not found: {ckpt_path}")
        state = torch.load(str(ckpt_path), map_location="cpu")
        torch_model.load_state_dict(state)

    dev = device if device is not None else torch.device("cpu")
    try:
        torch_model = torch_model.to(dev)
    except Exception:
        # Some versions might manage device placement internally; continue.
        pass

    # -------------------------
    # Train-scope
    # -------------------------
    scope = str(getattr(self.ft_cfg, "train_scope", "all")).strip().lower()

    def _set_requires_grad(mod: torch.nn.Module, flag: bool) -> None:
        for p in mod.parameters():
            p.requires_grad = bool(flag)

    def _unfreeze_module(mod: torch.nn.Module) -> None:
        for p in mod.parameters():
            p.requires_grad = True

    def _unfreeze_head_linears(mod: torch.nn.Module, k: int = 4) -> bool:
        # TimesFM uses linear projections near the end for forecast outputs.
        linears: List[torch.nn.Linear] = []
        for _name, m in mod.named_modules():
            if isinstance(m, torch.nn.Linear):
                linears.append(m)
        if not linears:
            return False
        for m in linears[-k:]:
            _unfreeze_module(m)
        return True

    def _unfreeze_last_block(mod: torch.nn.Module) -> bool:
        # Heuristic: pick the longest ModuleList and unfreeze its last element.
        candidates: List[torch.nn.ModuleList] = []
        for _name, m in mod.named_modules():
            if isinstance(m, torch.nn.ModuleList) and len(m) > 0:
                candidates.append(m)
        if not candidates:
            return False
        ml = max(candidates, key=lambda x: len(x))
        _unfreeze_module(ml[-1])
        return True

    if scope != "all":
        _set_requires_grad(torch_model, False)
        ok_head = _unfreeze_head_linears(torch_model, k=4)
        ok_block = True
        if scope == "last_block":
            ok_block = _unfreeze_last_block(torch_model)

        # Fallback: if we couldn't locate a head, train everything.
        if not ok_head or (scope == "last_block" and not ok_block):
            _set_requires_grad(torch_model, True)
            scope = "all"

    trainable_params = [p for p in torch_model.parameters() if p.requires_grad]
    if not trainable_params:
        # Safety fallback
        _set_requires_grad(torch_model, True)
        trainable_params = list(torch_model.parameters())
        scope = "all"

    torch_model.train()

    opt = torch.optim.AdamW(
        trainable_params,
        lr=float(self.ft_cfg.lr),
        weight_decay=float(self.ft_cfg.weight_decay),
    )

    train_loader = bundle.dataloaders["train"]
    val_loader = bundle.dataloaders.get("val", None)

    pred_len = int(self._pred_len)
    Dy = len(self._y2x_idx)

    max_batches_per_epoch = len(train_loader)
    if self.ft_cfg.max_train_batches is not None:
        max_batches_per_epoch = min(max_batches_per_epoch, int(self.ft_cfg.max_train_batches))

    max_steps: Optional[int] = None
    if self.ft_cfg.max_train_steps is not None:
        max_steps = int(self.ft_cfg.max_train_steps)

    # Fine-tune selection / early stop
    eval_every = int(getattr(self.ft_cfg, "eval_every", 1))
    patience = int(getattr(self.ft_cfg, "patience", 0))
    val_max_batches = getattr(self.ft_cfg, "val_max_batches", 50)
    if val_max_batches is not None:
        try:
            val_max_batches = int(val_max_batches)
            if val_max_batches <= 0:
                val_max_batches = None
        except Exception:
            val_max_batches = None

    mse = torch.nn.MSELoss(reduction="mean")

    def _eval_val_loss() -> float:
        if val_loader is None:
            return float("inf")
        torch_model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for bi, batch in enumerate(val_loader):
                if val_max_batches is not None and bi >= int(val_max_batches):
                    break

                x = batch["x"].to(dev)  # [B,L,Dx]
                y = batch["y"].to(dev)  # [B,label+pred,Dy]
                x_mask = batch.get("x_mask", None)
                if x_mask is None:
                    x_mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.float32, device=dev)
                else:
                    x_mask = x_mask.to(dev)

                B = int(x.shape[0])
                L = int(x.shape[1])

                ctx_y = x[:, :, self._y2x_idx]  # [B, L, Dy]
                ctx_flat = ctx_y.permute(0, 2, 1).contiguous().view(B * Dy, L)  # [B*Dy, L]

                pad = (x_mask == 0).to(dtype=torch.float32)  # [B, L]
                pad = pad.unsqueeze(1).repeat(1, Dy, 1).contiguous().view(B * Dy, L)

                y_future = y[:, -pred_len:, :]  # [B, pred_len, Dy]
                y_flat = y_future.permute(0, 2, 1).contiguous().view(B * Dy, pred_len)

                freq = torch.full((B * Dy, 1), int(self.ft_cfg.freq_type), dtype=torch.long, device=dev)

                out = torch_model(ctx_flat, pad, freq)
                pred = _timesfm_forward_to_point(out, pred_len=pred_len)[:, :pred_len]
                loss = mse(pred, y_flat)

                n = int(pred.shape[0])
                total += float(loss.item()) * n
                count += n

        torch_model.train()
        return total / max(1, count)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad = 0

    step = 0
    for epoch in range(int(self.ft_cfg.epochs)):
        for bi, batch in enumerate(train_loader):
            if bi >= max_batches_per_epoch:
                break

            x = batch["x"].to(dev)  # [B,L,Dx]
            y = batch["y"].to(dev)  # [B,label+pred,Dy]
            x_mask = batch.get("x_mask", None)
            if x_mask is None:
                x_mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.float32, device=dev)
            else:
                x_mask = x_mask.to(dev)

            B = int(x.shape[0])
            L = int(x.shape[1])

            ctx_y = x[:, :, self._y2x_idx]  # [B, L, Dy]
            ctx_flat = ctx_y.permute(0, 2, 1).contiguous().view(B * Dy, L)  # [B*Dy, L]

            pad = (x_mask == 0).to(dtype=torch.float32)  # [B, L]
            pad = pad.unsqueeze(1).repeat(1, Dy, 1).contiguous().view(B * Dy, L)

            y_future = y[:, -pred_len:, :]  # [B, pred_len, Dy]
            y_flat = y_future.permute(0, 2, 1).contiguous().view(B * Dy, pred_len)

            freq = torch.full((B * Dy, 1), int(self.ft_cfg.freq_type), dtype=torch.long, device=dev)

            out = torch_model(ctx_flat, pad, freq)
            pred = _timesfm_forward_to_point(out, pred_len=pred_len)[:, :pred_len]

            loss = mse(pred, y_flat)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if self.ft_cfg.grad_clip and self.ft_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(self.ft_cfg.grad_clip))
            opt.step()

            step += 1
            if max_steps is not None and step >= max_steps:
                break

        # Val selection + early stop
        if val_loader is not None and eval_every > 0 and ((epoch + 1) % eval_every == 0):
            v = _eval_val_loss()
            if v < best_val:
                best_val = v
                best_state = {k: t.detach().cpu().clone() for k, t in torch_model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if patience > 0 and bad >= patience:
                    break

        if max_steps is not None and step >= max_steps:
            break

    if best_state is not None:
        torch_model.load_state_dict(best_state)

    torch_model.eval()

    # Optional save.
    if self.ft_cfg.save_path is not None:
        out_path = Path(self.ft_cfg.save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch_model.state_dict(), str(out_path))
try:
    getattr(TimesFMFullFineTuneForecaster, "_full_finetune")
except Exception:
    TimesFMFullFineTuneForecaster._full_finetune = _full_finetune 
