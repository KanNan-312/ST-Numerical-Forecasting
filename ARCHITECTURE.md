# st-numeric-baselines – Code Architecture

## Repository layout

```
configs/                      # YAML config fragments
  default.yaml                # base defaults (split ratios, window shape, transforms, dataloader)
  dataset/                    # one file per dataset (path, id/time/target/feature cols, graph,
                               #  data.loader for non-CSV formats, optionally window:/split: overrides)
  task/                       # univariate.yaml / multivariate.yaml  (features_mode)
  models/                     # one file per model  (model.name, model.hparams)
scripts/                      # CLI entry points -- run_one.py is the main one
st_numeric_baselines/
  data/                       # loading (csv + metr_la/pems08/nyc_*/chi_* loaders), schema,
                               #  imputation, split, windowing, dataset/graph_dataset
  graph/                      # graph.npz loading, sparse-adjacency utils
  bundles/                    # RawBundle / ProcBundle dataclasses + builder
  transforms/                 # log / clip / zscore / pca stages + pipeline
  models/                     # the full registry (base, registry, hparams) --
                               #  naive/, stats/, ml/, dl/, gnn/, foundation/, ensemble_st.py, gc_moe.py
  metrics/                    # evaluator, loss helpers, reporting (per dataset x model x forecast-config)
  experiments/                # sweep.py (run_one_cfg), run_loader.py, artifacts.py, explainable_library.py
  case_library.py, oracle_selection.py, explain.py, shap_occlusion.py
archive/                      # retired pre-existing files, kept for reference
data/                         # real datasets (dc_house, seattle_house, chicago_crime)
runs/<dataset>/<model>__<task>__<window>/   # output directory (auto-created, gitignored)
```

Not installed as a package: each script in `scripts/` puts the repo root on `sys.path`
(dependencies come from `requirements.txt`, see README "Install").

---

## Entry point: `scripts/run_one.py`

```
run_one.py
  parse_args()          # --dataset, --task, --model, --device,
                         # --seq-len/--label-len/--pred-len/--test-stride/--train-ratio/--val-ratio/--test-cutoff-date, --set …
  load + deep_update    # merge default.yaml → dataset → task → model configs
  apply CLI overrides   # only for flags explicitly passed — YAML config is the source of truth otherwise
  run_one_cfg(cfg)      # ← all real work happens here  (experiments/sweep.py)
  make_run_dir()        # runs/<dataset>/<model>__<task>__<window>/  (<window> derived from effective seq_len/pred_len)
  save_yaml / save_json # config.yaml, metrics.json, env.json
  print JSON result
```

`--set key=value` can override any nested config key at the CLI (e.g. `--set graph.path=data/g.npz`).
There is no fixed window-preset file anymore: `window.seq_len`/`label_len`/`pred_len`/`test_stride`
and `split.train_ratio`/`val_ratio`/`test_start_date` live in `configs/default.yaml` (or a
per-dataset override in `configs/dataset/<name>.yaml`), and the CLI flags above only override
them when explicitly passed.

---

## Config system

Configs are plain YAML dicts that are **deep-merged** in this order:

```
configs/default.yaml           (baseline settings: split, window, dataloader, transforms, run)
configs/dataset/<dataset>.yaml (data.* — path/id/time/target/feature cols; graph.*;
                                 optionally its own window.*/split.* overrides)
configs/task/<task>.yaml       (sets task.features_mode: S / MS)
configs/models/<model>.yaml    (sets model.name, model.hparams.*)
```

`deep_update(base, override)` recursively merges dicts; scalars override.  
CLI `--set` overrides are applied last via `pop_cli_overrides`.

---

## Data pipeline

### Step 1 – Load raw table

```
load_aligned(path, target_col, id_col, time_col, drop_cols, feature_cols)
  read_table()              # csv / parquet / xlsx
  FeatureSchema.infer(df)   # feature_cols given → schema is exactly those + target;
                             # feature_cols=None → auto-infer numeric cols (drop non-numeric/id/time/drop_cols)
  clean_raw_table()         # parse dates, normalize ids, drop non-feature cols, add year/month
  align_to_tensor()         # pivot → np.ndarray [Z, T, D]  +  three_stage_impute()
  → AlignedData(zipcodes, dates, values[Z,T,D], time_marks[T,2], schema)
```

All of the above (target/id/time/feature columns, drop list) come from one
`configs/dataset/<name>.yaml` file, selected via `--dataset`. The benchmark never reads
lat/lon or builds a graph itself — GNN models load a precomputed graph.npz instead (see
`graph/loader.py` and the GNN section in README.md).

### Step 2 – Split

```
make_split(n_time, dates, train_ratio=0.7, val_ratio=0.1, test_start_date=None)
  test_start_date=None      → make_ratio_split(): TimeSplit(train=(0, t1), val=(t1, t2), test=(t2, T))
  test_start_date="2022-01-01" → test starts at the first date >= cutoff; train/val split
                                   the remaining pre-cutoff span by train_ratio/val_ratio
```
Boundaries are integer time-step indices into the `dates` axis.

### Step 3 – Build ProcBundle

`build_proc_bundle(aligned, split, spec, features_mode, pipeline, …)`

```
1. pipeline.fit_transform(values[Z,T,D], train_range=split.train)
      → values_proc[Z,T,D']   (fit stats on train slice only)

2. select x_cols / y_cols  from features_mode
      S  → x=[target], y=[target]
      MS → x=[all features], y=[target]
      M  → x=[all], y=[all]

3. generate_window_indices(values_proc, split_range, spec, stride)
      → list of (zip_idx, time_idx) anchor tuples per split
      stride=1 for train/val; stride=spec.test_stride for test (>1 skips windows to cut eval cost)

4. WindowDataset(aligned_proc, indices, spec)
      __getitem__ returns dict:
        x       [seq_len, Dx]          encoder input
        y       [label_len+pred_len, Dy]  decoder target (label window + horizon)
        x_mark  [seq_len, 2]           year/month time marks for encoder
        y_mark  [label_len+pred_len, 2] time marks for decoder
        meta    SampleMeta(zip, t_start, t_end)

5. DataLoader with collate_fn
      batches: x[B,L,Dx], y[B,Ly,Dy], x_mark[B,L,2], y_mark[B,Ly,2], x_mask[B,L]

→ ProcBundle(raw, pipeline, aligned_proc, x_cols, y_cols, datasets, dataloaders,
             raw_target_col, raw_target_index)
```

`x_mask` is 1 for real tokens, 0 for left-padding (used when `pad_to` > `seq_len`).

---

## Transform pipeline

Stages applied in order: `log → clip → zscore → pca` (configurable).  
Each stage is fitted on the **train slice only** and applied to the full tensor.

| Transform | Config key | Fitted params |
|-----------|-----------|---------------|
| `LogTransform` | `transforms.log` | none (stateless) |
| `ClipTransform` | `transforms.clip` | quantile / sigma bounds |
| `ZScoreTransform` | `transforms.zscore` | per-feature mean, std |
| `PCATransform` | `transforms.pca` | PCA components |

`pipeline.inverse(values, keep_log=False/True)` is used at evaluation to convert predictions back to raw price space for metric computation.

---

## Model system

### `BaseForecaster` (`models/base.py`)

```python
class BaseForecaster(ABC):
    def fit(self, bundle: ProcBundle, *, device) -> None:  # optional, default no-op
    def predict_batch(self, batch, *, bundle, device) -> torch.Tensor:  # [B, H, Dy]
```

### Registry (`models/registry.py`)

```python
@register("dlinear")
class DLinearForecaster(BaseForecaster): ...
```

`get_model("dlinear")` → calls the registered factory → returns a **new instance**.  
All public class attributes (e.g. `epochs`, `lr`, `hidden_size`) become hyper-parameters that `apply_hparams(model, cfg.model.hparams)` sets via `setattr`.

### Model families

| Family | Files | Notes |
|--------|-------|-------|
| DL | `dl/{dlinear,rnn,lstm,patchtst,timemixer,informer,autoformer,fedformer,itransformer,gpt4ts,timellm}.py` | Full training loop inside `fit()` |
| ML | `ml/{rf,xgb}.py` | Flatten windows → sklearn/XGBoost fit |
| Stats / Naive | `stats/ardl.py`, `naive/ar_univariate.py` | statsmodels-free Ridge/PCA ARDL, per-ZIP AR(p) |
| Foundation | `foundation/{chronos,timesfm}.py` | zero-shot, calibrated, or fine-tuned |
| GNN | `gnn/gnn_forecaster.py` (shared `GNNForecasterBase` training loop) + one file per model: `gcn_tcn_geo, graph_wavenet, stgcn, stsgcn, stllm_plus, dcrnn, stgformer, d2stgnn, cast, stexplainer, aist, st_hhol, staeformer, stid, agcrn, mtgnn, testam` | Graph exposed as instance state (`self._A_raw`/`self._A_norm`), not a forward-signature argument. A model sets `requires_graph = False` if it never needs one (`staeformer`, `st_hhol`, `stid`, `agcrn`, `mtgnn`, `testam`) — then `dataset.graph.path` can be left unset. |
| Ensembles / MoE | `ensemble_st.py` (fixed equal-weight), `gc_moe.py` (frozen experts + trained graph-conditioned router) | Both delegate to other registered models' own `fit()`/`predict_batch` rather than training one net themselves |

### Case library, oracle, and explainability (package-root, not a subpackage)

`case_library.py` (per-instance best-model-per-instance scoring across a set
of trained runs), `oracle_selection.py` (the selection-gap: oracle vs. best
single model vs. ensemble, upper-bound for per-instance model-selection
research), `explain.py` + `shap_occlusion.py` (occlusion-based and
grouped-Shapley feature/neighbor importance), plus
`experiments/explainable_library.py` (ties a univariate + multivariate +
STExplainer run together into one 3-model-family explanation report). See
the README's "Case library" / "Oracle" / "Explainable case library"
sections for the corresponding `scripts/build_*`/`explain_instance.py` CLI
entry points.

---

## DL training loop (inside `fit()`)

All eight DL models follow the same structure:

```python
def fit(self, bundle, *, device):
    train_dl = bundle.dataloaders["train"]
    val_dl   = bundle.dataloaders["val"]
    net = <ModelNet>(...).to(dev)
    opt = Adam(net.parameters(), lr=self.lr)

    best_val, best_state, bad_epochs = inf, None, 0
    _train_total = min(len(train_dl), max_train_batches) if max_train_batches else len(train_dl)

    epoch_bar = tqdm(range(self.epochs), desc="[model_name]", unit="ep")
    for ep in epoch_bar:

        # ── train ────────────────────────────────────────────
        net.train()
        train_bar = tqdm(train_dl, desc="  train", leave=False)
        for bi, batch in enumerate(train_bar):
            if max_train_batches and bi >= max_train_batches: break
            x, y_true = batch["x"], batch["y"][:, -pred_len:, :]
            # (informer/autoformer/fedformer also use x_mark, y_mark, dec_in)
            y_pred = net(x)
            loss = F.mse_loss(y_pred, y_true)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(net.parameters(), self.grad_clip)
            opt.step()
            train_bar.set_postfix({"loss": ...})

        # ── validate ──────────────────────────────────────────
        net.eval()
        with torch.no_grad():
            val_bar = tqdm(val_dl, desc="  val  ", leave=False)
            for batch in val_bar:
                ...accumulate SSE...
                val_bar.set_postfix({"mse": ...})
        val_mse = sse / n

        # ── early stopping ───────────────────────────────────
        if val_mse < best_val - 1e-12:
            best_val = val_mse
            best_state = deepcopy(net.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= self.patience: break

        epoch_bar.set_postfix({"train": ..., "val": ..., "best": ...})
        self.train_history.append({"epoch": ep+1, "train_mse": ..., "val_mse": ..., ...})

    net.load_state_dict(best_state)   # restore best checkpoint
```

`self.train_history` is a list of per-epoch dicts that ends up in `metrics.json` under `"train_history"`.

**Decoder-based models** (Informer, Autoformer, FEDformer) additionally:
- Build `dec_in = self._make_decoder_input(y_full, label_len, pred_len)` — the label window concatenated with a zero-padded forecast horizon.
- Pass `(x, x_mark, dec_in, y_mark)` to the network.

---

## Evaluation loop

Called twice after training: once for `"val"`, once for `"test"`.

```python
evaluate_forecaster(model, bundle, split="test", device, max_batches)
  → EvalResult(log_rmse, rmse, mape, mae, log_mae, n_points)
```

Internally uses `StreamingEvaluator` which:

1. Calls `model.predict_batch(batch, bundle, device)` → `y_pred [B, H, Dy]` (processed space)
2. Embeds predictions into the full feature dimension
3. Calls `pipeline.inverse(…, keep_log=False)` → fully inverted **raw price** values (pipeline-agnostic; correct for any model regardless of internal transforms)
4. Computes all five metrics on those raw prices in a single pass:
   - `log_rmse` — `RMSE(log1p(p_raw), log1p(t_raw))`
   - `rmse`     — `RMSE(p_raw, t_raw)` in dollar space
   - `mape`     — `mean |p_raw − t_raw| / |t_raw|`
   - `mae`      — `mean |p_raw − t_raw|`
   - `log_mae`  — `MAE(log1p(p_raw), log1p(t_raw))`

A second pass via `evaluate_mse_loss` computes **processed-space MSE** (no inverse transform) for all three splits; this is what the DL training loop minimises.

---

## Experiment runner: `run_one_cfg` (`experiments/sweep.py`)

```
run_one_cfg(cfg, device)
  load_aligned_from_cfg(cfg)          # Step 1: dispatches on data.loader (csv / metr_la /
                                       #  pems08 / nyc_taxi / chi_taxi / nyc_bike / nyc_crime)
  [optional: subsample n_zip ZIPs]
  build_bundle_from_cfg(aligned, cfg) # Steps 2–5 above
  _log_dataset_summary(aligned, bundle)  # prints ZIPs × T, split sizes, window, pipeline
  get_model(model.name)               # instantiate from registry
  apply_hparams(model, cfg.model.hparams)
  model.fit(bundle, device)           # training loop (tqdm bars visible here)
  evaluate_forecaster(model, bundle, split="val")
  evaluate_forecaster(model, bundle, split="test")
  evaluate_mse_loss(model, bundle, split="train/val/test")
  extract_train_history(model)
  → dict with "dataset", "model", "task", "window", "forecast_config"
              (seq_len/label_len/pred_len/test_stride/features_mode, as
              explicit fields -- not just the packed "window" string --
              so metrics/reporting.py can group/filter per dataset x model
              x forecast-config instead of only by the combined label),
              "val", "test", "n_train/val/test", "timing", "loss",
              "pipeline", ["train_history"]
```

---

## Output artefacts

`make_run_dir(root, name)` creates `runs/<name>/` containing:

| File | Contents |
|------|----------|
| `config.yaml` | Fully merged config used for the run |
| `metrics.json` | Return dict from `run_one_cfg` (val/test metrics, timing, loss, train history) |
| `env.json` | Python / package versions at run time |

---

## Key dataclasses at a glance

| Class | Location | Purpose |
|-------|----------|---------|
| `AlignedData` | `data/io.py` | Raw tensor `[Z, T, D]` + metadata before transforms |
| `FeatureSchema` | `data/schema.py` | Column roles: id, time, target, continuous, drop |
| `TimeSplit` | `data/split.py` | Integer index ranges for train/val/test |
| `WindowSpec` | `data/windowing.py` | `seq_len`, `label_len`, `pred_len`, `test_stride` |
| `GraphConfig` | `graph/loader.py` | `path` to a graph.npz (A + ids arrays) — from dataset config |
| `RawBundle` | `bundles/datatypes.py` | `AlignedData + TimeSplit + WindowSpec + features_mode + GraphConfig` |
| `ProcBundle` | `bundles/datatypes.py` | Everything a model needs: processed data, dataloaders, pipeline |
| `EvalResult` | `metrics/evaluator.py` | `log_rmse`, `rmse`, `mape`, `mae`, `log_mae`, `n_points` |
