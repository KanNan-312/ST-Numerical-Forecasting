# st-numeric-baselines

Numeric (non-LLM) spatiotemporal forecasting baselines across region-based
domains: housing (the namesake **HouseTS** dataset — a large-scale
multimodal spatiotemporal dataset for long-horizon housing-market
forecasting at the U.S. ZIP-code level), crime, and traffic/urban-mobility
panels (METR-LA, PEMS-08, the UrbanGPT benchmark). Covers statistical/ML
baselines, deep sequence models, graph neural networks, foundation-model
wrappers, and ensemble/mixture-of-experts model-selection research
(oracle/ensemble upper-bound analysis, GC-MoE, TESTAM) — see "Project
structure" below for the full map.

HouseTS itself aligns multiple modalities under a unified ZIP-month panel:
- **Monthly housing-market indicators**
- **Monthly POI counts**
- **Annual census / socioeconomic variables** aligned to the monthly timeline
- (Dataset also includes auxiliary modalities such as aerial imagery + derived annotations; see Kaggle for full contents.)

The benchmark supports **univariate** and **multivariate** forecasting with standardized train/val/test splitting, windowing, transforms, and evaluation.

## Install

```bash
pip install -r requirements.txt
```

`torch` is not in `requirements.txt` -- install the build matching your CUDA/CPU setup
separately (https://pytorch.org/get-started/locally/). The foundation-model wrappers
additionally need `transformers`, `chronos-forecasting` and `timesfm`; install those
separately as needed.

No package install is needed: every script under `scripts/` adds the repo root to
`sys.path`, so it runs directly (`python scripts/run_one.py ...`) from any directory.

## Project structure

```
configs/
  default.yaml          base split/window/transform/dataloader defaults
  dataset/<name>.yaml    one file per dataset (path, columns, graph, split/window overrides)
  task/<name>.yaml       univariate.yaml / multivariate.yaml (features_mode)
  models/<name>.yaml     one file per registered model (hparams)
data/                    real datasets (dc_house, seattle_house, chicago_crime)
st_numeric_baselines/
  data/                  loading, schema, imputation, split, windowing, dataset/dataloader
  graph/                 graph.npz loading + sparse-adjacency utils
  bundles/               RawBundle / ProcBundle dataclasses + builder
  transforms/            log / clip / zscore / pca pipeline stages
  models/                the full model registry (base.py, registry.py, hparams.py) —
                         naive/, stats/, ml/, dl/, gnn/, foundation/, plus ensemble_st.py / gc_moe.py
  metrics/               evaluator, loss helpers, reporting (per dataset x model x forecast-config)
  experiments/           sweep.py (run_one_cfg), run_loader.py, artifacts.py, explainable_library.py
  case_library.py, oracle_selection.py, explain.py, shap_occlusion.py   # case-library / explainability
scripts/                 CLI entry points — run_one.py is the main one; see "Quick start" below
archive/                 retired pre-existing files, kept for reference (see archive/README.md)
```

---

## Dataset

HouseTS data (tabular signals) is available via Google Drive:

- Google Drive download: https://drive.google.com/file/d/1OC_PTXfaGuQ50-mu2LkfQRLdhjPUbyu7/view?usp=sharing

HouseTS aerial imagery data is hosted on Kaggle:

- Kaggle dataset page: https://www.kaggle.com/datasets/shengkunwang/housets-dataset

### Expected local path

By default, the benchmark expects:

- `data/raw/HouseTS.csv`

You can also point to `.csv`, `.parquet`, or `.xlsx` via config/CLI.

### Any dataset via a dataset config

The benchmark is dataset-agnostic: everything specific to one dataset (file path,
id/time/target columns, which columns to model, drop list, and the graph settings for GNN
models) lives in one YAML file under `configs/dataset/`. `configs/dataset/dc_house.yaml`
is the working example (matches `data/dc_house/DC_House.csv`); `configs/dataset/housets.yaml`
is a template for the full HouseTS.csv (not usable until you've downloaded it — see
"Dataset" above).

```yaml
dataset:
  name: dc_house
data:
  path: data/dc_house/DC_House.csv
  id_col: zipcode
  time_col: date
  target_col: price
  feature_cols: [median_sale_price, homes_sold, ...]  # null => auto-infer all numeric cols
  drop_cols: [city, metro, state, latitude, longitude]
  freq: M
graph:
  path: null            # graph.npz for GNN models — see "GNN models" section below
```

Minimal schema requirements for a new dataset CSV: an id column, a time column (parsed as
a timestamp), and the target column. If `feature_cols` is set, only those columns (plus
id/time/target) are read and modeled — everything else in the CSV is ignored. Missing
values are handled with a benchmark imputation routine; the loader adds `year` and `month`
time markers from the time column. The benchmark itself never reads lat/lon or builds a
graph — see "GNN models" below for how the graph is supplied.

Add a new dataset by copying `configs/dataset/dc_house.yaml` and pointing it at your CSV.

### Traffic datasets (METR-LA / PEMS-08)

Standard traffic benchmarks use different raw formats than the CSV panels above, so they
get their own loaders instead of a CSV config — set `data.loader` in the dataset config:

```yaml
data:
  loader: metr_la          # or pems08
  path: data/metr_la/metr-la.h5
  target_col: speed
  feature_cols: [time_of_day]   # DCRNN/Graph-WaveNet/MTGNN raw-channel convention
graph:
  path: data/metr_la/graph.npz   # build with scripts/build_traffic_graph.py
```

See `configs/dataset/metr_la.yaml` / `configs/dataset/pems08.yaml` for working examples
(seq_len=12/pred_len=12 at 5-min resolution, 70/10/20 chronological split, matching
DCRNN/Graph WaveNet/AGCRN's own convention). `st_numeric_baselines.data.io.load_metr_la`/
`load_pems08` build the standard `AlignedData` panel from each format directly; build the
matching `graph.npz` from a `from,to,cost` distances CSV with:

```bash
python scripts/build_traffic_graph.py \
  --distances-csv data/metr_la/distances_la_2012.csv \
  --sensor-ids data/metr_la/sensor_ids.txt \
  --out data/metr_la/graph.npz
```

(the Gaussian-kernel-threshold adjacency confirmed from DCRNN's own `gen_adj_mx.py`).
`PEMS-08`'s `.npz` has no embedded calendar timestamps — pass `data.start_time` (its
commonly-cited release date) to get a real day-of-week signal; without it, `time_of_day`
is still valid (a cyclic quantity) but day-of-week is zeroed out rather than fabricated.

**Time-of-day/day-of-week handling is genuinely split across model families** (confirmed
from each original repo's source) — the data pipeline supports both:
- `dcrnn`/`graph_wavenet`/`mtgnn`: want time-of-day as a plain extra **input channel**, no
  model-side change — just include `time_of_day` via `feature_cols` (both traffic loaders
  emit it automatically).
- `staeformer`/`stid`: want tod/dow via dedicated **embedding tables**
  (`nn.Embedding(steps_per_day, dim)` / `nn.Embedding(7, dim)`) — set `use_tod_dow: true`
  in the model config; they read it from `AlignedData.time_marks` (now
  `(tod_frac, dow)` for sub-daily data, auto-detected from the timestamp granularity —
  `(year, month)` unchanged for this benchmark's original monthly datasets), not from
  `x_cols`. Off by default.
- `agcrn`: the original has no time features at all in its traffic experiments; nothing
  to add here.

### Urban datasets (UrbanGPT benchmark: NYC-taxi / NYC-bike / NYC-crime / CHI-taxi)

From HKUDS/UrbanGPT (KDD'2024) — data on HuggingFace at
[`bjdwh/ST_data_urbangpt`](https://huggingface.co/datasets/bjdwh/ST_data_urbangpt).
**None of these four ship with a predefined graph** — confirmed directly from source:
UrbanGPT's own spatio-temporal encoder (`ST_Enc`) stores an `adj_mx` constructor argument
but never actually references it anywhere in its forward pass; the model is pure
dilated-convolution over the node axis, no graph convolution of any kind. So either use
one of this registry's `requires_graph=False` models (`staeformer`, `stid`, `agcrn`,
`mtgnn`, `testam`, `st_hhol`), or build your own graph.

```yaml
data:
  loader: nyc_taxi      # or chi_taxi, nyc_bike, nyc_crime
  path: data/nyc_taxi/all_nyc_taxi_263x105216x2.npz
  target_col: inflow
  feature_cols: [outflow]
  start_time: "2016-01-01"   # confirmed coverage start for the NYC datasets
graph:
  path: null
```

See `configs/dataset/{nyc_taxi,chi_taxi,nyc_bike,nyc_crime}.yaml` for working examples.
Confirmed from source (`instruction_generate/load_dataset.py`):

| loader | raw shape | regions | channels | sampling |
|---|---|---|---|---|
| `nyc_taxi`  | flat `[263, T, 2]`        | 263  | `inflow`, `outflow` | 30-min |
| `chi_taxi`  | flat `[77, T, 2]`         | 77   | `inflow`, `outflow` | hourly (inferred) |
| `nyc_bike`  | grid `[46, 47, T, 2]`     | 2162 | `inflow`, `outflow` | 30-min |
| `nyc_crime` | grid `[46, 47, T, 4]`     | 2162 | `burglaries`, `burglaries_aux`, `larcenies`, `larcenies_aux` | daily |

`nyc_bike`/`nyc_crime`'s grid shape is flattened row-major to a flat node axis with ids
`"{prefix}{i}_{j}"` recording the original grid position — which also means a
4-/8-connected **grid** adjacency is a natural, trivial graph to build yourself if you want
a `requires_graph=True` model on these two:

```bash
python scripts/build_grid_graph.py --ny 46 --nx 47 --id-prefix bike_ --out data/nyc_bike/graph.npz
```

`nyc_crime`'s daily sampling also exercises a third `time_marks` tier beyond the two
described above: `(dow, month)` for daily-cadence data (added alongside this feature,
since a plain `(year, month)` mark would collapse every day in a month together and lose
exactly the day-of-week signal that matters most for daily crime counts) — see
`st_numeric_baselines.data.io._build_time_marks`.


## Quick start

All examples below are run from the repository root.

### 1) Run a single experiment (config-driven)

The config runner merges, in order:
- `configs/default.yaml`
- `configs/dataset/<dataset>.yaml`
- `configs/task/<task>.yaml`
- `configs/models/<model>.yaml`

`run_one.py` itself never sets defaults for window/split values — it only overrides the
merged YAML config when a flag is explicitly passed. So the window shape (lookback/horizon)
and split (train/val ratio, test cutoff) are configured either in `configs/default.yaml`
(the global default) or per-dataset in `configs/dataset/<name>.yaml` (add a `window:`/`split:`
block there to override for just that dataset), and the CLI flags below are for one-off
overrides on top of whichever config is in effect.

Runs are written to `runs/<dataset>/<model>__<task>__<window>/`, where `<window>` is derived
from the effective `seq_len`/`pred_len` (e.g. `w12_h6`).

Example (dc_house, multivariate, model `dlinear`, using whatever window/split configs say):

```bash
python scripts/run_one.py \
  --dataset dc_house \
  --task multivariate \
  --model dlinear \
  --device gpu
```
### 2) Run a univariate baseline with a specific window shape

```bash
python scripts/run_one.py \
  --dataset dc_house \
  --task univariate \
  --seq-len 12 --label-len 6 --pred-len 6 \
  --model ar_univariate \
  --device cpu
```

### 3) Configure the split and cut evaluation cost on the test set

```bash
python scripts/run_one.py --dataset dc_house --model timesfm_zero \
  --seq-len 12 --pred-len 12 \
  --test-stride 3                    # only evaluate every 3rd test window
python scripts/run_one.py --dataset dc_house --model dlinear \
  --seq-len 6 --label-len 3 --pred-len 3 \
  --test-cutoff-date 2022-01-01      # test starts at this exact date instead of a ratio split
python scripts/run_one.py --dataset dc_house --model dlinear \
  --train-ratio 0.8 --val-ratio 0.1  # override the train/val split of the remaining (pre-test) span
```

---

## Window and split configuration

Window shape (`seq_len`/`label_len`/`pred_len`, i.e. lookback/decoder-label/horizon lengths)
and the train/val/test split live directly in config — there is no fixed set of window
presets to choose from. `configs/default.yaml` sets the global defaults:

```yaml
split:
  train_ratio: 0.7        # fraction of the pre-test span used for training
  val_ratio: 0.1          # fraction of the pre-test span used for validation
  test_start_date: null   # e.g. "2022-01-01" — overrides ratio-based test-set boundary

window:
  seq_len: 12             # lookback length
  label_len: 6            # decoder label length
  pred_len: 12            # forecast horizon
  test_stride: 1          # >1 strides through non-overlapping test windows to cut eval cost
```

Override either block for a specific dataset by adding a `window:`/`split:` block to its
`configs/dataset/<name>.yaml`, or override individual values from the CLI with
`--seq-len`/`--label-len`/`--pred-len`/`--test-stride`/`--train-ratio`/`--val-ratio`/
`--test-cutoff-date` (each only takes effect if explicitly passed), or with
`--set window.seq_len=12` / `--set split.train_ratio=0.8` for anything else.

---

## Supported Model Configs

The current `configs/models/` directory includes the following model configs.

### Statistical baselines

- `ar_univariate`
- `ardl`
- `arima`
- `var`
- `var_ms`

### Classical machine learning

- `rf`
- `xgb`

### Deep learning

- `rnn`
- `lstm`
- `dlinear`
- `timemixer`
- `patchtst`
- `informer`
- `autoformer`
- `fedformer`

### Graph neural networks

- `gcn_tcn`
- `graph_wavenet`
- `stgcn`
- `stsgcn`
- `stllm_plus`
- `dcrnn` — diffusion-convolutional seq2seq (Li et al., ICLR 2018), direct port
- `stgformer` — spatiotemporal graph transformer (Dreamzz5/STGformer), direct port
- `d2stgnn` — decoupled dynamic STGNN (VLDB 2022), direct port (dynamic graph is computed
  internally each forward pass; time-of-day/day-of-week features are omitted since this
  benchmark's datasets are monthly)
- `staeformer` — spatio-temporal adaptive embedding transformer (XDZhelheim/STAEformer,
  AAAI 2024), direct port — it doesn't use graph convolution or the adjacency at all,
  purely temporal + spatial self-attention over learned node/adaptive embeddings;
  time-of-day/day-of-week embeddings are **off by default** (monthly data has no
  sub-daily/weekly periodicity) but available via `use_tod_dow: true` for traffic data
  — see "Traffic datasets" below
- `cast` — causal spatio-temporal representation learning (yutong-xia/CaST), direct port
  (self-discovers pseudo-environments via a VQ codebook — no external environment labels needed)
- `stexplainer` — faithful port of the actually-shipped STGSAT model (HKUDS/STExplainer,
  source-verified): a GSAT (Graph Stochastic Attention) information-bottleneck pass over
  the real spatial adjacency and a second pass over a complete graph across the lookback
  timesteps, each producing a per-instance stochastic edge-attention gate regularized by
  `KL(Bernoulli(att) ‖ Bernoulli(r))` (r annealed 0.9→0.5 over training). Extended with a
  **third GSAT pass over the feature-channel axis** (not present in the original paper,
  which has no feature-level mechanism at all — this mirrors exactly how the paper already
  treats time as a complete-graph pass). Explanations are exposed per instance via
  `STExplainerForecaster.explain()` — see the explainable case library below.
- `aist` — **simplified** port of the attention-based interpretable crime model
  (YeasirRayhanPrince/aist): keeps only the graph-attention mechanism over crime counts,
  batched over all nodes jointly; drops the paper's required taxi/POI/street-crime data and
  its per-region training loop
- `st_hhol` — **simplified, static-graph** port inspired by ST-HHOL's hierarchical hypergraph
  idea: a trainable hypergraph convolution over the crime-count panel only, trained with the
  benchmark's normal offline time split; drops the paper's weather/POI/socioeconomic/311 data
  sources and its online/streaming training loop

**Models that never consult an adjacency at all** (`requires_graph = False` on the
forecaster class — `dataset.graph.path` doesn't need to be set; no `graph.npz`,
no coordinates, nothing): `staeformer`, `st_hhol` (above), plus three new additions —
- `stid` — Spatial-Temporal Identity (GestaltCogTeam/STID, CIKM'22), direct port: pure
  MLP over each node's flattened lookback window, concatenated with a learned per-node
  "spatial identity" embedding — no graph convolution or attention of any kind. The
  paper's own finding is that most of what STGNNs buy you comes from breaking
  spatial/temporal sample-indistinguishability, not the graph itself; day-of-week/
  time-of-day ("temporal identity") embeddings are **off by default** but available
  via `use_tod_dow: true` for traffic data — see "Traffic datasets" below
- `agcrn` — Adaptive Graph Convolutional Recurrent Network (LeiBAI/AGCRN, NeurIPS 2020),
  direct port of its core mechanism: learns its own adjacency purely from trainable node
  embeddings (`softmax(relu(E@E.T))`) and gives every node its own graph-conv weights via
  node-adaptive parameter learning, replacing a GRU's linear gates. The paper's
  scheduled-sampling decoder is replaced with a direct multi-horizon head (that mechanism
  is already covered by `dcrnn` in this registry)
- `mtgnn` — "Connecting the Dots" (nnzhan/MTGNN, KDD 2020), direct port of its core
  mechanism: learns a **directed**, top-k-sparsified adjacency from two node-embedding
  matrices, then alternates dilated-inception temporal convolution (parallel kernel
  sizes 2/3/6/7) with mix-hop graph propagation over the learned graph and its transpose
- `testam` — Time-Enhanced Spatio-temporal Attention Model with Mixture of experts
  (Lee & Ko, ICLR 2024): 3 experts (identity/no-graph, a learned-static-adjacency GCN, a
  fully dynamic-attention graph) gated by a shared memory bank with **hard top-1 routing**
  (same at train and eval), plus a warmup-then-routing-loss training schedule. Macro
  architecture confirmed from source; some tensor-level formulas are a documented
  reconstruction — see the module docstring for exactly which

### Ensembles / mixture-of-experts

- `ensemble_st` — a fixed, **equal-weight** ensemble over a configurable list of this
  registry's own models (`sub_models`, default `[gcn_tcn, stgformer, dcrnn]`): trains every
  selected sub-model itself, then averages their predictions with zero extra trainable
  parameters — the standard "zero-parameter ensemble" baseline this line of literature
  (e.g. GC-MoE below) compares learned routing against. Appears as one model, usable
  directly via `run_one.py --model ensemble_st`, no separate orchestration script needed.
- `gc_moe` — "Graph-Conditioned Mixture of Graph Neural Network Experts" (Ghaffari,
  Sheikhi & Gilman): trains a configurable list of experts (`expert_models`) fully, freezes
  every one of them, then trains only a small router that fuses 9 real graph-topology
  features (degree, closeness, clustering, PageRank, betweenness, k-core, eigenvector
  centrality, the Fiedler vector, the 3rd Laplacian eigenvector — via `networkx`) with a
  temporal-attention summary of the current input window into per-node **soft** mixture
  weights. Unlike `ensemble_st`/`testam`, this one genuinely needs a real adjacency
  (`graph.path` required) since the router is conditioned on it.

### Foundation-model variants

- `chronos2_zero`
- `chronos2_ft`
- `timesfm_xreg_zero`
- `timesfm_xreg_ft`

---

---

## GNN models: dataloader structure

GNN models (GCN-TCN, STGCN, GraphWaveNet, STSGCN, ST-LLM+) need a node adjacency graph.
The benchmark never builds this itself (no implicit lat/lon handling) — it only loads a
precomputed graph from a `.npz` file, pointed to by `graph.path` in the dataset config
(dataset-level, not per-model, since every GNN model on a dataset shares the same graph):

```python
np.savez("graph.npz", A=A, ids=np.array(ids))
```

- `A`: dense `[N, N]` adjacency matrix.
- `ids`: length-`N` array of region ids giving `A`'s row/column order (`ids[i]` is the
  region at row/col `i`) — these must match the dataset's id column values.

At load time (`st_numeric_baselines/graph/loader.py`), the graph is **reindexed by matching
`ids` against the dataset's actual region ids** — not by trusting row order — so it works
correctly regardless of what order the matrix was built in, and after any `--n-zip`
subsampling (only the subsampled regions are looked up; a region missing from `ids` raises
a clear error rather than silently misaligning).

```bash
python scripts/run_one.py --dataset dc_house --model gcn_tcn --seq-len 6 --label-len 3 --pred-len 6 \
  --set graph.path=data/dc_house_graph.npz
```

If you need a geographic k-NN graph from lat/lon, `scripts/build_knn_graph.py` is a
standalone offline builder (not imported by the benchmark) that produces this `graph.npz`
format from a lat/lon CSV:

```bash
python scripts/build_knn_graph.py --input data/DC_House.csv \
  --id-col zipcode --lat-col latitude --lon-col longitude \
  --k 10 --max-km 100 --out data/dc_house_graph.npz
```

### Why GNN dataloaders are different

| | DL models | GNN models |
|---|---|---|
| **Unit of one sample** | one ZIP × one time window | one time window × **all N ZIPs** |
| **Batch shape** | `[B, L, Dx]` — B mixes ZIPs and time positions | `[B, L, N, Dx]` — B is time positions only, N always equals total ZIPs |
| **Spatial coupling** | None — each ZIP is processed independently | Full — message-passing across N geographic neighbors per step |
| **Batch size meaning** | number of (ZIP, window) pairs | number of time windows (all N nodes included in each) |

**DL dataloader** (`WindowDataset`): generates one `(zip_i, t₀)` anchor per item.
The DataLoader collects B such anchors into a tensor `[B, L, Dx]`.
Spatial information across ZIPs is entirely absent; each row is independent.

**GNN dataloader** (`GraphWindowDataset`): generates one `t₀` anchor per item —
but returns the feature matrix for **all N nodes at that time step**.
The batch tensor `[B, L, N, Dx]` lets the network perform graph message-passing
across the N-dimension, so every ZIP can receive information from its
geographic neighbors.

After the GNN forward pass the output `[B, H, N, Dy]` is reshaped to
`[B×N, H, Dy]` so the standard `StreamingEvaluator` receives the same
`(n_samples, horizon, features)` format it expects from DL models.

### STSGCN

STSGCN (Wu et al., AAAI 2020, [code](https://github.com/Davidham3/STSGCN))
differs from STGCN in that it captures spatial and temporal dependencies
**synchronously** in a single graph operation rather than in separate sequential
stages.

It constructs a spatial-temporal synchronous adjacency
`A_st ∈ R^{T_local·N × T_local·N}` by stacking `T_local` copies of the spatial
graph on the diagonal and adding identity connections between consecutive steps:

```
A_st = [ A_s  I    0  ]
       [ I    A_s  I  ]   (T_local = 3)
       [ 0    I    A_s]
```

A GCN applied to the flattened `T_local·N`-node graph then aggregates across
both the spatial and temporal axes in one pass.  Each prediction step has its
own independent STSGCM branch, extracting the representation at the centre
time step.

---

## Alignment with BasicTS

A deeper follow-up check than the one originally done here. The first pass
only cross-referenced [BasicTS](https://github.com/GestaltCogTeam/BasicTS)'s
README "Spatial-Temporal Forecasting" table by model *name* and claimed "8
direct architectural matches" — that claim was **overstated** and has been
corrected below after actually fetching BasicTS's source.

**The corrected finding**: BasicTS's own vendored model zoo
(`src/basicts/models/`, 28 entries total, confirmed exhaustively) contains
exactly **one** model from that list of 8 — `STID`. The other seven
(`dcrnn`, `graph_wavenet`, `stgcn`, `agcrn`, `mtgnn`, `d2stgnn`,
`staeformer`) are **not implemented in BasicTS's codebase at all** — its
README table lists them with a venue + a link to the *original authors' own
repos* (e.g. `liyaguang/DCRNN`, `nnzhan/Graph-WaveNet`, `LeiBAI/AGCRN`,
`nnzhan/MTGNN`, `zezhishao/D2STGNN`, `XDZhelheim/STAEformer`) for
reproducibility, not as a BasicTS reimplementation. So "alignment with
BasicTS" was never the right frame for those seven — this registry's own
model docstrings already cite and port from those same original repos
directly, which is the correct and only real reference standard for them.

**For the one model BasicTS actually implements (`STID`), a real
line-by-line comparison was done** against `st_numeric_baselines/models/gnn/stid.py`:
- **Confirmed exact match**: BasicTS derives tod/dow indices from the
  lookback window's **last timestep only** (`inputs_timestamps[:, -1, 0/1]`)
  — exactly this registry's design (`stid.py`'s `x_mark[:, -1, :]`). This
  was a judgment call made independently on this side; BasicTS's real
  source confirms it was the right one.
- **Hyperparameter capacity differences, not bugs**: BasicTS's default
  config sets all embedding dims (spatial/tod/dow/input) to a uniform 32
  and `num_layers=1`; this registry's `configs/models/stid.yaml` uses
  `embed_dim=32` (matches) but `node_emb_dim`/`tod_embedding_dim`/
  `dow_embedding_dim=16` (half) and `n_layers=3`. Worth aligning if exact
  reproduction of BasicTS's own numbers is ever the goal; not a
  correctness issue either way.
- **Necessary generalization, not a divergence**: BasicTS's STID assumes a
  univariate per-node input (`Dx=1`); this registry's `history_encoder`
  flattens `seq_len * input_dim` together because every dataset here is
  genuinely multivariate (multiple feature columns per node). Required for
  this benchmark's data, not a deviation from a reference bug.

**Data/scaler pipeline**: BasicTS's z-score scaler and this registry's
`ZScoreTransform` use the identical formula (`(x-mean)/std`, zero-std
guarded to 1.0) fit on the train split only — confirmed equivalent. One
real terminology gap worth knowing: BasicTS's `norm_each_channel=True`
means per-node-per-feature (matches this registry's `scope: per_zip`
exactly), while `norm_each_channel=False` means **one single scalar for
the entire tensor** — this registry's own `scope: global` default is
neither of those; it's per-*feature*, shared across nodes (the right
choice here, since this benchmark's columns are heterogeneous — e.g. price
vs. homes_sold — and a single scalar across all of them would be wrong,
but it has no BasicTS equivalent, so don't assume the two "global"s mean
the same thing).

**This registry's domain-specific additions outside BasicTS's scope
entirely** (not a gap — BasicTS doesn't cover these problem settings at
all): `aist`/`st_hhol` (crime-specific simplified ports), `cast`/
`stexplainer` (causal/explainability research), `testam`/`gc_moe`/
`ensemble_st` (mixture-of-experts / model-selection research — this repo's
own distinctive angle).

---

## Case library: per-instance best model + neighbor explainability

Once you have trained checkpoints under `runs/`, `scripts/build_case_library.py`
scores every already-trained model on **every** (region, lookback/forecast
window) instance across the full train+val+test span — not just the
aggregate metrics in `metrics.json` — and records which model wins each
instance, in raw target units:

```bash
python scripts/build_case_library.py \
  --runs-root runs/dc_house \
  --models dlinear patchtst gcn_tcn stgformer \
  --out-dir runs/dc_house/case_library
```

Two CSVs are written under `--out-dir`:

1. **`case_library_detail.csv`** — one row per (model, instance):
   `model, model_category, split, region_id, lookback_start, forecast_start,
   forecast_end, mse, mae, rmse, y_true_raw, y_pred_raw`. `mse`/`mae`/`rmse`
   are over the whole forecast window (raw target units); `y_true_raw` and
   `y_pred_raw` hold that window's actual true and forecasted values, one
   value per horizon step, pipe-joined into a single cell (e.g.
   `412000.0000|418500.0000|421000.0000`). Computed first, directly from each
   model's predictions.
2. **`case_library.csv`** — the summary: one row per instance, the winning
   model only (lowest mae) — `region_id, lookback_start, forecast_start,
   forecast_end, model_best, model_best_mae, model_best_rmse,
   model_best_category` (category is `spatial_temporal` / `DL` / `foundation`,
   or `other` for statistical/ML baselines) — derived from the detail table.

It reuses each run's own saved checkpoint (no retraining — except models with
no train-dependent state, like `timesfm_zero`/`chronos2_zero`, which run
directly since there's nothing a missing checkpoint would lose) and evaluates
with `window.test_stride` forced to `1` regardless of what the run was
trained with, since the stride exists only to cut normal benchmark evaluation
cost, not for this exhaustive per-instance comparison — so this can be
considerably slower than a normal `run_one.py` evaluation. All runs passed
together must share the same dataset, window shape, and split boundaries
(validated up front, with a clear error listing any mismatch).

### Oracle / ensemble upper-bound analysis

Once you have `case_library`-comparable checkpoints for several models,
`scripts/build_oracle_report.py` answers the question that actually motivates
per-instance model selection (e.g. by an LLM): **is there upside to selecting
at all, and how much?** Compared on the held-out **test set only** by default
(`--splits val test` to widen it), using each run's own configured
`test_stride` (`--test-stride 1` overrides it, matching what
`build_case_library.py` forces for its own exhaustive per-instance
purpose, at the cost of roughly multiplying the instance count by the
original stride) — "oracle accuracy" should answer "how much upside is
there on unseen data", not on an artificially densified re-sampling of it.

```bash
python scripts/build_oracle_report.py \
  --runs-root runs/dc_house \
  --models dlinear patchtst gcn_tcn stgformer stid agcrn mtgnn \
  --ensemble-method mean \
  --out-dir runs/dc_house/oracle_report
```

It scores every model on every shared instance (reusing
`st_numeric_baselines.case_library.build_case_library`), then for each instance
computes:

- **Oracle** — the error of whichever model scored lowest MAE *on that
  instance*. Averaged over instances, this is a strict upper bound: it always
  beats the best single model in aggregate by construction (an average of
  pointwise minimums can't exceed any fixed model's average) — the real
  question is the **size of the gap**, i.e. how much accuracy a perfect
  per-instance selector could ever recover over just deploying the single
  best model.
- **Ensemble** — simple mean (or `--ensemble-method median`) of every model's
  raw prediction for that instance. Unlike the oracle, this is *not*
  guaranteed to beat the best single model — correlated biases across models
  can make the average worse — so it's measured, not assumed.

Three files land under `--out-dir`:

1. **`oracle_instance_detail.csv`** — one row per instance: the oracle's
   chosen model + its mse/mae/rmse, the ensemble's mse/mae/rmse, and every
   individual model's own MAE for that instance (wide format — this is the
   table to hand an LLM/classifier to learn *when* each model wins, i.e. to
   build the actual selector once the oracle analysis says selection is
   worthwhile).
2. **`oracle_summary.csv`** — one row per method (each model, `oracle`,
   `ensemble_<method>`): aggregate `mse`/`mae`/`rmse` over the instance set
   every model scored in common, sorted best-to-worst.
3. **`oracle_win_counts.csv`** — how often the oracle picks each model
   (`model, n_wins, win_pct`). A lopsided distribution (one model wins almost
   everywhere) means selection has little to add; an even spread across
   several models is the signal that it does.

The console output prints a direct verdict: the best single model's MAE, the
oracle's MAE and its % improvement over that best single model (the
selection gap), and whether the ensemble beats the best single model / the
oracle. Only instances scored by every model are compared — an instance one
model's own windowing dropped (e.g. a NaN somewhere in its lookback/forecast
slice for a DL/foundation model — see `explainable_library.py`'s
`_nan_dropped_window_counts`) is excluded and counted in a printed skip line,
since a "winner" can't be fairly attributed without every model's score for
that instance.

`scripts/explain_instance.py` gives an on-demand, model-agnostic breakdown of
one instance's forecast via **occlusion** (`st_numeric_baselines.explain.occlusion_sensitivity`):
replace one object — a neighbor node's whole feature vector, or one feature
channel — with its **mean over the training history** (an actual
in-distribution value, computed per-node/per-feature from the model's own
processed feature space, not an arbitrary placeholder like zero), rerun the
forward pass, and measure `sensitivity = mean(|y_full - y_occluded|)` over
the horizon, in raw target units (`y_full` is the model's own unperturbed
forecast, not the ground truth):

```bash
python scripts/explain_instance.py \
  --run-dir runs/dc_house/stgformer__multivariate__w6_h3 \
  --region 20001 --lookback-start 2021-06-01 --device cpu
```

- **Feature contribution** (any model — GNN, DL, foundation, statistical/ML):
  occludes one input variable at a time (own price history, homes_sold,
  inventory, ...) for the target region and reports each one's `sensitivity`
  and normalized `contribution_pct` share of the total forecast change.
- **Neighbor contribution** (spatiotemporal/GNN models only, skipped
  otherwise): occludes one neighbor region at a time (all of its features,
  replaced with that region's own training means) and reports the same,
  including a `self` row for how much the forecast relies on the target
  region's own history vs. its neighbors.

Both work identically across every model in the registry — not just ones
with built-in attention — since they only call the public
`model.predict_batch(...)` API.

---

## Explainable case library: 3 model families, node/feature/time explanation

`scripts/build_explainable_case_library.py` is a different, more specialized tool
from `build_case_library.py` above: instead of ranking many models against each
other, it always uses exactly **one** model per family — a univariate model, a
multivariate model, and STExplainer — reusing their already-trained checkpoints:

```bash
python scripts/build_explainable_case_library.py \
  --runs-root runs/dc_house \
  --univariate-model timesfm_zero \
  --multivariate-model chronos2_zero \
  --stexplainer-model stexplainer \
  --explain-n-instances 20 \
  --out-dir runs/dc_house/explainable_case_library
```

Writes three CSVs:
- **`forecasts_detail.csv`** — point forecasts from all 3 runs, every instance
  (same shape as `build_case_library.py`'s detail table).
- **`feature_importance.csv`** — grouped **exact Shapley** feature importance for
  the multivariate model (`st_numeric_baselines.shap_occlusion`, implementing
  arXiv:2604.28149's method: exact Shapley value over `2^N` coalitions of
  feature groups — one group per covariate by default, `2^N` model evaluations
  per instance). Masking is native (missing values, which Chronos's own
  tokenizer treats as such) for Chronos models, and training-history-mean
  substitution for any fixed-shape model (iTransformer, etc.) — matching the
  paper's own model-specific distinction. Computed for a sampled subset of test
  instances (`--explain-n-instances`), since it's `2^N` evaluations each.
- **`stexplainer_explanation.csv`** — STExplainer's own node/time/feature
  explanation for every instance (cheap: one extra forward pass per time
  window, not `2^N`). `node_importance` is genuinely target-specific (GAT
  attention rows are per-target by construction); `time_importance`/
  `feature_importance` are window-level, shared across every region's forecast
  in that time window — an architectural fact of the real GSAT mechanism
  (those branches operate on representations already aggregated across all
  regions/timesteps before their own GSAT pass runs), not a simplification.

## Chronos-2 reconciliation

A source-verified check of `_Chronos2Base` against the real `Chronos2Pipeline.predict_df`
API confirmed its covariate handling is already correct — passing every input
feature as a flat DataFrame column is the documented past-only-covariate
mechanism, and Chronos-2 does use them. The one divergence found: `Dy==1` is a
self-imposed restriction in this wrapper, not a real API limit (`predict_df`
supports genuine joint multi-target forecasting via `target=[...]`) — left
as-is since lifting it would need broader single-target-assumption changes
throughout the evaluator/case-library/explain code, and this benchmark's
explainability tools target single-target covariate importance, not
multi-target forecasting.

---

## Data Usage and Attribution

HouseTS integrates or aligns signals derived from several public data sources, including:

- housing-market time series
- OpenStreetMap-derived POI statistics
- U.S. Census / ACS socioeconomic variables
- USDA NAIP aerial imagery

Please review the paper and the upstream data-source licensing / attribution requirements before redistribution, publication of derivatives, or commercial use.

---

## Citation

If you use HouseTS or this benchmark code in your research, please cite:

```bibtex
@article{wang2025housets,
  title={HouseTS: A Large-Scale, Multimodal Spatiotemporal U.S. Housing Dataset and Benchmark},
  author={Wang, Shengkun and Sun, Yanshen and Chen, Fanglan and Wang, Linhan and Ramakrishnan, Naren and Lu, Chang-Tien and Chen, Yinlin},
  journal={arXiv preprint arXiv:2506.00765},
  year={2025}
}
```

---
