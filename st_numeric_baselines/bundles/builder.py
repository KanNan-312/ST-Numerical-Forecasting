from __future__ import annotations

from typing import Dict, Optional, Tuple

from st_numeric_baselines.data.dataset import WindowDataset
from st_numeric_baselines.data.dataloader import make_dataloader
from st_numeric_baselines.data.io import AlignedData, AlignedData as _AlignedData
from st_numeric_baselines.data.schema import FeatureSchema
from st_numeric_baselines.data.split import TimeSplit
from st_numeric_baselines.data.windowing import WindowSpec, generate_window_indices
from st_numeric_baselines.bundles.datatypes import ProcBundle, RawBundle
from st_numeric_baselines.graph.loader import GraphConfig
from st_numeric_baselines.transforms.pipeline import TransformPipeline


def _make_proc_schema(*, base: FeatureSchema, out_dim: int) -> FeatureSchema:
    if out_dim == len(base.continuous_cols):
        return base

    pcs = tuple(f"pc{i}" for i in range(out_dim))
    return FeatureSchema(
        id_col=base.id_col,
        time_col=base.time_col,
        target_col=pcs[0],  # placeholder; in PCA space there is no single 'price' column
        drop_cols=base.drop_cols,
        time_mark_cols=base.time_mark_cols,
        continuous_cols=pcs,
    )


def build_proc_bundle(
    aligned: AlignedData,
    *,
    split: TimeSplit,
    spec: WindowSpec,
    features_mode: str,
    pipeline: TransformPipeline,
    batch_size: int = 32,
    num_workers: int = 0,
    pad_to: Optional[int] = None,
    shuffle_train: bool = True,
    graph_cfg: Optional[GraphConfig] = None,
) -> ProcBundle:

    # 1) Fit+transform full tensor
    values_proc = pipeline.fit_transform(aligned.values, train_range=split.train)

    proc_schema = _make_proc_schema(base=aligned.schema, out_dim=values_proc.shape[-1])

    aligned_proc = _AlignedData(
        zipcodes=aligned.zipcodes,
        dates=aligned.dates,
        values=values_proc,
        time_marks=aligned.time_marks,
        schema=proc_schema,
    )

    # 2) Decide x/y columns
    if proc_schema is aligned.schema:
        x_cols, y_cols = proc_schema.select_xy_cols(features_mode)
    else:
        # PCA space: statistical/ML baselines forecast all components
        x_cols, y_cols = proc_schema.continuous_cols, proc_schema.continuous_cols

    # 3) Generate indices per split
    name_to_idx = {n: i for i, n in enumerate(proc_schema.continuous_cols)}
    x_idx = [name_to_idx[c] for c in x_cols]
    y_idx = [name_to_idx[c] for c in y_cols]

    indices: Dict[str, list[tuple[int, int]]] = {}
    for split_name in ("train", "val", "test"):
        split_range = split.range(split_name)
        allow_history = split_name != "train"
        stride = spec.test_stride if split_name == "test" else 1
        indices[split_name] = generate_window_indices(
            values=aligned_proc.values,
            split_range=split_range,
            split_start_for_targets=split_range[0],
            x_idx=x_idx,
            y_idx=y_idx,
            spec=spec,
            allow_history=allow_history,
            require_finite=True,
            stride=stride,
        )

    # 4) Datasets + dataloaders
    datasets: Dict[str, WindowDataset] = {}
    dataloaders: Dict[str, object] = {}

    for split_name in ("train", "val", "test"):
        ds = WindowDataset(
            aligned_proc,
            x_cols=x_cols,
            y_cols=y_cols,
            indices=indices[split_name],
            spec=spec,
        )
        datasets[split_name] = ds

        dl = make_dataloader(
            ds,
            batch_size=batch_size,
            shuffle=(shuffle_train if split_name == "train" else False),
            num_workers=num_workers,
            pad_to=pad_to,
        )
        dataloaders[split_name] = dl

    raw_target = aligned.schema.target_col
    raw_target_index = list(aligned.schema.continuous_cols).index(raw_target)

    raw = RawBundle(
        aligned=aligned,
        split=split,
        spec=spec,
        features_mode=features_mode,
        graph=graph_cfg if graph_cfg is not None else GraphConfig(),
    )

    return ProcBundle(
        raw=raw,
        pipeline=pipeline,
        aligned_proc=aligned_proc,
        x_cols=tuple(x_cols),
        y_cols=tuple(y_cols),
        datasets=datasets,
        dataloaders=dataloaders, 
        raw_target_col=raw_target,
        raw_target_index=raw_target_index,
    )
