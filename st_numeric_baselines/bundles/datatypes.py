from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader

from st_numeric_baselines.data.io import AlignedData
from st_numeric_baselines.data.split import TimeSplit
from st_numeric_baselines.data.windowing import WindowSpec
from st_numeric_baselines.data.dataset import WindowDataset
from st_numeric_baselines.graph.loader import GraphConfig
from st_numeric_baselines.transforms.pipeline import TransformPipeline


@dataclass(frozen=True)
class RawBundle:
    aligned: AlignedData
    split: TimeSplit
    spec: WindowSpec
    features_mode: str
    graph: GraphConfig = field(default_factory=GraphConfig)


@dataclass(frozen=True)
class ProcBundle:
    raw: RawBundle
    pipeline: TransformPipeline
    aligned_proc: AlignedData

    x_cols: Tuple[str, ...]
    y_cols: Tuple[str, ...]

    datasets: Dict[str, WindowDataset]
    dataloaders: Dict[str, DataLoader]
    raw_target_col: str
    raw_target_index: int
