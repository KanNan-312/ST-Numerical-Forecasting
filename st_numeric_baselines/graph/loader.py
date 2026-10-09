"""Load a precomputed node adjacency graph for GNN models.

The benchmark no longer builds graphs itself (e.g. from lat/lon) — it only
loads one from a ``.npz`` file with two arrays:

    np.savez("graph.npz", A=A, ids=np.array(ids))

    A:   [N, N] adjacency matrix (dense).
    ids: length-N array of region ids, giving the row/column order of ``A``
         (``ids[i]`` is the region at row/col ``i``). These must match the
         dataset's id column values (after the same normalization applied to
         the id column when loading the dataset).

See ``scripts/build_knn_graph.py`` for a standalone (not benchmark-imported)
script that builds such a file from a lat/lon CSV, if you need one.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from st_numeric_baselines.data.schema import normalize_zipcode


@dataclass(frozen=True)
class GeoGraph:
    edge_index: np.ndarray
    edge_weight: np.ndarray


_NULLISH_PATH_STRINGS = {"", "none", "null", "nul", "~", "nan"}


@dataclass(frozen=True)
class GraphConfig:
    """Dataset-level graph settings (shared by every GNN model on that dataset)."""

    path: Optional[str] = None  # path to a graph.npz file (A + ids arrays)

    def __post_init__(self) -> None:
        # YAML's null is `null`/`~`/empty, not the Python-looking bareword
        # `None` -- unquoted OR quoted `None` in a dataset YAML parses as the
        # literal string "None", not a null path. By the time this runs, that
        # string has typically already been through `resolve_relpaths` (which
        # joins any non-absolute string onto the repo root), turning it into
        # e.g. "<repo_root>/None" -- so check the path's *basename*, not the
        # raw value, to catch it either before or after that join. Without
        # this, a bogus non-empty path would look "set" and break every model
        # (including the requires_graph=False ones, which should just skip
        # graph loading entirely).
        if isinstance(self.path, str) and Path(self.path).name.strip().lower() in _NULLISH_PATH_STRINGS:
            object.__setattr__(self, "path", None)


def load_graph(path: str | Path, region_ids: Iterable[str]) -> GeoGraph:
    """Load ``path`` (a graph.npz with 'A' and 'ids') and align it to ``region_ids``."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"graph file not found: {p}")
    if p.suffix.lower() != ".npz":
        raise ValueError(
            f"graph file must be .npz containing 'A' and 'ids' arrays "
            f"(np.savez(path, A=A, ids=ids)), got {p.suffix}"
        )

    data = np.load(p, allow_pickle=False)
    if "A" not in data or "ids" not in data:
        raise ValueError(f"{p} must contain both 'A' and 'ids' arrays (np.savez(path, A=A, ids=ids))")

    A = np.asarray(data["A"])
    npz_ids = [normalize_zipcode(x) for x in data["ids"]]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"'A' in {p} must be a square [N,N] matrix, got shape {A.shape}")
    if A.shape[0] != len(npz_ids):
        raise ValueError(
            f"'A' in {p} has shape {A.shape} but 'ids' has {len(npz_ids)} entries; they must match"
        )

    ids = [normalize_zipcode(z) for z in region_ids]
    id_to_pos = {rid: i for i, rid in enumerate(npz_ids)}
    missing = [r for r in ids if r not in id_to_pos]
    if missing:
        raise ValueError(
            f"{len(missing)} region id(s) used by this run are not present in {p}'s 'ids' array "
            f"(e.g. {missing[:5]}). The graph file must cover every region used by this run "
            f"(check --n-zip subsampling and id normalization)."
        )

    perm = [id_to_pos[r] for r in ids]
    A_aligned = A[np.ix_(perm, perm)]

    src, dst = np.nonzero(A_aligned)
    edge_weight = A_aligned[src, dst].astype(np.float32)
    edge_index = np.stack([src.astype(np.int64), dst.astype(np.int64)], axis=0)

    return GeoGraph(edge_index=edge_index, edge_weight=edge_weight)
