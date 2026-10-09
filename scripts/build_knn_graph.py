"""Standalone builder: geographic k-NN adjacency graph -> graph.npz.

Not imported by the benchmark itself — the benchmark only ever loads a
finished graph.npz (see st_numeric_baselines/graph/loader.py). This script is a
one-off/offline tool: run it once against a CSV of (id, lat, lon) to produce
the graph.npz that a GNN model's dataset config then points `graph.path` at.

Usage
-----
    python scripts/build_knn_graph.py \
      --input data/DC_House.csv --id-col zipcode --lat-col latitude --lon-col longitude \
      --k 10 --max-km 100 \
      --out data/dc_house_graph.npz

Two interchangeable backends compute the same thing (great-circle k-NN over
lat/lon, degrees, symmetrized, with an optional distance cutoff):
  - "libpysal" (default): libpysal.weights.KNN with distance_metric="Arc",
    i.e. libpysal's own geographic k-NN search.
  - "manual": a hand-rolled haversine k-NN (sklearn.neighbors.NearestNeighbors
    if available, else a pure-numpy fallback) — kept as a dependency-free
    reference implementation / fallback when libpysal isn't installed.
Select with --backend.

Output: np.savez(out, A=A, ids=ids) where A is a dense [N,N] adjacency matrix
and ids[i] is the region id for row/col i (order is whatever appears in the
input file, deduplicated).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    NearestNeighbors = None

try:
    from libpysal.weights import KNN as PysalKNN
    from libpysal.cg import RADIUS_EARTH_KM as PYSAL_RADIUS_EARTH_KM
except Exception:  # pragma: no cover
    PysalKNN = None
    PYSAL_RADIUS_EARTH_KM = None

EARTH_RADIUS_KM = 6371.0088


def build_knn_adjacency_libpysal(
    ids: list,
    coords: np.ndarray,  # [N, 2] (lat, lon), degrees
    *,
    k: int = 10,
    max_km: Optional[float] = 100.0,
    include_self_loops: bool = True,
    symmetric: bool = True,
) -> np.ndarray:
    """k-NN adjacency via libpysal.weights.KNN (great-circle "Arc" distance)."""
    if PysalKNN is None:
        raise ImportError("libpysal is not installed. Install it with: pip install libpysal")

    n = len(ids)
    A = np.zeros((n, n), dtype=np.float32)

    if include_self_loops:
        np.fill_diagonal(A, 1.0)

    if n > 1 and k > 0:
        kk = min(k, n - 1)
        # libpysal's Arc_KDTree assumes points are (lng, lat) — the opposite of
        # this script's (lat, lon) convention (and of sklearn's haversine
        # metric) — so swap columns just for this call.
        coords_lnglat = coords[:, ::-1]
        # Building the KNN object gets us a KDTree built with libpysal's own
        # great-circle ("Arc") distance; re-querying it (k+1, to include self
        # at col 0) gives per-neighbor distances in km for the max_km filter,
        # which the plain KNN weights object doesn't expose on its own.
        w = PysalKNN(coords_lnglat, k=kk, radius=PYSAL_RADIUS_EARTH_KM, distance_metric="Arc")
        dist_km, nbrs = w.kdtree.query(coords_lnglat, k=kk + 1)
        for row in range(nbrs.shape[0]):
            for col in range(1, nbrs.shape[1]):  # skip self (col 0)
                j = int(nbrs[row, col])
                km = float(dist_km[row, col])
                if max_km is not None and km > max_km:
                    continue
                A[row, j] = 1.0

    if symmetric:
        A = np.maximum(A, A.T)

    return A


def build_knn_adjacency_manual(
    ids: list,
    coords: np.ndarray,  # [N, 2] (lat, lon), degrees
    *,
    k: int = 10,
    max_km: Optional[float] = 100.0,
    include_self_loops: bool = True,
    symmetric: bool = True,
) -> np.ndarray:
    """Reference implementation: hand-rolled haversine k-NN, no libpysal dependency."""
    n = len(ids)
    A = np.zeros((n, n), dtype=np.float32)

    if include_self_loops:
        np.fill_diagonal(A, 1.0)

    if n > 1 and k > 0:
        rad = np.deg2rad(coords)
        if NearestNeighbors is not None:
            nn = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="ball_tree", metric="haversine")
            nn.fit(rad)
            dist_rad, nbrs = nn.kneighbors(rad, return_distance=True)
            dist_km = dist_rad * EARTH_RADIUS_KM
            for row in range(nbrs.shape[0]):
                for col in range(1, nbrs.shape[1]):  # skip self (col 0)
                    j = int(nbrs[row, col])
                    km = float(dist_km[row, col])
                    if max_km is not None and km > max_km:
                        continue
                    A[row, j] = 1.0
        else:
            for i in range(n):
                lat1, lon1 = rad[i]
                dlat = rad[:, 0] - lat1
                dlon = rad[:, 1] - lon1
                a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(rad[:, 0]) * np.sin(dlon / 2.0) ** 2
                km = EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))
                km[i] = np.inf
                nn_idx = np.argpartition(km, kth=min(k, n - 1) - 1)[: min(k, n - 1)]
                for j in nn_idx:
                    if max_km is None or km[j] <= max_km:
                        A[i, j] = 1.0

    if symmetric:
        A = np.maximum(A, A.T)

    return A


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=str, required=True, help="CSV containing id/lat/lon columns")
    p.add_argument("--id-col", type=str, default="zipcode")
    p.add_argument("--lat-col", type=str, default="latitude")
    p.add_argument("--lon-col", type=str, default="longitude")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--max-km", type=float, default=100.0, help="<=0 disables the distance cutoff")
    p.add_argument(
        "--backend",
        type=str,
        default="libpysal",
        choices=["libpysal", "manual"],
        help="k-NN implementation: libpysal.weights.KNN (default), or the dependency-free "
        "hand-rolled haversine reference implementation",
    )
    p.add_argument("--out", type=str, required=True, help="output .npz path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input, usecols=[args.id_col, args.lat_col, args.lon_col])
    df[args.id_col] = df[args.id_col].astype(str).str.strip()
    df = df.dropna(subset=[args.lat_col, args.lon_col]).drop_duplicates(subset=[args.id_col])
    df = df.sort_values(args.id_col).reset_index(drop=True)

    ids = df[args.id_col].tolist()
    coords = df[[args.lat_col, args.lon_col]].to_numpy(dtype=np.float64)
    max_km = None if args.max_km is not None and args.max_km <= 0 else args.max_km

    backend = args.backend
    if backend == "libpysal" and PysalKNN is None:
        print("[warn] libpysal not installed; falling back to --backend manual")
        backend = "manual"

    build_fn = build_knn_adjacency_libpysal if backend == "libpysal" else build_knn_adjacency_manual
    A = build_fn(ids, coords, k=args.k, max_km=max_km)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, A=A, ids=np.array(ids))
    print(f"Wrote graph with {len(ids)} nodes, {int((A != 0).sum())} directed edges -> {out_path}")


if __name__ == "__main__":
    main()
