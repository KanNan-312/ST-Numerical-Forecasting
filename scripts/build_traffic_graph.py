"""Standalone builder: distance-based sensor adjacency (METR-LA/PEMS-style) -> graph.npz.

Not imported by the benchmark itself — same convention as
scripts/build_knn_graph.py: a one-off/offline tool producing the graph.npz
a GNN model's dataset config then points `graph.path` at.

Implements the Gaussian-kernel-threshold adjacency confirmed from DCRNN's own
`gen_adj_mx.py` (liyaguang/DCRNN), reused as-is by Graph WaveNet:

    A[i,j] = exp(-(dist[i,j] / std)**2)   if A[i,j] >= normalized_k else 0

where ``std`` is the standard deviation of every finite pairwise distance in
the input file, and ``normalized_k`` (default 0.1) is the sparsification
threshold from the original paper's default config.

Usage
-----
    python scripts/build_traffic_graph.py \
      --distances-csv data/metr_la/distances_la_2012.csv \
      --sensor-ids data/metr_la/sensor_ids.txt \
      --normalized-k 0.1 \
      --out data/metr_la/graph.npz

``--distances-csv`` must have ``from,to,cost`` columns (METR-LA's own format
— ``from``/``to`` are sensor ids, ``cost`` is the pairwise distance in
meters). ``--sensor-ids`` is a text file with one comma-separated line of
every sensor id, in the row/col order you want in the output ``graph.npz``
(pass the dataset's own id ordering — e.g. the column order of the METR-LA
``.h5`` file — so ``graph.npz``'s ``ids`` lines up with the data loader's
``AlignedData.zipcodes``). If omitted, ids are inferred as the sorted union
of every id seen in the distances CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def build_gaussian_threshold_adjacency(
    ids: List[str],
    distances: pd.DataFrame,  # columns: from, to, cost
    *,
    normalized_k: float = 0.1,
) -> np.ndarray:
    """Gaussian-kernel-thresholded adjacency, confirmed from DCRNN's gen_adj_mx.py."""
    n = len(ids)
    id_to_idx = {str(i): idx for idx, i in enumerate(ids)}
    dist_mx = np.full((n, n), np.inf, dtype=np.float64)

    for _, row in distances.iterrows():
        src, dst = str(row["from"]), str(row["to"])
        if src not in id_to_idx or dst not in id_to_idx:
            continue
        dist_mx[id_to_idx[src], id_to_idx[dst]] = float(row["cost"])

    finite = dist_mx[np.isfinite(dist_mx)]
    std = float(finite.std()) if finite.size else 1.0
    if std <= 0:
        std = 1.0

    A = np.exp(-np.square(dist_mx / std))
    A[~np.isfinite(dist_mx)] = 0.0
    A[A < normalized_k] = 0.0
    return A.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distances-csv", type=str, required=True, help="CSV with from,to,cost columns")
    p.add_argument("--sensor-ids", type=str, default=None,
                   help="text file, one comma-separated line of sensor ids (row/col order); "
                   "omit to infer as the sorted union of ids seen in --distances-csv")
    p.add_argument("--normalized-k", type=float, default=0.1,
                   help="sparsification threshold (DCRNN's own default: 0.1)")
    p.add_argument("--out", type=str, required=True, help="output .npz path")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    distances = pd.read_csv(args.distances_csv, dtype={"from": str, "to": str})
    if not {"from", "to", "cost"}.issubset(distances.columns):
        raise ValueError(f"{args.distances_csv} must have from,to,cost columns, got {list(distances.columns)}")

    if args.sensor_ids:
        with open(args.sensor_ids) as f:
            ids = [s.strip() for s in f.read().strip().split(",") if s.strip()]
    else:
        ids = sorted(set(distances["from"]) | set(distances["to"]))

    A = build_gaussian_threshold_adjacency(ids, distances, normalized_k=args.normalized_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, A=A, ids=np.array(ids))
    print(f"Wrote graph with {len(ids)} nodes, {int((A != 0).sum())} directed edges -> {out_path}")


if __name__ == "__main__":
    main()
