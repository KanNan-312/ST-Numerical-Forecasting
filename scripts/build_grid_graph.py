"""Standalone builder: 4-/8-connected grid adjacency -> graph.npz.

For datasets that are a literal 2D grid of regions (e.g. UrbanGPT's
NYC-bike/NYC-crime, which this benchmark's ``load_nyc_bike``/
``load_nyc_crime`` flatten row-major to ids ``"{prefix}{i}_{j}"``) but ship
with no predefined graph at all (confirmed from source — see those loaders'
docstrings), the grid position itself is a natural, trivial-to-build
adjacency: each cell is connected to its immediate grid neighbors.

Not imported by the benchmark itself — same convention as
scripts/build_knn_graph.py and scripts/build_traffic_graph.py: a one-off
tool producing the graph.npz a GNN model's dataset config then points
`graph.path` at.

Usage
-----
    python scripts/build_grid_graph.py \
      --ny 46 --nx 47 --id-prefix bike_ --connectivity 4 \
      --out data/nyc_bike/graph.npz

``--id-prefix`` must match the loader's own prefix (``"bike_"`` for
``load_nyc_bike``, ``"crime_"`` for ``load_nyc_crime``) so ``graph.npz``'s
``ids`` line up exactly with ``AlignedData.zipcodes``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_grid_adjacency(ny: int, nx: int, *, connectivity: int = 4, include_self_loops: bool = True) -> np.ndarray:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    n = ny * nx
    A = np.zeros((n, n), dtype=np.float32)

    def idx(i: int, j: int) -> int:
        return i * nx + j

    if connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        offsets = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if not (di == 0 and dj == 0)]

    for i in range(ny):
        for j in range(nx):
            u = idx(i, j)
            if include_self_loops:
                A[u, u] = 1.0
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < ny and 0 <= nj < nx:
                    A[u, idx(ni, nj)] = 1.0
    return A


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ny", type=int, required=True, help="grid height (e.g. 46 for NYC-bike/-crime)")
    p.add_argument("--nx", type=int, required=True, help="grid width (e.g. 47 for NYC-bike/-crime)")
    p.add_argument("--id-prefix", type=str, required=True,
                   help="must match the dataset loader's id prefix, e.g. 'bike_' or 'crime_'")
    p.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    p.add_argument("--no-self-loops", action="store_true")
    p.add_argument("--out", type=str, required=True, help="output .npz path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    A = build_grid_adjacency(args.ny, args.nx, connectivity=args.connectivity, include_self_loops=not args.no_self_loops)
    ids = np.array([f"{args.id_prefix}{i}_{j}" for i in range(args.ny) for j in range(args.nx)])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, A=A, ids=ids)
    print(f"Wrote {args.connectivity}-connected grid graph with {len(ids)} nodes "
          f"({int((A != 0).sum())} directed edges) -> {out_path}")


if __name__ == "__main__":
    main()
