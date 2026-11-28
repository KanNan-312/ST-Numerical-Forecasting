"""Download + lay out PEMS03/04/07/08 (ASTGCN/STSGCN/AGCRN's traffic-flow
benchmarks) so they're immediately usable by ``configs/dataset/pems0{3,4,7,8}.yaml``.

Source: guoshnBJTU/ASTGNN's own ``data/`` folder on GitHub
(https://github.com/guoshnBJTU/ASTGNN/tree/main/data), fetched via
raw.githubusercontent.com -- a stable, directly-downloadable mirror of the
same files everyone in this line of work actually uses (no Google/Baidu
drive click-through). Confirmed byte-identical to this repo's own
pre-existing ``data/pems08/PEMS08.npz``/``PEMS08.csv`` by direct comparison.

Each dataset ships a ``.npz`` (key ``'data'``, shape ``[T, N, C]`` --
C=3 flow/occupancy/speed for PEMS04/08, C=1 flow-only for PEMS03/07,
confirmed by inspection) and a ``from,to,cost`` sensor-distance CSV (PEMS03's
is actually ``from,to,distance`` -- renamed here to ``cost`` for a uniform
contract). For PEMS04/07/08 the CSV's ``from``/``to`` are already the same
0..N-1 integer row/col positions used inside the ``.npz`` array, confirmed
by checking their id range against each dataset's node count. **PEMS03 is
the one exception**: its CSV's ``from``/``to`` are the *original* Caltrans
PeMS sensor ids (6-digit numbers), not row positions -- the accompanying
``PEMS03.txt`` (one id per line) gives the row order needed to translate
them, confirmed by cross-checking: line *i* of ``PEMS03.txt`` is the raw
sensor id whose readings live at ``data[:, i, :]``. This script resolves
that mapping internally and always writes ``graph.npz``'s ``ids`` as plain
``"0".."N-1"`` strings (same convention as the other three, and as
``st_numeric_baselines.data.io.load_pems08``'s synthesized
``AlignedData.zipcodes``) regardless of which id space the source CSV used.

Usage
-----
    python scripts/download_pems_dataset.py --dataset pems08
    python scripts/download_pems_dataset.py --dataset pems03 pems04 pems07 pems08 --data-root data
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.build_traffic_graph import build_gaussian_threshold_adjacency

_BASE = "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/main/data"


@dataclass(frozen=True)
class PemsSpec:
    upper_name: str          # "PEMS03" etc -- the source repo's naming
    expected_n: int          # node count, for a sanity check after download
    expected_c: int          # channel count, for a sanity check after download
    needs_id_remap: bool = False  # True only for PEMS03 (see module docstring)


SPECS = {
    "pems03": PemsSpec(upper_name="PEMS03", expected_n=358, expected_c=1, needs_id_remap=True),
    "pems04": PemsSpec(upper_name="PEMS04", expected_n=307, expected_c=3),
    "pems07": PemsSpec(upper_name="PEMS07", expected_n=883, expected_c=1),
    "pems08": PemsSpec(upper_name="PEMS08", expected_n=170, expected_c=3),
}


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "st-numeric-baselines-dataset-download/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as f:
        f.write(resp.read())


def download_one(name: str, *, data_root: Path, force: bool) -> None:
    if name not in SPECS:
        raise SystemExit(f"Unknown dataset {name!r} (expected one of: {sorted(SPECS)})")
    spec = SPECS[name]
    out_dir = data_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = out_dir / f"{spec.upper_name}.npz"
    out_csv = out_dir / f"{spec.upper_name}.csv"
    out_graph = out_dir / "graph.npz"

    if out_npz.exists() and out_graph.exists() and not force:
        print(f"[{name}] {out_npz} and {out_graph} already present -- skipping (pass --force to redo)")
        return

    _download(f"{_BASE}/{spec.upper_name}/{spec.upper_name}.npz", out_npz)
    _download(f"{_BASE}/{spec.upper_name}/{spec.upper_name}.csv", out_csv)

    data = np.load(out_npz)["data"]
    t, n, c = data.shape
    if n != spec.expected_n or c != spec.expected_c:
        raise SystemExit(
            f"[{name}] downloaded {out_npz} has shape {data.shape}, expected "
            f"N={spec.expected_n} C={spec.expected_c} -- source may have changed, refusing to build graph.npz"
        )
    print(f"[{name}] wrote {out_npz} (T={t}, N={n}, C={c}), {out_csv}")

    distances = pd.read_csv(out_csv, dtype={"from": str, "to": str})
    distances = distances.rename(columns={"distance": "cost"})  # PEMS03's CSV names it "distance"
    if not {"from", "to", "cost"}.issubset(distances.columns):
        raise SystemExit(f"[{name}] {out_csv} must have from,to,(cost|distance) columns, got {list(distances.columns)}")

    if spec.needs_id_remap:
        # PEMS03 only: from/to are raw PeMS sensor ids -- PEMS03.txt's line
        # order gives the npz row each raw id belongs to (see module
        # docstring). Match the CSV in that raw-id space, then relabel the
        # output to plain positional ids like every other PEMS0x dataset.
        out_id_map = out_dir / f"{spec.upper_name}.txt"
        _download(f"{_BASE}/{spec.upper_name}/{spec.upper_name}.txt", out_id_map)
        raw_ids_in_row_order = [line.strip() for line in out_id_map.read_text().splitlines() if line.strip()]
        if len(raw_ids_in_row_order) != n:
            raise SystemExit(
                f"[{name}] {out_id_map} has {len(raw_ids_in_row_order)} ids, expected {n} (one per npz row)"
            )
        A = build_gaussian_threshold_adjacency(raw_ids_in_row_order, distances, normalized_k=0.1)
        out_id_map.unlink()  # only needed transiently to build the adjacency above
    else:
        # PEMS04/07/08: from/to already ARE the 0..N-1 npz row positions --
        # pass that order explicitly (never let it be inferred as a sorted
        # set of strings, which would wrongly lexicographically order
        # "0","1","10","100",... instead of numerically).
        positional_ids = [str(i) for i in range(n)]
        A = build_gaussian_threshold_adjacency(positional_ids, distances, normalized_k=0.1)

    ids_out = np.array([str(i) for i in range(n)])
    np.savez(out_graph, A=A, ids=ids_out)
    print(f"[{name}] wrote {out_graph} ({n} nodes, {int((A != 0).sum())} directed edges)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", default=list(SPECS), choices=list(SPECS),
                   help="which dataset(s) to download (default: all four)")
    p.add_argument("--data-root", type=str, default=str(REPO_ROOT / "data"),
                   help="root data/ directory (default: repo's data/)")
    p.add_argument("--force", action="store_true", help="redownload/rebuild even if output files already exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    for name in args.dataset:
        download_one(name, data_root=data_root, force=args.force)
    print("Done.")


if __name__ == "__main__":
    main()
