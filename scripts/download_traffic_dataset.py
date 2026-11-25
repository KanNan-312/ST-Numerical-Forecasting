"""Download + lay out METR-LA or PEMS-BAY (DCRNN/Graph-WaveNet's traffic-
speed benchmarks) so they're immediately usable by
``configs/dataset/metr_la.yaml`` / ``configs/dataset/pems_bay.yaml``.

Source: the ``TorchSpatiotemporal/ProcessedDatasets`` Hugging Face dataset
(https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets),
pinned to tag ``v1.0.0`` — a stable, directly-downloadable HTTPS re-package
of the original DCRNN release (liyaguang/DCRNN; no Google Drive / manual
click-through needed, unlike most other mirrors of this data). Each zip
contains the ``.h5`` speed matrix, a ``from,to,cost`` sensor-distance CSV,
and (METR-LA only) a sensor-id ordering file.

This script:
  1. downloads and extracts the dataset's zip into ``data/<dataset>/``
  2. renames the ``.h5``/distances files to the names the dataset config
     expects (hyphenated, matching the original DCRNN release's own naming)
  3. writes ``sensor_ids.txt`` (comma-separated, in the exact column order
     of the ``.h5`` file) if the zip didn't already ship one — PEMS-BAY's
     doesn't, since its h5 column order already *is* the canonical sensor
     order; METR-LA's shipped ``sensor_ids_la.txt`` is also re-verified to
     match before being trusted, rather than assumed
  4. builds ``graph.npz`` from the distance CSV via
     ``scripts/build_traffic_graph.py``'s own Gaussian-kernel-threshold
     adjacency, using that verified sensor order for row/col alignment

Usage
-----
    python scripts/download_traffic_dataset.py --dataset metr_la
    python scripts/download_traffic_dataset.py --dataset pems_bay
    python scripts/download_traffic_dataset.py --dataset metr_la pems_bay --data-root data
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import h5py
import pandas as pd

from scripts.build_traffic_graph import build_gaussian_threshold_adjacency
import numpy as np

_HF_BASE = "https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/traffic"


@dataclass(frozen=True)
class TrafficDatasetSpec:
    url: str
    zip_h5_name: str          # filename inside the zip
    zip_distances_name: str   # filename inside the zip ("from,to,cost" columns)
    zip_sensor_ids_name: Optional[str]  # filename inside the zip, or None if not shipped
    out_h5_name: str          # filename to write under data/<dataset>/
    out_distances_name: str
    out_sensor_ids_name: str = "sensor_ids.txt"
    out_graph_name: str = "graph.npz"


SPECS = {
    "metr_la": TrafficDatasetSpec(
        url=f"{_HF_BASE}/metr_la.zip",
        zip_h5_name="metr_la.h5",
        zip_distances_name="distances_la.csv",
        zip_sensor_ids_name="sensor_ids_la.txt",
        out_h5_name="metr-la.h5",
        out_distances_name="distances_la.csv",
    ),
    "pems_bay": TrafficDatasetSpec(
        url=f"{_HF_BASE}/pems_bay.zip",
        zip_h5_name="pems_bay.h5",
        zip_distances_name="distances_bay.csv",
        zip_sensor_ids_name=None,  # not shipped -- derived from the h5's own column order below
        out_h5_name="pems-bay.h5",
        out_distances_name="distances_bay.csv",
    ),
}


def _download(url: str, dest: Path) -> None:
    print(f"Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "st-numeric-baselines-dataset-download/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)


def _h5_sensor_order(h5_path: Path) -> list[str]:
    """Column order of the pandas-fixed-format HDF5 store, read directly via
    h5py (no ``tables``/pytables dependency needed just to recover this).
    """
    with h5py.File(h5_path, "r") as f:
        return [c.decode("utf-8") for c in f["data/axis0"][:]]


def download_one(name: str, *, data_root: Path, force: bool) -> None:
    if name not in SPECS:
        raise SystemExit(f"Unknown dataset {name!r} (expected one of: {sorted(SPECS)})")
    spec = SPECS[name]
    out_dir = data_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_h5 = out_dir / spec.out_h5_name
    out_distances = out_dir / spec.out_distances_name
    out_sensor_ids = out_dir / spec.out_sensor_ids_name
    out_graph = out_dir / spec.out_graph_name

    if out_h5.exists() and out_graph.exists() and not force:
        print(f"[{name}] {out_h5} and {out_graph} already present -- skipping (pass --force to redo)")
        return

    with tempfile.TemporaryDirectory(prefix=f"st_numeric_{name}_") as tmp:
        tmp_dir = Path(tmp)
        zip_path = tmp_dir / f"{name}.zip"
        _download(spec.url, zip_path)

        print(f"[{name}] extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_dir)

        src_h5 = tmp_dir / spec.zip_h5_name
        src_distances = tmp_dir / spec.zip_distances_name
        if not src_h5.exists():
            raise SystemExit(f"[{name}] expected {spec.zip_h5_name} in the downloaded zip, not found")
        if not src_distances.exists():
            raise SystemExit(f"[{name}] expected {spec.zip_distances_name} in the downloaded zip, not found")

        shutil.copyfile(src_h5, out_h5)
        shutil.copyfile(src_distances, out_distances)
        print(f"[{name}] wrote {out_h5}, {out_distances}")

        # Sensor order: always re-derive from the h5 itself (the authoritative
        # source — it's what the data loader's AlignedData.zipcodes will be),
        # and cross-check it against the shipped sensor-ids file if there is
        # one, rather than trusting either blindly.
        h5_order = _h5_sensor_order(out_h5)
        if spec.zip_sensor_ids_name is not None:
            src_ids = tmp_dir / spec.zip_sensor_ids_name
            if src_ids.exists():
                shipped_order = [s.strip() for s in src_ids.read_text().strip().split(",") if s.strip()]
                if shipped_order != h5_order:
                    raise SystemExit(
                        f"[{name}] shipped {spec.zip_sensor_ids_name} does not match the h5's own column "
                        "order -- refusing to build a graph.npz that could silently misalign rows"
                    )
        out_sensor_ids.write_text(",".join(h5_order))
        print(f"[{name}] wrote {out_sensor_ids} ({len(h5_order)} sensors, verified against the h5 column order)")

        distances = pd.read_csv(out_distances, dtype={"from": str, "to": str})
        if not {"from", "to", "cost"}.issubset(distances.columns):
            raise SystemExit(
                f"[{name}] {out_distances} must have from,to,cost columns, got {list(distances.columns)}"
            )
        A = build_gaussian_threshold_adjacency(h5_order, distances, normalized_k=0.1)
        np.savez(out_graph, A=A, ids=np.array(h5_order))
        print(f"[{name}] wrote {out_graph} ({len(h5_order)} nodes, {int((A != 0).sum())} directed edges)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", default=list(SPECS), choices=list(SPECS),
                   help="which dataset(s) to download (default: both)")
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
