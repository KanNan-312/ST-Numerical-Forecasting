"""Shared "find the run directories a case-library/oracle/explain tool
should operate on" logic, used by scripts/build_case_library.py,
scripts/build_oracle_report.py, and scripts/build_explainable_case_library.py.

Previously each of those scripts carried its own copy of this logic (and had
drifted: one silently ignored its own ``--models`` flag). Centralizing it
here means a fix (or a new convenience like ``--dataset``/``--window``) lands
in all three at once instead of requiring three separate edits that are easy
to let drift out of sync again.

``scripts/run_one.py`` writes runs to ``<out_root>/<dataset>/<window>/<model>__<task>/``
(window nested as its own directory level precisely so that a dataset's run
folder is never an ambiguous mix of several window configs) -- the
``--dataset``/``--window`` convenience here composes that same path, so you
don't have to spell it out by hand.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from st_numeric_baselines.utils.config import load_yaml


def add_runs_root_args(p: argparse.ArgumentParser, *, default_out_root: str) -> None:
    """Add the ``--runs-root`` / ``--dataset`` / ``--window`` / ``--out-root``
    flags shared by the case-library / oracle-report / explainable-library
    scripts. Pass ``--runs-root`` directly for full control, or ``--dataset``
    (+ ``--window``, e.g. ``w12_h12``) to compose
    ``<out-root>/<dataset>/<window>`` instead.
    """
    p.add_argument("--runs-root", type=str, default=None,
                    help="directory containing one subdirectory per run (config.yaml + checkpoint.pt); "
                    "omit to compose it from --dataset/--window instead")
    p.add_argument("--dataset", type=str, default=None,
                    help="dataset name -- composes --out-root/<dataset>/<window> when --runs-root is omitted")
    p.add_argument("--window", type=str, default=None,
                    help="window label, e.g. w12_h12 (matches scripts/run_one.py's run-dir naming); "
                    "required alongside --dataset unless --runs-root is given directly")
    p.add_argument("--out-root", type=str, default=default_out_root,
                    help="root runs/ directory, used only to compose --runs-root from --dataset/--window")


def resolve_runs_root(args: argparse.Namespace) -> Path:
    """Resolve the directory holding one dataset+window's run folders from
    whichever of ``--runs-root`` / ``--dataset``+``--window`` was given.
    """
    if args.runs_root is not None:
        return Path(args.runs_root).resolve()
    if args.dataset is None:
        raise SystemExit("pass either --runs-root, or --dataset (with --window)")
    if args.window is None:
        raise SystemExit(
            "--dataset was given without --window -- pass --window (e.g. w12_h12) to pick "
            "one window config, or pass --runs-root directly if you really want every window "
            "under the dataset folder treated as one pool of runs"
        )
    return (Path(args.out_root) / args.dataset / args.window).resolve()


def discover_run_dirs(runs_root: Path, models: Optional[List[str]] = None) -> List[Path]:
    """Direct children of ``runs_root`` that look like a run (have a
    ``config.yaml`` -- ``checkpoint.pt`` may legitimately be missing for
    checkpoint_optional models, e.g. ``timesfm_zero``/``chronos2_zero``,
    which ``load_run()`` handles by running ``fit()`` directly instead).

    ``models`` (optional): restrict to runs whose ``config.yaml``'s
    ``model.name`` is in this list -- every requested name must match at
    least one run, or this raises listing what's missing. Omit to return
    every run found.
    """
    candidates = [
        child for child in sorted(runs_root.iterdir())
        if child.is_dir() and (child / "config.yaml").exists()
    ]
    if not candidates:
        raise SystemExit(f"No run directories with config.yaml found under {runs_root}")

    if models is None:
        return candidates

    wanted = set(models)
    selected = []
    for run_dir in candidates:
        cfg = load_yaml(run_dir / "config.yaml")
        name = str((cfg.get("model", {}) or {}).get("name", ""))
        if name in wanted:
            selected.append(run_dir)

    found_names = {str((load_yaml(r / "config.yaml").get("model", {}) or {}).get("name", "")) for r in candidates}
    missing = wanted - found_names
    if missing:
        raise SystemExit(f"No run found under {runs_root} for model(s): {sorted(missing)}")
    return selected


def find_one_run(runs_root: Path, model_name: str) -> Path:
    """Exactly one run matching ``model_name`` under ``runs_root`` -- for
    tools that need a single named run per model (e.g.
    ``build_explainable_case_library.py``'s one-model-per-family selection).
    """
    matches = discover_run_dirs(runs_root, models=[model_name])
    if len(matches) > 1:
        raise SystemExit(
            f"Multiple runs found under {runs_root} for model {model_name!r}: {matches} -- "
            "point --runs-root at a directory with exactly one run per model, or rename the extras"
        )
    return matches[0]
