from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    metrics_path: Path
    config_path: Path
    env_path: Path


def make_run_dir(
    *,
    root: str | Path,
    name: str,
    exist_ok: bool = False,
) -> RunPaths:
    rootp = Path(root)
    rootp.mkdir(parents=True, exist_ok=True)
    run_dir = rootp / name
    run_dir.mkdir(parents=True, exist_ok=exist_ok)

    return RunPaths(
        run_dir=run_dir,
        metrics_path=run_dir / "metrics.json",
        config_path=run_dir / "config.yaml",
        env_path=run_dir / "env.json",
    )


def save_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_yaml(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)


def collect_env() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": os.path.realpath(os.sys.executable),
    }
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        env["git_sha"] = sha
    except Exception:
        env["git_sha"] = None
    try:
        freeze = subprocess.check_output(["pip", "freeze"], stderr=subprocess.DEVNULL).decode().splitlines()
        env["pip_freeze"] = freeze
    except Exception:
        env["pip_freeze"] = None
    return env
