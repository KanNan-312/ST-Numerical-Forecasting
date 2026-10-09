from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(obj)} in {p}")
    return obj


def deep_update(base: MutableMapping[str, Any], other: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


def resolve_relpaths(cfg: Dict[str, Any], *, root: str | Path, keys: Optional[list[str]] = None) -> Dict[str, Any]:
    rootp = Path(root).resolve()
    keys = keys or ["data.path"]
    for key in keys:
        parts = key.split(".")
        cur: Any = cfg
        for p in parts[:-1]:
            if not isinstance(cur, dict) or p not in cur:
                cur = None
                break
            cur = cur[p]
        if cur is None or not isinstance(cur, dict):
            continue
        leaf = parts[-1]
        val = cur.get(leaf, None)
        if isinstance(val, str):
            pp = Path(val)
            if not pp.is_absolute():
                cur[leaf] = str((rootp / pp).resolve())
    return cfg


def get_in(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in path.split("."):
        if not isinstance(cur, Mapping) or p not in cur:
            return default
        cur = cur[p]
    return cur


def pop_cli_overrides(overrides: Optional[list[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not overrides:
        return out

    def _parse_scalar(s: str) -> Any:
        ss = s.strip()
        if ss.lower() in ("true", "false"):
            return ss.lower() == "true"
        if ss.lower() in ("null", "none"):
            return None
        try:
            if "." in ss:
                return float(ss)
            return int(ss)
        except Exception:
            return ss

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}, expected key=value")
        k, v = item.split("=", 1)
        k = k.strip()
        v_parsed = _parse_scalar(v)
        parts = k.split(".")
        cur = out
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]  # type: ignore[assignment]
        cur[parts[-1]] = v_parsed
    return out
