
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


def _is_numeric_dtype(dtype) -> bool:
    try:
        return pd.api.types.is_numeric_dtype(dtype)
    except Exception:
        return False


def normalize_zipcode(z: object) -> str:
    if z is None or (isinstance(z, float) and pd.isna(z)):
        return ""
    if isinstance(z, (int,)):
        s = str(z)
    elif isinstance(z, float):
        # Excel often reads ZIP codes as float 
        s = str(int(z))
    else:
        s = str(z).strip()

    # Remove trailing ".0" if present
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]

    # Keep only digits if it's purely numeric-ish
    if s.isdigit():
        if len(s) < 5:
            s = s.zfill(5)
        return s

    # Otherwise return as-is
    return s


@dataclass(frozen=True)
class FeatureSchema:
    id_col: str = "zipcode"
    time_col: str = "date"
    target_col: str = "price"

    # columns to drop from modeling
    drop_cols: Tuple[str, ...] = ("city", "city_full", "metro")

    # time marker columns derived from time_col
    time_mark_cols: Tuple[str, ...] = ("year", "month")

    # all continuous variables available for modeling
    continuous_cols: Tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def infer(
        cls,
        df: pd.DataFrame,
        *,
        id_col: str = "zipcode",
        time_col: str = "date",
        target_col: str = "price",
        drop_cols: Sequence[str] = ("city", "city_full"),
        extra_exclude: Sequence[str] = (),
        feature_cols: Optional[Sequence[str]] = None,
    ) -> "FeatureSchema":
        """Build the modeling schema for a dataset.

        If ``feature_cols`` is given, the schema uses exactly those columns
        (plus ``target_col``) — nothing else is read/modeled. Otherwise
        continuous columns are auto-inferred from dtypes, excluding id/time
        and known non-features (previous default behavior).
        """
        if target_col not in df.columns:
            raise ValueError(f"target_col={target_col!r} not found in df columns")

        if feature_cols is not None:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"feature_cols not found in df columns: {missing}")
            continuous = [target_col] + [c for c in feature_cols if c != target_col]
            return cls(
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                drop_cols=tuple(drop_cols),
                time_mark_cols=("year", "month"),
                continuous_cols=tuple(continuous),
            )

        exclude = set(drop_cols) | set(extra_exclude) | {id_col, time_col}
        exclude |= {"year", "month"}

        continuous = []
        for c in df.columns:
            if c in exclude:
                continue
            if _is_numeric_dtype(df[c].dtype):
                continuous.append(c)

        # move target to front for stable ordering
        continuous = [target_col] + [c for c in continuous if c != target_col]

        return cls(
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            drop_cols=tuple(drop_cols),
            time_mark_cols=("year", "month"),
            continuous_cols=tuple(continuous),
        )

    def select_xy_cols(self, features_mode: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        mode = features_mode.upper()
        if mode == "S":
            return (self.target_col,), (self.target_col,)
        if mode == "MS":
            return self.continuous_cols, (self.target_col,)
        if mode == "M":
            return self.continuous_cols, self.continuous_cols
        raise ValueError(f"Unknown features_mode={features_mode!r}; expected one of 'S','MS','M'")
