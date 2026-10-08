
from __future__ import annotations

from dataclasses import dataclass, replace as _dc_replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .schema import FeatureSchema, normalize_zipcode


@dataclass
class AlignedData:
    zipcodes: List[str]
    dates: List[pd.Timestamp]
    values: np.ndarray
    time_marks: np.ndarray
    schema: FeatureSchema

    @property
    def n_zip(self) -> int:
        return len(self.zipcodes)

    @property
    def n_time(self) -> int:
        return len(self.dates)

    @property
    def n_features(self) -> int:
        return int(self.values.shape[-1])


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}. Use csv/parquet/xlsx.")


def clean_raw_table(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    df = df.copy()

    # Drop empty rows
    if schema.time_col not in df.columns or schema.id_col not in df.columns:
        raise ValueError(f"Input df must contain {schema.id_col!r} and {schema.time_col!r}")

    df = df.dropna(subset=[schema.id_col, schema.time_col], how="any")

    # Parse dates
    if not pd.api.types.is_datetime64_any_dtype(df[schema.time_col]):
        df[schema.time_col] = pd.to_datetime(df[schema.time_col], errors="coerce")
    df = df.dropna(subset=[schema.time_col], how="any")

    # Normalize zipcode
    df[schema.id_col] = df[schema.id_col].map(normalize_zipcode)
    df = df[df[schema.id_col] != ""]

    # Drop non-feature columns
    for c in schema.drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Add time markers
    df["year"] = df[schema.time_col].dt.year.astype(int)
    df["month"] = df[schema.time_col].dt.month.astype(int)

    # Sort
    df = df.sort_values([schema.id_col, schema.time_col]).reset_index(drop=True)

    return df


def three_stage_impute(
    values: np.ndarray,
    *,
    per_feature_global_median: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Impute missing values in a (Z, T, D) array using a three-stage strategy:
    (1) forward-fill within each entity's time series, (2) fill remaining
    missing values with the entity-specific feature median, and (3) fall back
    to the global feature median. Entirely missing series are filled directly
    with the global feature median.
    """
    x = values.copy()

    Z, T, D = x.shape
    if per_feature_global_median is None:
        per_feature_global_median = np.nanmedian(x.reshape(-1, D), axis=0)
        per_feature_global_median = np.where(np.isfinite(per_feature_global_median), per_feature_global_median, 0.0)

    for z in range(Z):
        for d in range(D):
            s = x[z, :, d]
            if np.all(np.isnan(s)):
                x[z, :, d] = per_feature_global_median[d]
                continue
            ss = pd.Series(s)
            ss = ss.ffill()
            s2 = ss.to_numpy()
            if np.isnan(s2).any():
                med = np.nanmedian(s2)
                if np.isfinite(med):
                    s2 = np.where(np.isnan(s2), med, s2)
            if np.isnan(s2).any():
                s2 = np.where(np.isnan(s2), per_feature_global_median[d], s2)

            x[z, :, d] = s2

    return x


def _build_time_marks(dates: Sequence[pd.Timestamp]) -> Tuple[np.ndarray, Tuple[str, str]]:
    """Build ``[T, 2]`` time marks, auto-detecting granularity from the median
    gap between consecutive timestamps — three tiers:

    - Gap >= 20 days (this benchmark's original monthly housing/crime-by-ZIP
      use case): ``(year, month)`` — unchanged from the original behavior.
    - 1 day <= gap < 20 days (daily panels, e.g. UrbanGPT's NYC-crime, which
      samples once per day): ``(dow, month)`` — day-of-week (the dominant
      weekly periodicity at this cadence) plus month (seasonality). A plain
      ``(year, month)`` here would collapse every day in a month to the same
      mark and lose the day-of-week signal entirely, which is usually the
      more important one for daily crime/count data.
    - Gap < 1 day (5-min/hourly traffic panels): ``(tod_frac, dow)`` —
      ``tod_frac`` = fraction of the day elapsed, matching DCRNN/Graph
      WaveNet's own ``(ts - ts.floor('D')) / 1day`` convention; ``dow`` =
      day-of-week (Monday=0..Sunday=6), matching STAEformer/STID's
      ``nn.Embedding(7, dim)`` convention.
    """
    T = len(dates)
    if T >= 2:
        deltas_ns = np.diff(np.array([pd.Timestamp(d).value for d in dates]))
        median_delta_days = float(np.median(deltas_ns)) / 1e9 / 86400.0
    else:
        median_delta_days = 31.0  # degenerate T<2: fall back to the monthly scheme

    if median_delta_days >= 20.0:
        tm = np.zeros((T, 2), dtype=np.float32)
        for t, d in enumerate(dates):
            ts = pd.Timestamp(d)
            tm[t, 0] = float(ts.year)
            tm[t, 1] = float(ts.month)
        return tm, ("year", "month")

    if median_delta_days >= 1.0:
        tm = np.zeros((T, 2), dtype=np.float32)
        for t, d in enumerate(dates):
            ts = pd.Timestamp(d)
            tm[t, 0] = float(ts.dayofweek)
            tm[t, 1] = float(ts.month)
        return tm, ("dow", "month")

    tm = np.zeros((T, 2), dtype=np.float32)
    for t, d in enumerate(dates):
        ts = pd.Timestamp(d)
        tod_frac = (ts - ts.normalize()) / pd.Timedelta(days=1)
        tm[t, 0] = float(tod_frac)
        tm[t, 1] = float(ts.dayofweek)
    return tm, ("tod_frac", "dow")


def align_to_tensor(
    df: pd.DataFrame,
    schema: FeatureSchema,
    *,
    impute: bool = True,
    coerce_negative_to_zero: bool = True,
) -> AlignedData:
    df = df.copy()

    # Ensure schema.continuous_cols exist
    missing_cols = [c for c in schema.continuous_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns required by schema.continuous_cols: {missing_cols}")

    dates = sorted(df[schema.time_col].unique())
    dates = [pd.Timestamp(d) for d in dates]
    date_to_t = {d: i for i, d in enumerate(dates)}

    zipcodes = sorted(df[schema.id_col].unique())
    zip_to_i = {z: i for i, z in enumerate(zipcodes)}

    Z = len(zipcodes)
    T = len(dates)
    D = len(schema.continuous_cols)

    values = np.full((Z, T, D), np.nan, dtype=np.float32)

    tm, time_mark_cols = _build_time_marks(dates)
    if time_mark_cols != schema.time_mark_cols:
        schema = _dc_replace(schema, time_mark_cols=time_mark_cols)

    cols = list(schema.continuous_cols)
    values_arr = df[cols].to_numpy(dtype=np.float32)  # shape: (num_rows, D)
    zi_arr = df[schema.id_col].map(zip_to_i).to_numpy()
    ti_arr = df[schema.time_col].map(lambda d: date_to_t[pd.Timestamp(d)]).to_numpy()
    values[zi_arr, ti_arr, :] = values_arr

    # for _, row in df.iterrows():
    #     z = row[schema.id_col]
    #     d = row[schema.time_col]
    #     zi = zip_to_i[z]
    #     ti = date_to_t[pd.Timestamp(d)]
    #     vals = row[cols].to_numpy(dtype=np.float32, copy=False)
    #     values[zi, ti, :] = vals

    if coerce_negative_to_zero:
        values = np.where(values < 0, 0.0, values)

    if impute:
        values = three_stage_impute(values)

    return AlignedData(
        zipcodes=list(zipcodes),
        dates=dates,
        values=values.astype(np.float32, copy=False),
        time_marks=tm,
        schema=schema,
    )


def load_aligned(
    path: Union[str, Path],
    *,
    schema: Optional[FeatureSchema] = None,
    target_col: str = "price",
    id_col: str = "zipcode",
    time_col: str = "date",
    drop_cols: Sequence[str] = ("city", "city_full", "metro"),
    feature_cols: Optional[Sequence[str]] = None,
    impute: bool = True,
) -> AlignedData:
    df = read_table(path)

    if schema is None:
        schema = FeatureSchema.infer(
            df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            drop_cols=drop_cols,
            feature_cols=feature_cols,
        )

    df = clean_raw_table(df, schema)
    return align_to_tensor(df, schema, impute=impute)


def load_metr_la(
    path: Union[str, Path],
    *,
    target_col: str = "speed",
    feature_cols: Optional[Sequence[str]] = None,
    impute: bool = True,
) -> AlignedData:
    """Load METR-LA's standard ``.h5`` format (pandas DataFrame, DatetimeIndex
    rows x sensor-id columns, 5-min speed readings — confirmed from
    liyaguang/DCRNN and nnzhan/Graph-WaveNet's own data loaders) into this
    benchmark's ``AlignedData`` contract (``[N, T, D]``).

    A ``time_of_day`` column (fraction of the day in ``[0,1)``, matching
    DCRNN/Graph WaveNet's own ``generate_training_data.py`` convention
    exactly) is always added and can be selected via ``feature_cols`` for
    models that want it as a raw input channel (DCRNN/Graph
    WaveNet/MTGNN — confirmed from source to need no model-side change for
    this). Models that instead want tod/dow via dedicated embedding tables
    (STAEformer/STID) get it from ``AlignedData.time_marks``
    (``tod_frac, dow``) regardless of ``feature_cols``.

    Requires ``pandas``' HDF5 support (the ``tables`` package).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_hdf(path).sort_index()
    sensor_ids = [str(c) for c in df.columns]
    dates = [pd.Timestamp(d) for d in df.index]
    n, t = len(sensor_ids), len(dates)

    speed = df.to_numpy(dtype=np.float32).T[:, :, None]  # [N, T, 1]
    # METR-LA encodes missing sensor readings as 0 -- not a real zero speed.
    speed = np.where(speed == 0.0, np.nan, speed)

    tod = np.array([(d - d.normalize()) / pd.Timedelta(days=1) for d in dates], dtype=np.float32)
    tod_tiled = np.tile(tod[None, :, None], (n, 1, 1))  # [N, T, 1]
    values = np.concatenate([speed, tod_tiled], axis=-1)  # [N, T, 2]
    available_cols = [target_col, "time_of_day"]

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in available_cols]
        if missing:
            raise ValueError(f"feature_cols not available for METR-LA: {missing} (available: {available_cols})")
        keep = [target_col] + [c for c in feature_cols if c != target_col]
    else:
        keep = available_cols
    values = values[:, :, [available_cols.index(c) for c in keep]]

    if impute:
        values = three_stage_impute(values)

    tm, time_mark_cols = _build_time_marks(dates)
    schema = FeatureSchema(
        id_col="sensor_id", time_col="date", target_col=target_col,
        drop_cols=(), time_mark_cols=time_mark_cols, continuous_cols=tuple(keep),
    )
    return AlignedData(zipcodes=sensor_ids, dates=dates, values=values.astype(np.float32, copy=False),
                        time_marks=tm, schema=schema)


def load_pems08(
    path: Union[str, Path],
    *,
    target_col: str = "flow",
    feature_cols: Optional[Sequence[str]] = None,
    start_time: Optional[str] = None,
    freq_minutes: int = 5,
    impute: bool = True,
) -> AlignedData:
    """Load a PEMS0x-style ``.npz`` (key ``'data'``, shape ``[T, N, 3]`` =
    flow/occupancy/speed — confirmed convention) into this benchmark's
    ``AlignedData`` contract.

    PEMS0x releases don't embed real calendar timestamps, only a fixed
    ``freq_minutes``-interval step index — pass ``start_time`` (e.g.
    ``"2016-07-01"``, PEMS08's commonly-cited release start date) to anchor
    a real date axis and get a correct day-of-week signal. Without it, the
    date axis is anchored at an arbitrary epoch: the ``time_of_day``/
    ``tod_frac`` signal is still valid (it's a cyclic quantity, correct
    regardless of the anchor), but day-of-week is meaningless and is zeroed
    out in ``time_marks`` rather than fabricated.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    raw = np.load(path)
    if "data" not in raw:
        raise ValueError(f"{path} must contain a 'data' array (np.savez(path, data=...) PEMS0x format)")
    data = raw["data"]
    if data.ndim != 3:
        raise ValueError(f"expected data.ndim == 3 ([T,N,C]), got shape {data.shape}")
    t, n, c = data.shape

    channel_names = ["flow", "occupancy", "speed"][:c]
    if target_col not in channel_names:
        raise ValueError(f"target_col={target_col!r} not in available PEMS08 channels {channel_names}")

    anchor = start_time if start_time is not None else "1970-01-01"
    dates = list(pd.date_range(anchor, periods=t, freq=f"{int(freq_minutes)}min"))
    sensor_ids = [str(i) for i in range(n)]

    values = data.transpose(1, 0, 2).astype(np.float32)  # [N, T, C]
    tod = np.array([(d - d.normalize()) / pd.Timedelta(days=1) for d in dates], dtype=np.float32)
    tod_tiled = np.tile(tod[None, :, None], (n, 1, 1))  # [N, T, 1]
    values = np.concatenate([values, tod_tiled], axis=-1)
    available_cols = channel_names + ["time_of_day"]

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in available_cols]
        if missing:
            raise ValueError(f"feature_cols not available for PEMS08: {missing} (available: {available_cols})")
        keep = [target_col] + [c for c in feature_cols if c != target_col]
    else:
        keep = available_cols
    values = values[:, :, [available_cols.index(c) for c in keep]]

    if impute:
        values = three_stage_impute(values)

    tm, time_mark_cols = _build_time_marks(dates)
    if start_time is None:
        tm[:, 1] = 0.0  # day-of-week needs a real calendar anchor -- don't fabricate it

    schema = FeatureSchema(
        id_col="sensor_id", time_col="date", target_col=target_col,
        drop_cols=(), time_mark_cols=time_mark_cols, continuous_cols=tuple(keep),
    )
    return AlignedData(zipcodes=sensor_ids, dates=dates, values=values.astype(np.float32, copy=False),
                        time_marks=tm, schema=schema)


def _urbangpt_time_axis(t: int, start_time: Optional[str], freq_minutes: int) -> List[pd.Timestamp]:
    anchor = start_time if start_time is not None else "1970-01-01"
    return list(pd.date_range(anchor, periods=t, freq=f"{int(freq_minutes)}min"))


def _zero_anchor_dependent_marks(tm: np.ndarray, time_mark_cols: Tuple[str, str], start_time: Optional[str]) -> None:
    """Zero out whichever time-mark columns need a real calendar anchor to
    be meaningful, when none was given (``start_time is None``) — every
    mark except ``tod_frac`` (a cyclic quantity, valid regardless of the
    anchor). In-place. See ``load_pems08``'s docstring for the same
    reasoning applied to its always-subdaily (hence single-column) case.
    """
    if start_time is not None:
        return
    for i, name in enumerate(time_mark_cols):
        if name != "tod_frac":
            tm[:, i] = 0.0


def _select_channels(
    values: np.ndarray, available_cols: List[str], *, target_col: str, feature_cols: Optional[Sequence[str]],
) -> Tuple[np.ndarray, List[str]]:
    if target_col not in available_cols:
        raise ValueError(f"target_col={target_col!r} not in available channels {available_cols}")
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in available_cols]
        if missing:
            raise ValueError(f"feature_cols not available: {missing} (available: {available_cols})")
        keep = [target_col] + [c for c in feature_cols if c != target_col]
    else:
        keep = [target_col] + [c for c in available_cols if c != target_col]
    return values[:, :, [available_cols.index(c) for c in keep]], keep


def _load_flat_urbangpt_npz(
    path: Union[str, Path],
    *,
    channel_names: List[str],
    id_prefix: str,
    target_col: str,
    feature_cols: Optional[Sequence[str]],
    start_time: Optional[str],
    freq_minutes: int,
    impute: bool,
) -> AlignedData:
    """Shared loader for UrbanGPT-style flat ``[N, T, C]`` npz panels
    (NYC-taxi, CHI-taxi — confirmed shape/key from HKUDS/UrbanGPT's own
    ``instruction_generate/load_dataset.py``: ``np.load(path)['data']``)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    raw = np.load(path)
    if "data" not in raw:
        raise ValueError(f"{path} must contain a 'data' array (np.savez(path, data=...))")
    data = raw["data"]
    if data.ndim != 3:
        raise ValueError(f"expected data.ndim == 3 ([N,T,C]), got shape {data.shape}")
    n, t, c = data.shape
    if c != len(channel_names):
        raise ValueError(f"expected {len(channel_names)} channels {channel_names}, got {c} in {path}")

    dates = _urbangpt_time_axis(t, start_time, freq_minutes)
    ids = [f"{id_prefix}{i}" for i in range(n)]
    values, keep = _select_channels(data.astype(np.float32), channel_names, target_col=target_col, feature_cols=feature_cols)

    if impute:
        values = three_stage_impute(values)

    tm, time_mark_cols = _build_time_marks(dates)
    _zero_anchor_dependent_marks(tm, time_mark_cols, start_time)

    schema = FeatureSchema(id_col="region_id", time_col="date", target_col=target_col,
                            drop_cols=(), time_mark_cols=time_mark_cols, continuous_cols=tuple(keep))
    return AlignedData(zipcodes=ids, dates=dates, values=values.astype(np.float32, copy=False),
                        time_marks=tm, schema=schema)


def _load_grid_urbangpt_npz(
    path: Union[str, Path],
    *,
    channel_names: List[str],
    id_prefix: str,
    target_col: str,
    feature_cols: Optional[Sequence[str]],
    start_time: Optional[str],
    freq_minutes: int,
    impute: bool,
) -> AlignedData:
    """Shared loader for UrbanGPT-style gridded ``[Ny, Nx, T, C]`` npz panels
    (NYC-bike, NYC-crime — confirmed shape/key from source). Flattened to
    ``N = Ny*Nx`` nodes in row-major (``i*Nx+j``) order via a plain reshape,
    with ids ``"{id_prefix}{i}_{j}"`` recording the original grid position
    (useful for a future grid-adjacency graph builder, since these datasets
    ship with **no predefined graph** — confirmed from source, see
    ``load_nyc_bike``/``load_nyc_crime`` docstrings).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    raw = np.load(path)
    if "data" not in raw:
        raise ValueError(f"{path} must contain a 'data' array (np.savez(path, data=...))")
    data = raw["data"]
    if data.ndim != 4:
        raise ValueError(f"expected data.ndim == 4 ([Ny,Nx,T,C]), got shape {data.shape}")
    ny, nx, t, c = data.shape
    if c != len(channel_names):
        raise ValueError(f"expected {len(channel_names)} channels {channel_names}, got {c} in {path}")

    dates = _urbangpt_time_axis(t, start_time, freq_minutes)
    ids = [f"{id_prefix}{i}_{j}" for i in range(ny) for j in range(nx)]
    flat = data.reshape(ny * nx, t, c).astype(np.float32)
    values, keep = _select_channels(flat, channel_names, target_col=target_col, feature_cols=feature_cols)

    if impute:
        values = three_stage_impute(values)

    tm, time_mark_cols = _build_time_marks(dates)
    _zero_anchor_dependent_marks(tm, time_mark_cols, start_time)

    schema = FeatureSchema(id_col="region_id", time_col="date", target_col=target_col,
                            drop_cols=(), time_mark_cols=time_mark_cols, continuous_cols=tuple(keep))
    return AlignedData(zipcodes=ids, dates=dates, values=values.astype(np.float32, copy=False),
                        time_marks=tm, schema=schema)


def load_nyc_taxi(
    path: Union[str, Path], *, target_col: str = "inflow", feature_cols: Optional[Sequence[str]] = None,
    start_time: Optional[str] = "2016-01-01", freq_minutes: int = 30, impute: bool = True,
) -> AlignedData:
    """NYC-taxi from the UrbanGPT benchmark (HKUDS/UrbanGPT, KDD 2024) — flat
    ``[263, T, 2]`` npz, confirmed channels ``inflow``/``outflow`` (per the
    dataset's own HuggingFace preview: ``bjdwh/ST_data_urbangpt``). 30-min
    sampling, Jan 2016 - Dec 2021 (confirmed coverage). **No predefined
    graph ships with this dataset** (confirmed from source: UrbanGPT's own
    ``ST_Enc`` stores but never uses an ``adj_mx`` — it's pure dilated-conv
    over the node axis, no graph convolution at all) — use one of this
    registry's ``requires_graph=False`` models, or build your own (e.g. via
    ``scripts/build_knn_graph.py`` if you have region coordinates).
    """
    return _load_flat_urbangpt_npz(
        path, channel_names=["inflow", "outflow"], id_prefix="taxi_",
        target_col=target_col, feature_cols=feature_cols, start_time=start_time,
        freq_minutes=freq_minutes, impute=impute,
    )


def load_chi_taxi(
    path: Union[str, Path], *, target_col: str = "inflow", feature_cols: Optional[Sequence[str]] = None,
    start_time: Optional[str] = None, freq_minutes: int = 60, impute: bool = True,
) -> AlignedData:
    """CHI-taxi from the UrbanGPT benchmark — flat ``[77, T, 2]`` npz,
    channels ``inflow``/``outflow`` (by analogy with NYC-taxi's confirmed
    convention — not independently re-confirmed for Chicago specifically).
    Hourly sampling (inferred from its 17520-step length = 2 years hourly);
    its exact calendar start date wasn't confirmed, so pass ``start_time``
    if you know it — otherwise ``time_marks``' day-of-week is zeroed rather
    than fabricated (time-of-day stays valid, since it's cyclic regardless
    of the anchor). **No predefined graph** — same as ``load_nyc_taxi``.
    """
    return _load_flat_urbangpt_npz(
        path, channel_names=["inflow", "outflow"], id_prefix="chi_",
        target_col=target_col, feature_cols=feature_cols, start_time=start_time,
        freq_minutes=freq_minutes, impute=impute,
    )


def load_nyc_bike(
    path: Union[str, Path], *, target_col: str = "inflow", feature_cols: Optional[Sequence[str]] = None,
    start_time: Optional[str] = "2016-01-01", freq_minutes: int = 30, impute: bool = True,
) -> AlignedData:
    """NYC-bike from the UrbanGPT benchmark — gridded ``[46, 47, T, 2]`` npz
    (2162 = 46*47 regions, confirmed count), flattened row-major to a flat
    node axis. Channels ``inflow``/``outflow`` (by analogy with NYC-taxi's
    confirmed convention — not independently re-confirmed for bike
    specifically). 30-min sampling, Jan 2016 - Dec 2021 (confirmed
    coverage). **No predefined graph ships with this dataset** (confirmed
    from source — same as ``load_nyc_taxi``), but since this one *is* a
    literal 2D grid, a 4-/8-connected grid adjacency is a natural (and
    easy) graph to build yourself from the ``"{i}_{j}"`` ids this loader
    assigns — not provided out of the box here.
    """
    return _load_grid_urbangpt_npz(
        path, channel_names=["inflow", "outflow"], id_prefix="bike_",
        target_col=target_col, feature_cols=feature_cols, start_time=start_time,
        freq_minutes=freq_minutes, impute=impute,
    )


def load_nyc_crime(
    path: Union[str, Path], *, target_col: str = "burglaries", feature_cols: Optional[Sequence[str]] = None,
    start_time: Optional[str] = "2016-01-01", freq_minutes: int = 1440, impute: bool = True,
) -> AlignedData:
    """NYC-crime from the UrbanGPT benchmark — gridded ``[46, 47, T, 4]``
    npz, flattened row-major. Daily sampling, Jan 2016 - Dec 2021 (confirmed
    coverage). Channel naming: the array is sliced in two confirmed pairs
    (``[...,0:2]``, ``[...,2:4]``) in UrbanGPT's own loader, and the
    dataset's HuggingFace preview confirms the two *pair leaders* are
    ``burglaries``/``larcenies`` counts — the second channel of each pair's
    exact meaning wasn't independently confirmed, so it's named
    ``*_aux`` here; override ``feature_cols``/``target_col`` if you know
    the true semantics. **No predefined graph** (same as ``load_nyc_taxi``;
    see ``load_nyc_bike`` re: building your own grid adjacency).
    """
    return _load_grid_urbangpt_npz(
        path, channel_names=["burglaries", "burglaries_aux", "larcenies", "larcenies_aux"], id_prefix="crime_",
        target_col=target_col, feature_cols=feature_cols, start_time=start_time,
        freq_minutes=freq_minutes, impute=impute,
    )


def load_aligned_from_cfg(cfg: Dict[str, Any]) -> AlignedData:
    """Dispatch to the right loader based on ``data.loader`` (default
    ``"csv"``, so every existing dataset config is unaffected), then apply
    ``data.n_zip`` subsampling. Shared by ``experiments.sweep.run_one_cfg``
    and ``experiments.run_loader.load_run`` so both paths pick up new
    loaders identically.
    """
    data_cfg = cfg.get("data", {}) or {}
    loader = str(data_cfg.get("loader", "csv")).lower()

    if loader == "csv":
        aligned = load_aligned(
            data_cfg.get("path"),
            target_col=str(data_cfg.get("target_col", "price")),
            id_col=str(data_cfg.get("id_col", "zipcode")),
            time_col=str(data_cfg.get("time_col", "date")),
            drop_cols=data_cfg.get("drop_cols", ("city", "city_full", "metro")),
            feature_cols=data_cfg.get("feature_cols"),
            impute=bool(data_cfg.get("impute", True)),
        )
    elif loader == "metr_la":
        aligned = load_metr_la(
            data_cfg.get("path"),
            target_col=str(data_cfg.get("target_col", "speed")),
            feature_cols=data_cfg.get("feature_cols"),
            impute=bool(data_cfg.get("impute", True)),
        )
    elif loader == "pems08":
        aligned = load_pems08(
            data_cfg.get("path"),
            target_col=str(data_cfg.get("target_col", "flow")),
            feature_cols=data_cfg.get("feature_cols"),
            start_time=data_cfg.get("start_time"),
            freq_minutes=int(data_cfg.get("freq_minutes", 5)),
            impute=bool(data_cfg.get("impute", True)),
        )
    elif loader in ("nyc_taxi", "chi_taxi", "nyc_bike", "nyc_crime"):
        _urbangpt_loaders = {
            "nyc_taxi": (load_nyc_taxi, "inflow", 30),
            "chi_taxi": (load_chi_taxi, "inflow", 60),
            "nyc_bike": (load_nyc_bike, "inflow", 30),
            "nyc_crime": (load_nyc_crime, "burglaries", 1440),
        }
        fn, default_target, default_freq = _urbangpt_loaders[loader]
        aligned = fn(
            data_cfg.get("path"),
            target_col=str(data_cfg.get("target_col", default_target)),
            feature_cols=data_cfg.get("feature_cols"),
            start_time=data_cfg.get("start_time", "2016-01-01" if loader != "chi_taxi" else None),
            freq_minutes=int(data_cfg.get("freq_minutes", default_freq)),
            impute=bool(data_cfg.get("impute", True)),
        )
    else:
        raise ValueError(
            f"Unknown data.loader={loader!r} (expected one of: csv, metr_la, pems08, "
            "nyc_taxi, chi_taxi, nyc_bike, nyc_crime)"
        )

    n_zip = int(data_cfg.get("n_zip", 0) or 0)
    return subsample_zips(aligned, n_zip)


def subsample_zips(aligned: AlignedData, n_zip: int) -> AlignedData:
    """Keep only the first ``n_zip`` ZIPs (no-op if ``n_zip <= 0`` or already smaller)."""
    if n_zip <= 0 or aligned.n_zip <= n_zip:
        return aligned
    zips = aligned.zipcodes[:n_zip]
    zip_mask = np.isin(np.array(aligned.zipcodes), np.array(zips))
    kept = list(np.array(aligned.zipcodes)[zip_mask])
    return AlignedData(
        zipcodes=kept,
        dates=aligned.dates,
        values=aligned.values[zip_mask],
        time_marks=aligned.time_marks,
        schema=aligned.schema,
    )
