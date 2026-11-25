
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: Tuple[int, int]
    val: Tuple[int, int]
    test: Tuple[int, int]

    def range(self, name: str) -> Tuple[int, int]:
        name = name.lower()
        if name == "train":
            return self.train
        if name == "val":
            return self.val
        if name == "test":
            return self.test
        raise ValueError(f"Unknown split name: {name}")

    @property
    def n_time(self) -> int:
        return self.test[1]


def make_ratio_split(n_time: int, train_ratio: float = 0.7, val_ratio: float = 0.1) -> TimeSplit:
    if n_time <= 0:
        raise ValueError("n_time must be positive")
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be in (0,1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    train_end = int(round(n_time * train_ratio))
    val_end = int(round(n_time * (train_ratio + val_ratio)))

    # Ensure at least 1 time step in each split when possible
    train_end = max(1, min(train_end, n_time - 2))
    val_end = max(train_end + 1, min(val_end, n_time - 1))

    return TimeSplit(train=(0, train_end), val=(train_end, val_end), test=(val_end, n_time))


def make_date_split(
    dates: Sequence[pd.Timestamp],
    *,
    train_end: str,
    val_end: str,
) -> TimeSplit:
    if len(dates) == 0:
        raise ValueError("dates is empty")

    dts = [pd.Timestamp(d) for d in dates]
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    if train_end_ts >= val_end_ts:
        raise ValueError("train_end must be < val_end")

    # find first index > boundary
    def _end_exclusive(bound: pd.Timestamp) -> int:
        for i, d in enumerate(dts):
            if d > bound:
                return i
        return len(dts)

    train_end_ex = _end_exclusive(train_end_ts)
    val_end_ex = _end_exclusive(val_end_ts)

    if train_end_ex < 1:
        raise ValueError("train_end is before the first timestamp")
    if val_end_ex <= train_end_ex:
        raise ValueError("val_end is not after train_end in the provided date axis")
    if val_end_ex >= len(dts):
        # still allow, but then test would be empty
        pass

    return TimeSplit(train=(0, train_end_ex), val=(train_end_ex, val_end_ex), test=(val_end_ex, len(dts)))


def make_exact_date_split(
    dates: Sequence[pd.Timestamp],
    *,
    val_start_date: str,
    test_start_date: str,
) -> TimeSplit:
    """Fully explicit split for calendar-indexed panels: train is everything
    before ``val_start_date``, val is ``[val_start_date, test_start_date)``,
    test is everything ``>= test_start_date``. No ratio is involved — the
    boundaries are exact dates, so the split stays identical as more history
    is appended, which ``make_ratio_split``'s proportional boundaries don't.

    Intended for datasets with a genuine, intuitive calendar axis (e.g. a
    monthly ZIP/region panel loaded from CSV) where "val starts Jan 2022,
    test starts Jan 2023" is how you'd naturally describe the split, as
    opposed to dense regularly-sampled numeric panels (e.g. PEMS/METR-LA
    5-min traffic npz data) where a 70/10/20 ratio split is the field's own
    convention (see :func:`make_ratio_split`).
    """
    if len(dates) == 0:
        raise ValueError("dates is empty")

    dts = [pd.Timestamp(d) for d in dates]
    val_start_ts = pd.Timestamp(val_start_date)
    test_start_ts = pd.Timestamp(test_start_date)
    if val_start_ts >= test_start_ts:
        raise ValueError(
            f"val_start_date={val_start_date!r} must be before test_start_date={test_start_date!r}"
        )

    def _first_at_or_after(ts: pd.Timestamp) -> int:
        return next((i for i, d in enumerate(dts) if d >= ts), len(dts))

    val_start = _first_at_or_after(val_start_ts)
    test_start = _first_at_or_after(test_start_ts)

    if val_start < 1:
        raise ValueError(f"val_start_date={val_start_date!r} leaves no room for a train split")
    if test_start <= val_start:
        raise ValueError(
            f"test_start_date={test_start_date!r} is not after val_start_date={val_start_date!r} "
            "in the provided date axis"
        )
    if test_start >= len(dts):
        raise ValueError(f"test_start_date={test_start_date!r} leaves no room for a test split")

    return TimeSplit(train=(0, val_start), val=(val_start, test_start), test=(test_start, len(dts)))


def make_split(
    n_time: int,
    dates: Optional[Sequence[pd.Timestamp]] = None,
    *,
    mode: str = "ratio",
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    val_start_date: Optional[str] = None,
    test_start_date: Optional[str] = None,
) -> TimeSplit:
    """Dispatch to one of two split protocols.

    ``mode="date"`` (recommended for calendar-indexed CSV panels, e.g. a
    monthly ZIP/region housing or crime panel): requires both
    ``val_start_date`` and ``test_start_date`` — see :func:`make_exact_date_split`.
    Train/val/test boundaries are exact dates, not ratios.

    ``mode="ratio"`` (default; recommended for dense regularly-sampled numeric
    panels, e.g. PEMS/METR-LA npz traffic data, where a 70/10/20-style split
    is the field's own convention): ``train_ratio``/``val_ratio`` split the
    series proportionally (see :func:`make_ratio_split`). For backward
    compatibility, passing ``test_start_date`` alone (no ``val_start_date``)
    under ``mode="ratio"`` keeps the legacy behavior: the test split starts at
    the first date ``>= test_start_date`` and everything before it is divided
    into train/val by ``train_ratio``/``val_ratio`` (relative to each other).
    """
    mode = (mode or "ratio").lower()

    if mode == "date":
        if val_start_date is None or test_start_date is None:
            raise ValueError("split.mode='date' requires both val_start_date and test_start_date")
        if not dates:
            raise ValueError("dates is required when split.mode='date'")
        if len(dates) != n_time:
            raise ValueError(f"len(dates)={len(dates)} does not match n_time={n_time}")
        return make_exact_date_split(dates, val_start_date=val_start_date, test_start_date=test_start_date)

    if mode != "ratio":
        raise ValueError(f"Unknown split.mode={mode!r} (expected 'ratio' or 'date')")

    if test_start_date is None:
        return make_ratio_split(n_time, train_ratio=train_ratio, val_ratio=val_ratio)

    # legacy partial-date behavior: exact test cutoff, ratio-split train/val before it
    if not dates:
        raise ValueError("dates is required when test_start_date is set")
    if len(dates) != n_time:
        raise ValueError(f"len(dates)={len(dates)} does not match n_time={n_time}")

    dts = [pd.Timestamp(d) for d in dates]
    cutoff = pd.Timestamp(test_start_date)
    test_start = next((i for i, d in enumerate(dts) if d >= cutoff), len(dts))
    if test_start < 2:
        raise ValueError(f"test_start_date={test_start_date!r} leaves no room for train/val")
    if test_start >= n_time:
        raise ValueError(f"test_start_date={test_start_date!r} is after the last available date")

    pre_n = test_start
    train_frac = train_ratio / (train_ratio + val_ratio)
    train_end = int(round(pre_n * train_frac))
    train_end = max(1, min(train_end, pre_n - 1))

    return TimeSplit(train=(0, train_end), val=(train_end, test_start), test=(test_start, n_time))
