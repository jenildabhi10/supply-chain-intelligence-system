"""
Shared utilities for the feature engineering pipeline.

All functions are pure (no I/O, no side effects) so they are trivially testable.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

# ─── Week alignment ───────────────────────────────────────────────────────────

def week_ending_sunday(dt: pd.Timestamp | date | str) -> pd.Timestamp:
    """
    Snap any date to the Sunday that ends its calendar week (Mon–Sun).

    Examples:
        Thursday 2024-04-11  →  Sunday 2024-04-14
        Sunday   2024-04-14  →  Sunday 2024-04-14
        Monday   2024-04-08  →  Sunday 2024-04-14
    """
    ts = pd.Timestamp(dt).normalize()
    days_ahead = (6 - ts.weekday()) % 7       # 0 if already Sunday
    return ts + pd.Timedelta(days=days_ahead)


def align_series_to_week(
    df: pd.DataFrame,
    date_col: str,
    new_col: str = "week_ending",
) -> pd.DataFrame:
    """Add a `week_ending` column snapped to Sunday for every row in df."""
    df = df.copy()
    df[new_col] = pd.to_datetime(df[date_col]).apply(week_ending_sunday)
    return df


def week_spine(
    start: pd.Timestamp | str,
    end:   pd.Timestamp | str,
) -> pd.Series:
    """
    Return a Series of weekly Sunday dates covering [start, end] inclusive.
    The first element is the Sunday of the week that contains `start`.
    """
    start_snap = week_ending_sunday(start)
    end_snap   = week_ending_sunday(end)
    return pd.date_range(start=start_snap, end=end_snap, freq="W-SUN")


def build_port_week_spine(
    port_ids:   list[str],
    start_date: pd.Timestamp | str,
    end_date:   pd.Timestamp | str,
) -> pd.DataFrame:
    """
    Create the full cartesian product of (port_id × week_ending).

    This is the base DataFrame onto which all feature DataFrames are joined.
    A row with all-NaN features means we have the time slot but no data yet.
    """
    weeks  = week_spine(start_date, end_date)
    rows   = [(pid, week) for pid in port_ids for week in weeks]
    return pd.DataFrame(rows, columns=["port_id", "week_ending"])


# ─── Rolling / window statistics ─────────────────────────────────────────────

def rolling_quantile(
    series: pd.Series,
    window: int,
    quantile: float,
    min_periods: int = 4,
) -> pd.Series:
    """Rolling quantile with a minimum period guard."""
    return series.rolling(window=window, min_periods=min_periods).quantile(quantile)


def rolling_zscore(
    series:      pd.Series,
    window:      int = 52,
    min_periods: int = 4,
) -> pd.Series:
    """Z-score of each value relative to a rolling mean/std window."""
    roll   = series.rolling(window=window, min_periods=min_periods)
    mean   = roll.mean()
    std    = roll.std().clip(lower=1e-6)   # avoid /0
    return (series - mean) / std


def pct_change_safe(series: pd.Series, periods: int = 1) -> pd.Series:
    """
    Percentage change with division-by-zero safety.
    Returns 0.0 when the previous value is 0 or NaN.
    """
    prev = series.shift(periods)
    change = ((series - prev) / prev.replace(0, np.nan)) * 100
    return change.fillna(0.0)


# ─── NA handling ──────────────────────────────────────────────────────────────

def fill_count_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill NA in count columns with 0 (no data observed = zero events)."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return df


def fill_flag_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill NA in binary flag columns with 0."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return df


def clip_outliers(
    series:    pd.Series,
    low_pct:   float = 1.0,
    high_pct:  float = 99.0,
) -> pd.Series:
    """Clip a Series to [low_pct, high_pct] percentile bounds."""
    lo = np.nanpercentile(series.dropna(), low_pct)
    hi = np.nanpercentile(series.dropna(), high_pct)
    return series.clip(lower=lo, upper=hi)


# ─── Data completeness tracking ───────────────────────────────────────────────

ALL_SOURCES = ["nws", "gdelt", "bts", "usgs", "firms"]


def compute_data_completeness(
    df: pd.DataFrame,
    source_indicator_cols: dict[str, str],
) -> pd.Series:
    """
    Compute a 0–1 data completeness score per row.

    source_indicator_cols: {source_name: column_name_that_is_non_null_when_source_present}
    A source "counts" as present for a row if its indicator column is not NaN.
    """
    n = len(source_indicator_cols)
    if n == 0:
        return pd.Series(0.0, index=df.index)

    present = sum(
        df[col].notna().astype(int)
        for col in source_indicator_cols.values()
        if col in df.columns
    )
    return (present / n).round(3)


# ─── GDELT date parsing ───────────────────────────────────────────────────────

def gdelt_sqldate_to_timestamp(sqldate: pd.Series) -> pd.Series:
    """Convert GDELT's YYYYMMDD integer/string dates to Timestamps."""
    return pd.to_datetime(sqldate.astype(str), format="%Y%m%d", errors="coerce")
