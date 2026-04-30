"""
News feature builder — GDELT events → (port_id, week_ending) feature rows.

Output columns:
  gdelt_event_count          total filtered GDELT events near the port
  gdelt_strike_count         events with root code 14 (protests/strikes)
  gdelt_coerce_count         events with root code 17 (coercion / sanctions)
  gdelt_conflict_count       events with root code 20 (military/conflict)
  gdelt_goldstein_avg        mean Goldstein scale (-10=destabilising, +10=cooperative)
  gdelt_goldstein_min        worst (most destabilising) Goldstein score
  gdelt_avg_tone             mean AvgTone (negative = negative news framing)
  gdelt_num_articles_total   total article count (signal strength proxy)
  gdelt_num_mentions_total   total mention count
  gdelt_weighted_disruption  weighted disruption signal:
                             sum(-goldstein_scale * num_mentions) for neg-goldstein rows
                             Higher = more disruption coverage

The Goldstein scale runs from -10 (maximally destabilising) to +10 (cooperative).
We treat events with GoldsteinScale < 0 near ports as disruption signals.
"""

from __future__ import annotations

import pandas as pd

from src.features.utils import (
    align_series_to_week,
    fill_count_columns,
    gdelt_sqldate_to_timestamp,
)

COUNT_COLS = [
    "gdelt_event_count", "gdelt_strike_count",
    "gdelt_coerce_count", "gdelt_conflict_count",
    "gdelt_num_articles_total", "gdelt_num_mentions_total",
]


def compute_news_features(gdelt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate GDELT events into weekly news-signal features per port.

    Args:
        gdelt_df: flat DataFrame from BronzeReader.read_gdelt()

    Returns:
        DataFrame with columns [port_id, week_ending, <news features>]
        Empty if gdelt_df is empty.
    """
    if gdelt_df.empty or "nearest_port_id" not in gdelt_df.columns:
        return pd.DataFrame()

    df = gdelt_df.copy()
    df = df.rename(columns={"nearest_port_id": "port_id"})

    # Parse GDELT dates (YYYYMMDD) → Timestamps for week alignment
    df["_event_date"] = gdelt_sqldate_to_timestamp(df["sql_date"])
    df = align_series_to_week(df, date_col="_event_date")

    # Numeric coercions
    df["goldstein_scale"]  = pd.to_numeric(df.get("goldstein_scale",  pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    df["avg_tone"]         = pd.to_numeric(df.get("avg_tone",         pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    df["num_mentions"]     = pd.to_numeric(df.get("num_mentions",     pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
    df["num_articles"]     = pd.to_numeric(df.get("num_articles",     pd.Series(dtype=float)), errors="coerce").fillna(0).astype(int)
    df["event_root_code"]  = df.get("event_root_code", pd.Series("")).fillna("").astype(str)

    # Root-code flags per row
    df["_strike"]   = (df["event_root_code"] == "14").astype(int)
    df["_coerce"]   = (df["event_root_code"] == "17").astype(int)
    df["_conflict"] = (df["event_root_code"] == "20").astype(int)

    # Weighted disruption: sum of (-goldstein * mentions) for negative-goldstein events
    neg_mask = df["goldstein_scale"] < 0
    df["_disruption_weight"] = 0.0
    df.loc[neg_mask, "_disruption_weight"] = (
        -df.loc[neg_mask, "goldstein_scale"] * df.loc[neg_mask, "num_mentions"]
    )

    # Aggregate to (port_id, week_ending)
    agg = (
        df.groupby(["port_id", "week_ending"], as_index=False)
        .agg(
            gdelt_event_count          = ("global_event_id",   "count"),
            gdelt_strike_count         = ("_strike",           "sum"),
            gdelt_coerce_count         = ("_coerce",           "sum"),
            gdelt_conflict_count       = ("_conflict",         "sum"),
            gdelt_goldstein_avg        = ("goldstein_scale",   "mean"),
            gdelt_goldstein_min        = ("goldstein_scale",   "min"),
            gdelt_avg_tone             = ("avg_tone",          "mean"),
            gdelt_num_articles_total   = ("num_articles",      "sum"),
            gdelt_num_mentions_total   = ("num_mentions",      "sum"),
            gdelt_weighted_disruption  = ("_disruption_weight","sum"),
        )
    )

    agg = fill_count_columns(agg, COUNT_COLS)
    agg["gdelt_goldstein_avg"]       = agg["gdelt_goldstein_avg"].fillna(0.0)
    agg["gdelt_goldstein_min"]       = agg["gdelt_goldstein_min"].fillna(0.0)
    agg["gdelt_avg_tone"]            = agg["gdelt_avg_tone"].fillna(0.0)
    agg["gdelt_weighted_disruption"] = agg["gdelt_weighted_disruption"].fillna(0.0)

    return agg
