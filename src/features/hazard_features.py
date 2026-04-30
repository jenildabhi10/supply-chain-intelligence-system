"""
Hazard feature builder — USGS earthquakes + NASA FIRMS fires
→ (port_id, week_ending) feature rows.

Output columns (earthquake):
  eq_count                  earthquakes within radius in the week
  eq_max_magnitude          largest magnitude
  eq_energy_proxy           sum of 10^(1.5 * mag) — proportional to seismic energy
  eq_tsunami_flag           1 if any event had a tsunami warning
  eq_depth_min_km           shallowest event depth (shallow = more surface damage)

Output columns (fire):
  fire_count                total VIIRS fire detections within radius
  fire_high_confidence_count  detections with confidence == "high"
  fire_max_frp              peak Fire Radiative Power in MW (intensity proxy)
  fire_total_frp            sum FRP (cumulative energy proxy)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.utils import align_series_to_week, fill_count_columns, fill_flag_columns

EQ_COUNT_COLS   = ["eq_count"]
EQ_FLAG_COLS    = ["eq_tsunami_flag"]
FIRE_COUNT_COLS = ["fire_count", "fire_high_confidence_count"]


def compute_earthquake_features(usgs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate USGS earthquake events into weekly seismic-hazard features per port.

    Args:
        usgs_df: flat DataFrame from BronzeReader.read_usgs()

    Returns:
        DataFrame with columns [port_id, week_ending, <earthquake features>]
        Empty if usgs_df is empty.
    """
    if usgs_df.empty or "nearest_port_id" not in usgs_df.columns:
        return pd.DataFrame()

    df = usgs_df.copy()
    df = df.rename(columns={"nearest_port_id": "port_id"})
    df = align_series_to_week(df, date_col="occurred_at")

    df["magnitude"]    = pd.to_numeric(df.get("magnitude",    pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    df["depth_km"]     = pd.to_numeric(df.get("depth_km",     pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    df["tsunami_flag"] = df.get("tsunami_flag", pd.Series(False)).fillna(False).astype(int)

    # Seismic energy proxy: E ∝ 10^(1.5 * M) (simplified Gutenberg-Richter)
    df["_energy_proxy"] = np.power(10.0, 1.5 * df["magnitude"])

    agg = (
        df.groupby(["port_id", "week_ending"], as_index=False)
        .agg(
            eq_count          = ("usgs_id",      "count"),
            eq_max_magnitude  = ("magnitude",    "max"),
            eq_energy_proxy   = ("_energy_proxy","sum"),
            eq_tsunami_flag   = ("tsunami_flag", "max"),
            eq_depth_min_km   = ("depth_km",     "min"),
        )
    )

    agg = fill_count_columns(agg, EQ_COUNT_COLS)
    agg = fill_flag_columns(agg,  EQ_FLAG_COLS)
    agg["eq_max_magnitude"] = agg["eq_max_magnitude"].fillna(0.0)
    agg["eq_energy_proxy"]  = agg["eq_energy_proxy"].fillna(0.0)
    agg["eq_depth_min_km"]  = agg["eq_depth_min_km"].fillna(np.nan)
    return agg


def compute_fire_features(firms_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate NASA FIRMS fire detections into weekly fire-hazard features per port.

    Args:
        firms_df: flat DataFrame from BronzeReader.read_firms()

    Returns:
        DataFrame with columns [port_id, week_ending, <fire features>]
        Empty if firms_df is empty or FIRMS key was not configured.
    """
    if firms_df.empty or "nearest_port_id" not in firms_df.columns:
        return pd.DataFrame()

    df = firms_df.copy()
    df = df.rename(columns={"nearest_port_id": "port_id"})
    df = align_series_to_week(df, date_col="acquired_at")

    df["frp"]              = pd.to_numeric(df.get("frp",        pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    df["confidence"]       = df.get("confidence", pd.Series("")).fillna("")
    df["_high_confidence"] = (df["confidence"] == "high").astype(int)

    agg = (
        df.groupby(["port_id", "week_ending"], as_index=False)
        .agg(
            fire_count                = ("lat",             "count"),
            fire_high_confidence_count= ("_high_confidence","sum"),
            fire_max_frp              = ("frp",             "max"),
            fire_total_frp            = ("frp",             "sum"),
        )
    )

    agg = fill_count_columns(agg, FIRE_COUNT_COLS)
    agg["fire_max_frp"]   = agg["fire_max_frp"].fillna(0.0)
    agg["fire_total_frp"] = agg["fire_total_frp"].fillna(0.0)
    return agg
