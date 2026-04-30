"""
Weather feature builder — NWS alerts → (port_id, week_ending) feature rows.

Output columns:
  alert_count              total NWS alerts in the week
  extreme_alert_count      alerts with severity == "Extreme"
  severe_alert_count       alerts with severity == "Severe"
  has_hurricane_alert      1 if any hurricane/typhoon alert present
  has_high_wind_alert      1 if any high-wind or gale alert present
  has_flood_alert          1 if any flood/storm-surge alert present
  has_fog_alert            1 if any fog/visibility alert present
  max_severity_score       highest severity ordinal (Extreme=4, Severe=3, …)
"""

from __future__ import annotations

import pandas as pd

from src.features.utils import align_series_to_week, fill_count_columns, fill_flag_columns

# Map NWS severity labels to ordinal scores
SEVERITY_SCORE = {"Extreme": 4, "Severe": 3, "Moderate": 2, "Minor": 1}

# Event-type keyword groups for flag columns
_HURRICANE_KW  = ("hurricane", "typhoon", "tropical storm", "tropical cyclone")
_HIGH_WIND_KW  = ("high wind", "wind advisory", "gale", "windstorm")
_FLOOD_KW      = ("flood", "storm surge", "coastal flood", "flash flood")
_FOG_KW        = ("dense fog", "fog advisory", "low visibility")

COUNT_COLS = [
    "alert_count", "extreme_alert_count", "severe_alert_count",
]
FLAG_COLS = [
    "has_hurricane_alert", "has_high_wind_alert", "has_flood_alert", "has_fog_alert",
]


def compute_weather_features(alerts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate NWS alerts into weekly features per port.

    Args:
        alerts_df: flat DataFrame from BronzeReader.read_nws()

    Returns:
        DataFrame with columns [port_id, week_ending, <weather features>]
        Empty if alerts_df is empty.
    """
    if alerts_df.empty or "port_id" not in alerts_df.columns:
        return pd.DataFrame()

    df = align_series_to_week(alerts_df, date_col="effective")

    # Severity ordinal
    df["severity_score"] = df["severity"].map(SEVERITY_SCORE).fillna(0).astype(int)

    # Event-type flags per row
    event_lower = df["event_type"].str.lower().fillna("")
    df["_hurricane"] = event_lower.apply(lambda x: int(any(k in x for k in _HURRICANE_KW)))
    df["_high_wind"] = event_lower.apply(lambda x: int(any(k in x for k in _HIGH_WIND_KW)))
    df["_flood"]     = event_lower.apply(lambda x: int(any(k in x for k in _FLOOD_KW)))
    df["_fog"]       = event_lower.apply(lambda x: int(any(k in x for k in _FOG_KW)))

    # Aggregate to (port_id, week_ending)
    agg = (
        df.groupby(["port_id", "week_ending"], as_index=False)
        .agg(
            alert_count          = ("alert_id",       "count"),
            extreme_alert_count  = ("severity_score", lambda s: (s == 4).sum()),
            severe_alert_count   = ("severity_score", lambda s: (s == 3).sum()),
            has_hurricane_alert  = ("_hurricane",     "max"),
            has_high_wind_alert  = ("_high_wind",     "max"),
            has_flood_alert      = ("_flood",         "max"),
            has_fog_alert        = ("_fog",           "max"),
            max_severity_score   = ("severity_score", "max"),
        )
    )

    agg = fill_count_columns(agg, COUNT_COLS)
    agg = fill_flag_columns(agg, FLAG_COLS)
    return agg
