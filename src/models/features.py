"""
Model feature contract — shared between training and prediction.

This file is the single source of truth for which columns from the Gold table
feed into each model.  Never read feature lists from anywhere else.

Design notes:
- FEATURE_COLUMNS contains only lag/rolling/external signal features.
  `median_berthing_time_hrs` is intentionally excluded — it is the basis of
  the label and including the raw value leaks target information.
- Port ID is label-encoded and appended at training time so the model can
  learn fixed port-level effects (e.g. LA/LB is structurally more congested).
- All FEATURE_COLUMNS must exist in the Gold table produced by Phase 2.
  Add new columns there first, then add them here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import US_PORTS

if TYPE_CHECKING:
    import pandas as pd

# ─── Supervised model features ────────────────────────────────────────────────

FEATURE_COLUMNS: list[str] = [
    # Port activity — lag features (computed from past, zero leakage)
    "berthing_time_lag_1w",
    "berthing_time_lag_2w",
    "berthing_time_lag_4w",
    "berthing_time_pct_change_1w",
    "call_count_pct_change_1w",

    # Port activity — rolling window statistics
    "berthing_time_4w_avg",
    "berthing_time_4w_std",
    "berthing_time_52w_avg",
    "berthing_time_52w_std",
    "berthing_time_z_score_52w",

    # Calendar seasonality
    "week_of_year",
    "month",
    "is_q4",

    # Weather signals (NWS)
    "alert_count",
    "extreme_alert_count",
    "severe_alert_count",
    "has_hurricane_alert",
    "has_high_wind_alert",
    "has_flood_alert",
    "max_severity_score",

    # News signals (GDELT)
    "gdelt_event_count",
    "gdelt_strike_count",
    "gdelt_coerce_count",
    "gdelt_goldstein_avg",
    "gdelt_goldstein_min",
    "gdelt_weighted_disruption",

    # Seismic hazard (USGS)
    "eq_count",
    "eq_max_magnitude",
    "eq_energy_proxy",
    "eq_tsunami_flag",

    # Fire hazard (FIRMS)
    "fire_count",
    "fire_max_frp",
]

# Anomaly detector uses a tighter set — only features with clean distributions
ANOMALY_FEATURE_COLUMNS: list[str] = [
    "berthing_time_lag_1w",
    "berthing_time_pct_change_1w",
    "berthing_time_z_score_52w",
    "berthing_time_4w_avg",
    "alert_count",
    "gdelt_event_count",
    "gdelt_weighted_disruption",
    "eq_count",
    "fire_count",
]

TARGET_COLUMN  = "is_disruption_week"
PORT_ID_COLUMN = "port_id"

# ─── Port ID encoding ─────────────────────────────────────────────────────────

# Stable integer encoding for port IDs — must not change between training runs
PORT_ID_ENCODING: dict[str, int] = {
    port.id: idx for idx, port in enumerate(sorted(US_PORTS, key=lambda p: p.id))
}
PORT_ID_DECODING: dict[int, str] = {v: k for k, v in PORT_ID_ENCODING.items()}


def encode_port_ids(df) -> pd.Series:
    """Map port_id strings to stable integers for XGBoost."""
    return df[PORT_ID_COLUMN].map(PORT_ID_ENCODING).fillna(-1).astype(int)


def get_feature_matrix(df, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Extract feature matrix from a Gold DataFrame.

    Adds encoded port_id, fills NaN with 0 for count/flag features,
    and forward-fills lag features within each port group.
    """

    cols = FEATURE_COLUMNS + (extra_cols or [])
    # Keep only columns that exist (graceful when new features haven't been added yet)
    available = [c for c in cols if c in df.columns]
    X = df[available].copy()

    # Add port encoding
    X["port_id_enc"] = encode_port_ids(df)

    # Fill NA: count/flag/signal columns → 0, lag/rolling → forward-fill then 0
    lag_cols    = [c for c in available if "lag" in c or "rolling" in c or "z_score" in c or "avg" in c or "std" in c]
    signal_cols = [c for c in available if c not in lag_cols]

    X[signal_cols] = X[signal_cols].fillna(0)
    X[lag_cols]    = X[lag_cols].ffill().fillna(0)

    return X.astype(float)
