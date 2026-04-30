"""
Port activity feature builder — BTS metrics → (port_id, week_ending) features + labels.

This is the most critical module in Phase 2 because it:
  1. Produces the ground-truth disruption labels used to train XGBoost
  2. Generates lagged and rolling features that carry predictive signal

Output columns (port activity):
  median_berthing_time_hrs        raw BTS measurement (the target variable in regression)
  avg_berthing_time_hrs
  vessel_call_count

Lagged features (predictive):
  berthing_time_lag_1w            same metric, 1 week earlier
  berthing_time_lag_2w            2 weeks earlier
  berthing_time_lag_4w            4 weeks earlier
  berthing_time_pct_change_1w     week-over-week % change
  call_count_pct_change_1w        week-over-week call volume change

Rolling statistics (distributional context):
  berthing_time_4w_avg            4-week moving average
  berthing_time_4w_std
  berthing_time_52w_avg           52-week rolling mean (seasonal baseline)
  berthing_time_52w_std
  berthing_time_z_score_52w       deviation from 52-week normal
  berthing_time_p75_52w           rolling 75th percentile (used for label)

Calendar features:
  week_of_year                    1–52 (captures seasonality)
  month                           1–12
  is_q4                           binary flag for peak season (Oct–Dec)

Label:
  is_disruption_week              1 if median_berthing_time_hrs > rolling 52w 75th percentile
                                  This is the supervised classification target.

Label design rationale:
  Using a port-specific rolling baseline (not a global threshold) means the label
  captures "abnormal for THIS port" rather than "slow overall." A fast port
  (e.g., Savannah) and a congested port (e.g., LA/LB) have very different
  normal distributions — a global threshold would systematically mislabel one of them.
"""

from __future__ import annotations

import pandas as pd
import structlog

from src.features.utils import (
    pct_change_safe,
    rolling_zscore,
    week_ending_sunday,
)

log = structlog.get_logger(__name__)

# Minimum data points needed before we trust the rolling statistics
MIN_PERIODS_SHORT  = 2   # for 4-week rolling
MIN_PERIODS_LONG   = 4   # for 52-week rolling (label computation)


def compute_port_features(bts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw BTS berthing-time records into weekly port-activity features.

    Args:
        bts_df: flat DataFrame from BronzeReader.read_bts()

    Returns:
        DataFrame with columns [port_id, week_ending, <port features>, is_disruption_week]
        Sorted by (port_id, week_ending) ascending.
        Empty if bts_df is empty or lacks required columns.
    """
    required = {"port_id", "week_ending", "median_berthing_time_hrs"}
    if bts_df.empty or not required.issubset(bts_df.columns):
        log.debug("port_features_skipped", reason="empty or missing columns")
        return pd.DataFrame()

    df = bts_df.copy()

    # Ensure week_ending is a Timestamp snapped to Sunday
    df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce").apply(
        lambda d: week_ending_sunday(d) if pd.notna(d) else pd.NaT
    )
    df = df.dropna(subset=["week_ending"])

    # Coerce numeric columns
    df["median_berthing_time_hrs"] = pd.to_numeric(df["median_berthing_time_hrs"], errors="coerce")
    df["avg_berthing_time_hrs"]    = pd.to_numeric(df.get("avg_berthing_time_hrs",    pd.Series(dtype=float)), errors="coerce")
    df["vessel_call_count"]        = pd.to_numeric(df.get("vessel_call_count",        pd.Series(dtype=float)), errors="coerce")

    # Deduplicate: if there are multiple rows for (port, week), keep the latest
    df = (
        df.sort_values("ingested_at", ascending=True)
          .drop_duplicates(subset=["port_id", "week_ending"], keep="last")
          .sort_values(["port_id", "week_ending"])
          .reset_index(drop=True)
    )

    # Compute per-port features using groupby + transform
    df = _add_lag_and_rolling_features(df)
    df = _add_disruption_label(df)
    df = _add_calendar_features(df)

    log.info(
        "port_features_computed",
        rows=len(df),
        ports=df["port_id"].nunique(),
        disruption_rate=f"{df['is_disruption_week'].mean():.2%}" if "is_disruption_week" in df.columns else "n/a",
    )
    return df


# ─── Private helpers ──────────────────────────────────────────────────────────

def _add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged and rolling window features within each port group.
    Must operate per-port because we don't want lag leakage across ports.
    """
    result_frames = []

    for port_id, group in df.groupby("port_id", sort=True):
        g = group.copy().sort_values("week_ending").reset_index(drop=True)
        bt = g["median_berthing_time_hrs"]
        vc = g["vessel_call_count"]

        # Lag features
        g["berthing_time_lag_1w"] = bt.shift(1)
        g["berthing_time_lag_2w"] = bt.shift(2)
        g["berthing_time_lag_4w"] = bt.shift(4)

        # % change
        g["berthing_time_pct_change_1w"] = pct_change_safe(bt, periods=1)
        g["call_count_pct_change_1w"]    = pct_change_safe(vc, periods=1)

        # 4-week rolling
        g["berthing_time_4w_avg"] = bt.rolling(4, min_periods=MIN_PERIODS_SHORT).mean()
        g["berthing_time_4w_std"] = bt.rolling(4, min_periods=MIN_PERIODS_SHORT).std()

        # 52-week rolling
        g["berthing_time_52w_avg"] = bt.rolling(52, min_periods=MIN_PERIODS_LONG).mean()
        g["berthing_time_52w_std"] = bt.rolling(52, min_periods=MIN_PERIODS_LONG).std()

        # Z-score relative to 52-week baseline
        g["berthing_time_z_score_52w"] = rolling_zscore(bt, window=52, min_periods=MIN_PERIODS_LONG)

        result_frames.append(g)

    return pd.concat(result_frames, ignore_index=True)


def _add_disruption_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute is_disruption_week per port using its own rolling 75th percentile.

    For each port, the rolling 75th percentile is computed over a 52-week window.
    A week is labelled as a disruption if its berthing time exceeds that baseline.

    This produces an imbalanced binary label (~20-30% positive rate in normal conditions),
    which is realistic and aligns with PR-AUC as the evaluation metric.
    """
    result_frames = []

    for port_id, group in df.groupby("port_id", sort=True):
        g = group.copy().sort_values("week_ending").reset_index(drop=True)
        bt = g["median_berthing_time_hrs"]

        # Rolling 75th percentile over past 52 weeks (not including current week)
        # Use shift(1) so the label is defined relative to the PAST, preventing leakage
        g["berthing_time_p75_52w"] = (
            bt.shift(1)
              .rolling(52, min_periods=MIN_PERIODS_LONG)
              .quantile(0.75)
        )

        # Label: is this week's berthing time worse than the 75th percentile baseline?
        # Where the baseline is NaN (not enough history), the label must be NaN too —
        # a False label here would be a false negative that corrupts model training.
        p75 = g["berthing_time_p75_52w"]
        g["is_disruption_week"] = pd.array(
            [
                pd.NA if pd.isna(threshold) else int(bt_val > threshold)
                for bt_val, threshold in zip(bt, p75)
            ],
            dtype="Int64",
        )

        result_frames.append(g)

    return pd.concat(result_frames, ignore_index=True)


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add seasonality signals from the week_ending date."""
    we = pd.to_datetime(df["week_ending"])
    df["week_of_year"] = we.dt.isocalendar().week.astype(int)
    df["month"]        = we.dt.month.astype(int)
    df["is_q4"]        = (we.dt.month >= 10).astype(int)  # Oct–Dec peak season
    return df
