"""
Feature engineering pipeline — orchestrates all feature builders and writes Gold.

This module is the entry point for Phase 2.  It:
  1. Reads all Bronze sources via BronzeReader
  2. Runs each feature builder in sequence
  3. Builds a week spine from available BTS data (or falls back to synthetic)
  4. Left-joins all feature DataFrames onto the spine
  5. Fills NAs appropriately per column type
  6. Appends data completeness tracking columns
  7. Writes the result as Gold/port_week_features.parquet

The Gold table schema is the contract between Phase 2 (features) and Phase 3 (ML).
Do not rename columns without updating the model training code.

Run directly:
    python -m src.features.pipeline

Or call from the scheduler after each BTS ingestion:
    from src.features.pipeline import run_pipeline
    run_pipeline(lake)
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import structlog

from config import US_PORTS, settings
from src.features.bronze_reader import BronzeReader
from src.features.hazard_features import compute_earthquake_features, compute_fire_features
from src.features.news_features import compute_news_features
from src.features.port_features import compute_port_features
from src.features.utils import (
    build_port_week_spine,
    compute_data_completeness,
    week_ending_sunday,
)
from src.features.weather_features import compute_weather_features
from src.storage.lake import DataLake

log = structlog.get_logger(__name__)

# ─── Gold table column order ──────────────────────────────────────────────────
# This list defines the canonical column order for the Gold feature table.
# Any column not produced by a feature builder is filled with its default.

GOLD_COLUMNS_ORDERED = [
    # Keys
    "port_id", "week_ending",

    # Port activity (BTS) — primary features + label
    "median_berthing_time_hrs", "avg_berthing_time_hrs", "vessel_call_count",
    "berthing_time_lag_1w", "berthing_time_lag_2w", "berthing_time_lag_4w",
    "berthing_time_pct_change_1w", "call_count_pct_change_1w",
    "berthing_time_4w_avg", "berthing_time_4w_std",
    "berthing_time_52w_avg", "berthing_time_52w_std",
    "berthing_time_z_score_52w", "berthing_time_p75_52w",

    # Calendar
    "week_of_year", "month", "is_q4",

    # Weather (NWS)
    "alert_count", "extreme_alert_count", "severe_alert_count",
    "has_hurricane_alert", "has_high_wind_alert", "has_flood_alert", "has_fog_alert",
    "max_severity_score",

    # News (GDELT)
    "gdelt_event_count", "gdelt_strike_count", "gdelt_coerce_count", "gdelt_conflict_count",
    "gdelt_goldstein_avg", "gdelt_goldstein_min", "gdelt_avg_tone",
    "gdelt_num_articles_total", "gdelt_num_mentions_total", "gdelt_weighted_disruption",

    # Seismic (USGS)
    "eq_count", "eq_max_magnitude", "eq_energy_proxy", "eq_tsunami_flag", "eq_depth_min_km",

    # Fire (FIRMS)
    "fire_count", "fire_high_confidence_count", "fire_max_frp", "fire_total_frp",

    # Data quality metadata
    "data_completeness_score",   # 0.0–1.0: fraction of sources that contributed

    # Label (target variable for XGBoost)
    "is_disruption_week",        # 1 = disruption, 0 = normal, NaN = baseline not yet available
]

# Columns to fill with 0 when NaN (event-count features: no data = no events)
_ZERO_FILL_COLS = [
    "alert_count", "extreme_alert_count", "severe_alert_count",
    "has_hurricane_alert", "has_high_wind_alert", "has_flood_alert", "has_fog_alert",
    "max_severity_score",
    "gdelt_event_count", "gdelt_strike_count", "gdelt_coerce_count", "gdelt_conflict_count",
    "gdelt_goldstein_avg", "gdelt_goldstein_min", "gdelt_avg_tone",
    "gdelt_num_articles_total", "gdelt_num_mentions_total", "gdelt_weighted_disruption",
    "eq_count", "eq_max_magnitude", "eq_energy_proxy", "eq_tsunami_flag",
    "fire_count", "fire_high_confidence_count", "fire_max_frp", "fire_total_frp",
]

# Source indicator columns used for completeness scoring
# Value is the column that is non-null when that source contributed
_SOURCE_INDICATORS = {
    "bts":   "median_berthing_time_hrs",
    "nws":   "alert_count",
    "gdelt": "gdelt_event_count",
    "usgs":  "eq_count",
    "firms": "fire_count",
}


# ─── Main pipeline function ───────────────────────────────────────────────────

def run_pipeline(lake: DataLake | None = None) -> pd.DataFrame:
    """
    Execute the full feature engineering pipeline and write to Gold.

    Returns the final Gold DataFrame regardless of write success.
    """
    started = datetime.utcnow()
    reader  = BronzeReader()

    # ── Step 1: check data availability ──────────────────────────────────────
    availability = reader.data_availability()
    log.info("pipeline_start", availability=availability)

    if all(v == 0 for v in availability.values()):
        log.warning("pipeline_no_data", msg="All Bronze sources empty — run the scheduler first")
        return pd.DataFrame()

    # ── Step 2: run all feature builders ─────────────────────────────────────
    weather_df = compute_weather_features(reader.read_nws())
    news_df    = compute_news_features(reader.read_gdelt())
    port_df    = compute_port_features(reader.read_bts())
    eq_df      = compute_earthquake_features(reader.read_usgs())
    fire_df    = compute_fire_features(reader.read_firms())

    log.info(
        "feature_builders_complete",
        weather_rows = len(weather_df),
        news_rows    = len(news_df),
        port_rows    = len(port_df),
        eq_rows      = len(eq_df),
        fire_rows    = len(fire_df),
    )

    # ── Step 3: build the week spine ─────────────────────────────────────────
    spine = _build_spine(port_df)
    if spine.empty:
        log.warning("pipeline_no_spine", msg="No spine built — BTS data required")
        return pd.DataFrame()

    log.info("spine_built", rows=len(spine), ports=spine["port_id"].nunique())

    # ── Step 4: join all features onto spine ─────────────────────────────────
    join_key = ["port_id", "week_ending"]
    gold = spine.copy()

    for name, df in [
        ("port",    port_df),
        ("weather", weather_df),
        ("news",    news_df),
        ("eq",      eq_df),
        ("fire",    fire_df),
    ]:
        if not df.empty and join_key[0] in df.columns and join_key[1] in df.columns:
            df["week_ending"] = pd.to_datetime(df["week_ending"])
            gold = gold.merge(df, on=join_key, how="left", suffixes=("", f"_{name}"))
            log.debug("joined", source=name, gold_rows=len(gold))

    # ── Step 5: NA filling ────────────────────────────────────────────────────
    for col in _ZERO_FILL_COLS:
        if col in gold.columns:
            gold[col] = pd.to_numeric(gold[col], errors="coerce").fillna(0)

    # ── Step 6: data completeness score ──────────────────────────────────────
    gold["data_completeness_score"] = compute_data_completeness(gold, _SOURCE_INDICATORS)

    # ── Step 7: enforce column order, add missing columns ────────────────────
    for col in GOLD_COLUMNS_ORDERED:
        if col not in gold.columns:
            gold[col] = pd.NA

    gold = gold[GOLD_COLUMNS_ORDERED]
    gold = gold.sort_values(["port_id", "week_ending"]).reset_index(drop=True)

    # ── Step 8: write to Gold layer ───────────────────────────────────────────
    if lake:
        lake.write_gold_parquet("port_week_features", gold)
    else:
        _write_gold_direct(gold)

    elapsed = (datetime.utcnow() - started).total_seconds()
    log.info(
        "pipeline_complete",
        rows              = len(gold),
        ports             = gold["port_id"].nunique(),
        disruption_weeks  = int(gold["is_disruption_week"].sum()),
        completeness_avg  = f"{gold['data_completeness_score'].mean():.2f}",
        elapsed_s         = round(elapsed, 2),
    )
    return gold


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_spine(port_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the (port_id, week_ending) spine from BTS data.
    Falls back to a synthetic 104-week spine covering all monitored ports
    if BTS data is unavailable (e.g., during initial setup or testing).
    """
    port_ids = [p.id for p in US_PORTS]

    if not port_df.empty and "week_ending" in port_df.columns:
        # Use the actual date range from BTS data
        we = pd.to_datetime(port_df["week_ending"]).dropna()
        if not we.empty:
            return build_port_week_spine(port_ids, start_date=we.min(), end_date=we.max())

    # Synthetic fallback: last 104 weeks
    log.debug("spine_synthetic_fallback")
    end   = week_ending_sunday(datetime.utcnow())
    start = end - pd.Timedelta(weeks=103)
    return build_port_week_spine(port_ids, start_date=start, end_date=end)


def _write_gold_direct(df: pd.DataFrame) -> None:
    """Write Gold without a DataLake instance (standalone pipeline runs)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = settings.gold_path / "port_week_features.parquet"
    settings.gold_path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out, compression="snappy")
    log.info("gold_written_direct", path=str(out), rows=len(df))


# ─── CLI entry point ─────────────────────────────────────────────────────────

def _print_gold_summary(df: pd.DataFrame) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table   = Table(title="Gold Table — port_week_features.parquet", show_lines=True)
    table.add_column("Column",    style="cyan",  no_wrap=True)
    table.add_column("Non-null",  style="green")
    table.add_column("Type",      style="white")
    table.add_column("Sample",    style="yellow")

    for col in df.columns:
        non_null = df[col].notna().sum()
        dtype    = str(df[col].dtype)
        sample   = str(df[col].dropna().iloc[0]) if non_null > 0 else "—"
        table.add_row(col, str(non_null), dtype, sample[:40])

    console.print(table)
    console.print(f"\n[bold]Rows:[/bold] {len(df)}  |  [bold]Ports:[/bold] {df['port_id'].nunique()}")

    if "is_disruption_week" in df.columns:
        label_counts = df["is_disruption_week"].value_counts()
        console.print(f"\n[bold]Disruption label distribution:[/bold]\n{label_counts.to_string()}")


if __name__ == "__main__":
    import logging
    import sys

    import structlog

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    gold = run_pipeline()
    if not gold.empty:
        _print_gold_summary(gold)
    else:
        print("\nNo data yet — run `python scheduler.py` first to populate Bronze.\n")
