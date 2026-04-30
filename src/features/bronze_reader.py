"""
BronzeReader — loads raw Bronze files into flat, analysis-ready DataFrames.

Abstracts away the two Bronze storage formats:
  - JSON (NWS, USGS, FIRMS): nested {"records": [...]} wrapper
  - Parquet (GDELT, BTS): DuckDB glob read

All methods return an empty DataFrame (not an exception) when data is absent.
Downstream feature builders treat an empty DataFrame as "source not available."
"""

from __future__ import annotations

import json

import duckdb
import pandas as pd
import structlog

from config import settings

log = structlog.get_logger(__name__)


# ─── JSON Bronze (NWS, USGS, FIRMS) ─────────────────────────────────────────

def _read_json_bronze(source: str) -> pd.DataFrame:
    """
    Load all Bronze JSON files for a source into a single flat DataFrame.

    Each Bronze JSON file has the structure:
        {"written_at": "...", "source": "...", "count": N, "records": [{...}, ...]}

    We extract and concatenate all "records" arrays.
    """
    source_dir = settings.bronze_path / source
    if not source_dir.exists():
        log.debug("bronze_dir_missing", source=source)
        return pd.DataFrame()

    records: list[dict] = []
    for json_file in sorted(source_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            records.extend(data.get("records", []))
        except Exception as exc:
            log.warning("bronze_json_read_error", file=str(json_file), error=str(exc))

    if not records:
        log.debug("bronze_empty", source=source)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    log.debug("bronze_loaded", source=source, rows=len(df), cols=list(df.columns))
    return df


# ─── Parquet Bronze (GDELT, BTS) ─────────────────────────────────────────────

def _read_parquet_bronze(source: str) -> pd.DataFrame:
    """
    Load all Bronze Parquet files for a source via DuckDB glob.
    DuckDB can read multiple Parquet files in a single SQL scan.
    """
    source_dir = settings.bronze_path / source
    if not source_dir.exists():
        log.debug("bronze_dir_missing", source=source)
        return pd.DataFrame()

    parquet_files = list(source_dir.glob("*.parquet"))
    if not parquet_files:
        log.debug("bronze_empty", source=source)
        return pd.DataFrame()

    # Use forward slashes — DuckDB glob works cross-platform with /
    glob_pattern = str(source_dir / "*.parquet").replace("\\", "/")
    try:
        conn = duckdb.connect()
        df   = conn.execute(f"SELECT * FROM read_parquet('{glob_pattern}')").df()
        conn.close()
        log.debug("bronze_loaded", source=source, rows=len(df), cols=list(df.columns))
        return df
    except Exception as exc:
        log.warning("bronze_parquet_read_error", source=source, error=str(exc))
        return pd.DataFrame()


# ─── Public API ───────────────────────────────────────────────────────────────

class BronzeReader:
    """
    Typed accessors for each Bronze source.

    Each method returns a flat DataFrame with column names matching
    the corresponding Pydantic schema in src/storage/schemas.py.
    Datetime columns are converted to pd.Timestamp.
    """

    def read_nws(self) -> pd.DataFrame:
        """Load NWS weather alerts from Bronze JSON files."""
        df = _read_json_bronze("nws")
        if df.empty:
            return df
        # Parse datetime strings
        for col in ("effective", "expires", "ingested_at"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
        return df

    def read_gdelt(self) -> pd.DataFrame:
        """Load GDELT events from Bronze Parquet files."""
        df = _read_parquet_bronze("gdelt")
        if df.empty:
            return df
        if "ingested_at" in df.columns:
            df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce")
        # sql_date may be stored as int or string — normalise to string
        if "sql_date" in df.columns:
            df["sql_date"] = df["sql_date"].astype(str).str.zfill(8)
        return df

    def read_bts(self) -> pd.DataFrame:
        """Load BTS port metrics from Bronze Parquet files."""
        df = _read_parquet_bronze("bts")
        if df.empty:
            return df
        if "ingested_at" in df.columns:
            df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce")
        if "week_ending" in df.columns:
            df["week_ending"] = pd.to_datetime(df["week_ending"], errors="coerce")
        return df

    def read_usgs(self) -> pd.DataFrame:
        """Load USGS earthquake events from Bronze JSON files."""
        df = _read_json_bronze("usgs")
        if df.empty:
            return df
        if "occurred_at" in df.columns:
            df["occurred_at"] = pd.to_datetime(df["occurred_at"], errors="coerce")
        return df

    def read_firms(self) -> pd.DataFrame:
        """Load NASA FIRMS fire detections from Bronze JSON files."""
        df = _read_json_bronze("firms")
        if df.empty:
            return df
        if "acquired_at" in df.columns:
            df["acquired_at"] = pd.to_datetime(df["acquired_at"], errors="coerce")
        return df

    def data_availability(self) -> dict[str, int]:
        """Return row count per source — quick health check."""
        return {
            "nws":   len(self.read_nws()),
            "gdelt": len(self.read_gdelt()),
            "bts":   len(self.read_bts()),
            "usgs":  len(self.read_usgs()),
            "firms": len(self.read_firms()),
        }
