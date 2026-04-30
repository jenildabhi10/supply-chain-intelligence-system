"""
DataLake — Bronze / Silver / Gold storage layer.

Design principles:
  - Bronze files are IMMUTABLE raw snapshots (never overwrite).
  - Silver files are cleaned, typed, partitioned Parquet (overwritten per-partition).
  - Gold is the model-ready feature table built by the features pipeline.
  - DuckDB is the query engine across all layers plus the metadata/lineage store.
  - Every ingestion run is recorded so we can audit exactly what data fed each model.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

from config import settings
from src.storage.schemas import IngestionRun, IngestionStatus

logger = structlog.get_logger(__name__)


# ─── DDL for the metadata database ────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id          VARCHAR PRIMARY KEY,
    source          VARCHAR NOT NULL,
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    status          VARCHAR NOT NULL DEFAULT 'success',
    records_fetched INTEGER  NOT NULL DEFAULT 0,
    records_stored  INTEGER  NOT NULL DEFAULT 0,
    bronze_paths    VARCHAR,   -- JSON array of file paths
    error_message   VARCHAR,
    extra           VARCHAR    -- JSON blob for source-specific metadata
);

CREATE TABLE IF NOT EXISTS bronze_files (
    file_id      VARCHAR PRIMARY KEY,
    run_id       VARCHAR NOT NULL REFERENCES ingestion_runs(run_id),
    source       VARCHAR NOT NULL,
    file_path    VARCHAR NOT NULL,
    file_size_bytes BIGINT,
    row_count    INTEGER,
    written_at   TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS data_quality_checks (
    check_id     VARCHAR PRIMARY KEY,
    run_id       VARCHAR NOT NULL,
    source       VARCHAR NOT NULL,
    check_name   VARCHAR NOT NULL,
    passed       BOOLEAN NOT NULL,
    details      VARCHAR,
    checked_at   TIMESTAMP NOT NULL
);
"""


class DataLake:
    """
    Thin wrapper around DuckDB + Parquet files for the medallion architecture.

    Usage:
        lake = DataLake()
        # Write raw data to Bronze
        paths = lake.write_bronze("nws", records_list)
        # Register the ingestion run
        lake.record_run(run)
        # Query across layers
        df = lake.query("SELECT * FROM read_parquet('data/silver/nws/*.parquet')")
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or settings.DB_PATH
        self._ensure_dirs()
        self._conn = duckdb.connect(self._db_path)
        self._conn.execute(_DDL)
        self._conn.commit()
        logger.info("data_lake_ready", db=self._db_path)

    # ── Directory bootstrap ────────────────────────────────────────────────────

    def _ensure_dirs(self) -> None:
        for source in ("nws", "gdelt", "bts", "usgs", "firms"):
            (settings.bronze_path / source).mkdir(parents=True, exist_ok=True)
        settings.silver_path.mkdir(parents=True, exist_ok=True)
        settings.gold_path.mkdir(parents=True, exist_ok=True)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Bronze writes ─────────────────────────────────────────────────────────

    def write_bronze_json(
        self,
        source: str,
        records: list[dict[str, Any]],
        run_id: str | None = None,
    ) -> Path:
        """
        Write a list of dicts as a timestamped JSON file to the Bronze layer.
        Returns the file path written.
        """
        if not records:
            logger.debug("bronze_write_skipped", source=source, reason="empty records")
            return Path()

        ts      = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fid     = run_id or uuid.uuid4().hex[:8]
        outfile = settings.bronze_path / source / f"{ts}_{fid}.json"

        payload = {
            "written_at": datetime.utcnow().isoformat(),
            "source":     source,
            "run_id":     fid,
            "count":      len(records),
            "records":    records,
        }
        outfile.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")

        logger.info("bronze_written", source=source, path=str(outfile), count=len(records))
        return outfile

    def write_bronze_parquet(
        self,
        source: str,
        df: pd.DataFrame,
        run_id: str | None = None,
    ) -> Path:
        """Write a DataFrame as a timestamped Parquet file to the Bronze layer."""
        if df.empty:
            logger.debug("bronze_write_skipped", source=source, reason="empty dataframe")
            return Path()

        ts      = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fid     = run_id or uuid.uuid4().hex[:8]
        outfile = settings.bronze_path / source / f"{ts}_{fid}.parquet"

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, outfile, compression="snappy")

        logger.info(
            "bronze_written",
            source=source,
            path=str(outfile),
            rows=len(df),
            cols=list(df.columns),
        )
        return outfile

    # ── Silver writes ─────────────────────────────────────────────────────────

    def write_silver_parquet(
        self,
        source: str,
        df: pd.DataFrame,
        partition_date: str | None = None,
    ) -> Path:
        """
        Write cleaned DataFrame to Silver layer.
        Partition by date if provided (format: YYYY-MM-DD).
        Overwrites the partition (idempotent Silver).
        """
        if df.empty:
            return Path()

        partition = partition_date or datetime.utcnow().strftime("%Y-%m-%d")
        out_dir   = settings.silver_path / source / f"date={partition}"
        out_dir.mkdir(parents=True, exist_ok=True)
        outfile   = out_dir / "data.parquet"

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, outfile, compression="snappy")
        logger.info("silver_written", source=source, partition=partition, rows=len(df))
        return outfile

    # ── Gold writes ────────────────────────────────────────────────────────────

    def write_gold_parquet(self, name: str, df: pd.DataFrame) -> Path:
        """Write the final model-ready feature table to Gold layer."""
        if df.empty:
            return Path()
        outfile = settings.gold_path / f"{name}.parquet"
        table   = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, outfile, compression="snappy")
        logger.info("gold_written", name=name, rows=len(df))
        return outfile

    # ── Ingestion run tracking ─────────────────────────────────────────────────

    def record_run(self, run: IngestionRun) -> None:
        """Upsert an IngestionRun record into the metadata database."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO ingestion_runs
                (run_id, source, started_at, finished_at, status,
                 records_fetched, records_stored, bronze_paths, error_message, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run.run_id,
                run.source,
                run.started_at,
                run.finished_at,
                run.status.value,
                run.records_fetched,
                run.records_stored,
                json.dumps(run.bronze_paths),
                run.error_message,
                json.dumps(run.extra, default=str),
            ],
        )
        self._conn.commit()
        logger.debug("run_recorded", run_id=run.run_id, status=run.status.value)

    def get_last_run(self, source: str) -> IngestionRun | None:
        """Return the most recent IngestionRun for a given source."""
        row = self._conn.execute(
            """
            SELECT run_id, source, started_at, finished_at, status,
                   records_fetched, records_stored, bronze_paths, error_message, extra
            FROM   ingestion_runs
            WHERE  source = ?
            ORDER  BY started_at DESC
            LIMIT  1
            """,
            [source],
        ).fetchone()

        if row is None:
            return None

        return IngestionRun(
            run_id          = row[0],
            source          = row[1],
            started_at      = row[2],
            finished_at     = row[3],
            status          = IngestionStatus(row[4]),
            records_fetched = row[5],
            records_stored  = row[6],
            bronze_paths    = json.loads(row[7]) if row[7] else [],
            error_message   = row[8],
            extra           = json.loads(row[9]) if row[9] else {},
        )

    def get_recent_runs(self, limit: int = 20) -> pd.DataFrame:
        """Return a summary DataFrame of recent ingestion runs (for monitoring)."""
        return self._conn.execute(
            """
            SELECT source, status, started_at, records_fetched, records_stored, error_message
            FROM   ingestion_runs
            ORDER  BY started_at DESC
            LIMIT  ?
            """,
            [limit],
        ).df()

    # ── Generic query interface ────────────────────────────────────────────────

    def query(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        """Run any DuckDB SQL and return a DataFrame. DuckDB can read Parquet directly."""
        return self._conn.execute(sql, params or []).df()

    def bronze_file_count(self, source: str) -> int:
        result = self._conn.execute(
            "SELECT COUNT(*) FROM bronze_files WHERE source = ?", [source]
        ).fetchone()
        return result[0] if result else 0

    # ── Context manager ────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> DataLake:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
