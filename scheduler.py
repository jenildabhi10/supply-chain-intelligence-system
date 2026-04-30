"""
Ingestion scheduler — entry point for the data pipeline.

Runs each ingestion job on a configurable interval using APScheduler.
Each job:
  1. Calls the source client
  2. Serialises records to Bronze (immutable raw files)
  3. Records the run metadata in DuckDB
  4. Logs structured results

Design decisions:
  - Jobs run sequentially within their own thread (no concurrency issues on DuckDB).
  - A failed job logs and continues — it never kills the scheduler.
  - The GDELT job tracks its last-seen file URL in memory to skip duplicates.
  - All job intervals are tunable via .env.
"""

from __future__ import annotations

import logging
import sys

import structlog
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from rich.console import Console
from rich.table import Table

from config import settings
from src.features.pipeline import run_pipeline
from src.ingestion.bts_client import BTSClient
from src.ingestion.firms_client import FIRMSClient
from src.ingestion.gdelt_client import GDELTClient
from src.ingestion.nws_client import NWSClient
from src.ingestion.usgs_client import USGSClient
from src.storage.lake import DataLake

# ─── Logging setup ────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    stream=sys.stdout,
)

log     = structlog.get_logger(__name__)
console = Console()

# ─── Shared state ─────────────────────────────────────────────────────────────

lake          = DataLake()
gdelt_client  = GDELTClient()    # owns last-seen URL state
_gdelt_last_url: str = ""         # thread-safe for single-threaded scheduler


# ─── Job definitions ──────────────────────────────────────────────────────────

def job_nws() -> None:
    """Fetch NWS weather alerts for all port states."""
    log.info("job_start", source="nws")
    try:
        client = NWSClient()
        alerts, run = client.fetch_alerts()
        client.close()

        if alerts:
            path = lake.write_bronze_json(
                source  = "nws",
                records = [a.model_dump(mode="json") for a in alerts],
                run_id  = run.run_id,
            )
            run.bronze_paths = [str(path)] if path.name else []

        lake.record_run(run)
        log.info("job_done", source="nws", status=run.status.value, records=run.records_stored)
    except Exception as exc:
        log.error("job_error", source="nws", error=str(exc))


def job_gdelt() -> None:
    """Fetch latest GDELT 15-minute event update."""
    global _gdelt_last_url
    log.info("job_start", source="gdelt")
    try:
        gdelt_client._last_url = _gdelt_last_url
        events, run, new_url   = gdelt_client.fetch_events()
        _gdelt_last_url        = new_url or _gdelt_last_url

        if events:
            import pandas as pd
            df   = pd.DataFrame([e.model_dump(mode="json") for e in events])
            path = lake.write_bronze_parquet(source="gdelt", df=df, run_id=run.run_id)
            run.bronze_paths = [str(path)] if path.name else []

        lake.record_run(run)
        log.info("job_done", source="gdelt", status=run.status.value, records=run.records_stored)
    except Exception as exc:
        log.error("job_error", source="gdelt", error=str(exc))


def job_bts() -> None:
    """Fetch BTS port performance metrics (weekly berthing times)."""
    log.info("job_start", source="bts")
    try:
        client = BTSClient()
        metrics, run = client.fetch_port_metrics()
        client.close()

        if metrics:
            import pandas as pd
            df   = pd.DataFrame([m.model_dump(mode="json") for m in metrics])
            path = lake.write_bronze_parquet(source="bts", df=df, run_id=run.run_id)
            run.bronze_paths = [str(path)] if path.name else []

        lake.record_run(run)
        log.info("job_done", source="bts", status=run.status.value, records=run.records_stored)

        # BTS is the source of ground-truth labels — rebuild Gold after every successful fetch
        if run.records_stored > 0:
            log.info("feature_pipeline_trigger", reason="new_bts_data")
            try:
                run_pipeline(lake)
            except Exception as pipeline_exc:
                log.error("feature_pipeline_error", error=str(pipeline_exc))
    except Exception as exc:
        log.error("job_error", source="bts", error=str(exc))


def job_usgs() -> None:
    """Fetch USGS earthquake events near monitored ports."""
    log.info("job_start", source="usgs")
    try:
        client = USGSClient()
        events, run = client.fetch_earthquakes()
        client.close()

        if events:
            path = lake.write_bronze_json(
                source  = "usgs",
                records = [e.model_dump(mode="json") for e in events],
                run_id  = run.run_id,
            )
            run.bronze_paths = [str(path)] if path.name else []

        lake.record_run(run)
        log.info("job_done", source="usgs", status=run.status.value, records=run.records_stored)
    except Exception as exc:
        log.error("job_error", source="usgs", error=str(exc))


def job_firms() -> None:
    """Fetch NASA FIRMS fire detections near monitored ports."""
    log.info("job_start", source="firms")
    try:
        client = FIRMSClient()
        fires, run = client.fetch_fires()
        client.close()

        if fires:
            path = lake.write_bronze_json(
                source  = "firms",
                records = [f.model_dump(mode="json") for f in fires],
                run_id  = run.run_id,
            )
            run.bronze_paths = [str(path)] if path.name else []

        lake.record_run(run)
        log.info("job_done", source="firms", status=run.status.value, records=run.records_stored)
    except Exception as exc:
        log.error("job_error", source="firms", error=str(exc))


# ─── Startup summary ──────────────────────────────────────────────────────────

def _print_startup_banner() -> None:
    table = Table(title="Supply Chain Intelligence — Ingestion Scheduler", show_lines=True)
    table.add_column("Job",        style="cyan",  no_wrap=True)
    table.add_column("Source",     style="white")
    table.add_column("Interval",   style="green")
    table.add_column("Status",     style="yellow")

    rows = [
        ("NWS Weather Alerts",  "api.weather.gov",                     f"{settings.INGESTION_INTERVAL_MINUTES}m", "Active"),
        ("GDELT Events",        "data.gdeltproject.org",                f"{settings.GDELT_POLL_INTERVAL_MINUTES}m", "Active"),
        ("BTS Port Metrics",    "data.bts.gov",                        "6h",                                       "Active"),
        ("USGS Earthquakes",    "earthquake.usgs.gov",                  f"{settings.INGESTION_INTERVAL_MINUTES}m", "Active"),
        ("NASA FIRMS Fires",    "firms.modaps.eosdis.nasa.gov",        "1h",
         "Active" if settings.NASA_FIRMS_MAP_KEY else "Skipped (no key)"),
    ]
    for row in rows:
        table.add_row(*row)

    console.print(table)
    console.print(
        f"\n[bold green]Storage:[/bold green] {settings.DB_PATH}  |  "
        f"[bold green]Bronze:[/bold green] {settings.bronze_path}\n"
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    _print_startup_banner()

    # Run all jobs once immediately on startup (warm cache, verify connectivity)
    log.info("startup_run", msg="Running all jobs once on startup...")
    for fn in (job_nws, job_usgs, job_firms, job_bts, job_gdelt):
        fn()

    scheduler = BlockingScheduler(timezone="UTC")
    interval  = settings.INGESTION_INTERVAL_MINUTES

    # NWS + USGS: every N minutes (configurable)
    scheduler.add_job(job_nws,   IntervalTrigger(minutes=interval),   id="nws",   name="NWS")
    scheduler.add_job(job_usgs,  IntervalTrigger(minutes=interval),   id="usgs",  name="USGS")

    # GDELT: every 15 minutes (matches their update cadence)
    scheduler.add_job(
        job_gdelt,
        IntervalTrigger(minutes=settings.GDELT_POLL_INTERVAL_MINUTES),
        id="gdelt",
        name="GDELT",
    )

    # BTS: every 6 hours (weekly stats don't change more often)
    scheduler.add_job(job_bts, IntervalTrigger(hours=6), id="bts", name="BTS")

    # FIRMS: every hour (NRT fire data updates hourly)
    scheduler.add_job(job_firms, IntervalTrigger(hours=1), id="firms", name="FIRMS")

    log.info("scheduler_started", jobs=len(scheduler.get_jobs()))
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler_stopped")
    finally:
        lake.close()


if __name__ == "__main__":
    main()
