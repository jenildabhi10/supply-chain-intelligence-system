"""
GDELT 2.0 ingestion client.

Downloads the latest 15-minute event export, filters to disruption-relevant
CAMEO codes near monitored ports, and returns structured GdeltEvent records.

GDELT updates every 15 minutes.  We track the last-downloaded file URL
to avoid reprocessing.

Docs: https://www.gdeltproject.org/data.html
"""

from __future__ import annotations

import io
import uuid
import zipfile
from datetime import datetime

import pandas as pd
import structlog

from config import (
    DISRUPTION_ROOT_CODES,
    nearest_port,
    settings,
)
from src.ingestion.base_client import BaseClient
from src.storage.schemas import GdeltEvent, IngestionRun, IngestionStatus

logger = structlog.get_logger(__name__)

GDELT_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# GDELT 2.0 export columns — tab-separated, no header in file
GDELT_COLUMNS = [
    "GlobalEventID", "SQLDATE", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode", "QuadClass",
    "GoldsteinScale", "NumMentions", "NumSources", "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

# Columns we actually need after filtering — drop everything else to save memory
KEEP_COLUMNS = [
    "GlobalEventID", "SQLDATE", "EventCode", "EventBaseCode", "EventRootCode",
    "GoldsteinScale", "NumMentions", "NumArticles", "AvgTone",
    "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_Lat", "ActionGeo_Long",
    "SOURCEURL",
]


class GDELTClient(BaseClient):
    """
    Fetches the latest GDELT 2.0 15-minute events file and extracts
    supply-chain-relevant disruption events near monitored ports.

    Because full GDELT files can be 20–60 MB uncompressed, we:
      1. Download the compressed file (~2–5 MB)
      2. Parse only the columns we need
      3. Filter by CAMEO code AND geographic proximity to ports
      4. Store only the filtered rows (~hundreds, not millions)
    """

    BASE_URL         = "http://data.gdeltproject.org"
    CALLS_PER_SECOND = 0.5   # be polite — 1 call per 2 seconds
    TIMEOUT_SECONDS  = 60.0  # file downloads can be slow

    def __init__(self, last_seen_url: str = "") -> None:
        super().__init__()
        # Track the last URL we processed so we don't redownload
        self._last_url = last_seen_url

    def fetch_events(self) -> tuple[list[GdeltEvent], IngestionRun, str]:
        """
        Download and parse the latest GDELT update file.

        Returns:
            events:       filtered GdeltEvent records
            run:          IngestionRun metadata
            new_file_url: the URL of the file we just processed
                          (pass back to __init__ next time to detect duplicates)
        """
        run_id     = uuid.uuid4().hex
        started_at = datetime.utcnow()

        # Step 1: find the latest file URL
        export_url = self._get_latest_export_url()
        if not export_url:
            return [], self._failed_run(run_id, started_at, "Could not fetch lastupdate.txt"), ""

        # Step 2: skip if we already processed this file
        if export_url == self._last_url:
            logger.info("gdelt_no_new_file", url=export_url)
            run = IngestionRun(
                run_id          = run_id,
                source          = "gdelt",
                started_at      = started_at,
                finished_at     = datetime.utcnow(),
                status          = IngestionStatus.SKIPPED,
                records_fetched = 0,
                records_stored  = 0,
                extra           = {"reason": "no_new_file", "url": export_url},
            )
            return [], run, export_url

        # Step 3: download and parse
        df = self._download_and_parse(export_url)
        if df is None or df.empty:
            return [], self._failed_run(run_id, started_at, f"Parse failed: {export_url}"), export_url

        # Step 4: filter to relevant events
        events = self._filter_and_convert(df)

        self._last_url = export_url

        run = IngestionRun(
            run_id          = run_id,
            source          = "gdelt",
            started_at      = started_at,
            finished_at     = datetime.utcnow(),
            status          = IngestionStatus.SUCCESS,
            records_fetched = len(df),
            records_stored  = len(events),
            extra           = {"file_url": export_url, "total_rows": len(df), "filtered_rows": len(events)},
        )
        logger.info(
            "gdelt_fetch_complete",
            total_rows=len(df),
            filtered_rows=len(events),
            url=export_url,
        )
        return events, run, export_url

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_latest_export_url(self) -> str | None:
        """
        Parse lastupdate.txt — 3 lines, each: size md5 url
        Line 1 is the events export file.
        """
        text = self._get_text(GDELT_LASTUPDATE_URL)
        if not text:
            return None
        try:
            first_line = text.strip().splitlines()[0]
            # format: "12345678 abc123def url"
            parts = first_line.split()
            if len(parts) >= 3:
                return parts[2]
        except Exception as exc:
            logger.error("gdelt_lastupdate_parse_error", error=str(exc))
        return None

    def _download_and_parse(self, url: str) -> pd.DataFrame | None:
        """Download the ZIP, extract the CSV, return a trimmed DataFrame."""
        logger.info("gdelt_downloading", url=url)
        raw = self._get_bytes(url)
        if not raw:
            return None

        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(
                        f,
                        sep="\t",
                        header=None,
                        names=GDELT_COLUMNS,
                        usecols=KEEP_COLUMNS,
                        dtype={
                            "GlobalEventID": "Int64",
                            "EventCode":     str,
                            "EventRootCode": str,
                            "NumMentions":   "Int64",
                            "NumArticles":   "Int64",
                            "GoldsteinScale": float,
                            "AvgTone":       float,
                            "ActionGeo_Lat": float,
                            "ActionGeo_Long": float,
                        },
                        on_bad_lines="skip",
                        low_memory=False,
                    )
            logger.debug("gdelt_raw_rows", rows=len(df))
            return df
        except Exception as exc:
            logger.error("gdelt_parse_error", error=str(exc))
            return None

    def _filter_and_convert(self, df: pd.DataFrame) -> list[GdeltEvent]:
        """
        Apply two-stage filter:
          1. Keep rows whose EventRootCode starts with a disruption prefix
          2. Keep rows whose action geo is within NEWS_PROXIMITY_RADIUS_KM of a port
        Convert surviving rows to GdeltEvent objects.
        """
        # Stage 1: CAMEO root code filter
        df = df.dropna(subset=["EventRootCode"])
        mask_code = df["EventRootCode"].str.startswith(
            tuple(DISRUPTION_ROOT_CODES), na=False
        )
        df = df[mask_code].copy()
        logger.debug("gdelt_after_code_filter", rows=len(df))

        if df.empty:
            return []

        # Stage 2: geographic proximity filter
        df = df.dropna(subset=["ActionGeo_Lat", "ActionGeo_Long"])
        radius = settings.NEWS_PROXIMITY_RADIUS_KM

        events: list[GdeltEvent] = []
        for _, row in df.iterrows():
            lat = float(row["ActionGeo_Lat"])
            lon = float(row["ActionGeo_Long"])

            port, dist_km = nearest_port(lat, lon)
            if dist_km > radius:
                continue

            try:
                events.append(
                    GdeltEvent(
                        global_event_id      = int(row["GlobalEventID"]),
                        sql_date             = str(row["SQLDATE"]),
                        event_code           = str(row["EventCode"] or ""),
                        event_root_code      = str(row["EventRootCode"] or ""),
                        goldstein_scale      = float(row["GoldsteinScale"]) if pd.notna(row["GoldsteinScale"]) else 0.0,
                        num_mentions         = int(row["NumMentions"]) if pd.notna(row["NumMentions"]) else 0,
                        num_articles         = int(row["NumArticles"]) if pd.notna(row["NumArticles"]) else 0,
                        avg_tone             = float(row["AvgTone"]) if pd.notna(row["AvgTone"]) else 0.0,
                        action_geo_name      = str(row["ActionGeo_FullName"]) if pd.notna(row["ActionGeo_FullName"]) else None,
                        action_geo_lat       = lat,
                        action_geo_lon       = lon,
                        action_geo_country   = str(row["ActionGeo_CountryCode"]) if pd.notna(row["ActionGeo_CountryCode"]) else None,
                        nearest_port_id      = port.id,
                        distance_to_port_km  = round(dist_km, 2),
                        source_url           = str(row["SOURCEURL"]) if pd.notna(row["SOURCEURL"]) else "",
                    )
                )
            except Exception as exc:
                logger.warning("gdelt_row_parse_error", error=str(exc))
                continue

        logger.debug("gdelt_after_geo_filter", events=len(events))
        return events

    @staticmethod
    def _failed_run(run_id: str, started_at: datetime, msg: str) -> IngestionRun:
        return IngestionRun(
            run_id        = run_id,
            source        = "gdelt",
            started_at    = started_at,
            finished_at   = datetime.utcnow(),
            status        = IngestionStatus.FAILED,
            error_message = msg,
        )
