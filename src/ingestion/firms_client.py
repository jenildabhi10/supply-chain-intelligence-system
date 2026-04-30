"""
NASA FIRMS (Fire Information for Resource Management System) ingestion client.

Fetches near-real-time VIIRS fire detections near monitored ports.
Requires a free MAP_KEY from https://firms.modaps.eosdis.nasa.gov/api/

If NASA_FIRMS_MAP_KEY is not set, all methods degrade gracefully and return
a SKIPPED run — the pipeline never breaks on a missing optional key.

API docs: https://firms.modaps.eosdis.nasa.gov/api/
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime

import structlog

from config import US_PORTS, nearest_port, settings
from src.ingestion.base_client import BaseClient
from src.storage.schemas import FireDetection, IngestionRun, IngestionStatus

logger = structlog.get_logger(__name__)

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api"

# VIIRS SNPP Near-Real-Time product — best coverage for recent fires
FIRMS_PRODUCT = "VIIRS_SNPP_NRT"


class FIRMSClient(BaseClient):
    """
    Fetches active fire detections near monitored ports from NASA FIRMS.

    The FIRMS API uses a bounding-box query per area.  We build one bounding
    box that covers all monitored ports (CONUS), then filter by distance
    to individual ports on our side.
    """

    BASE_URL         = FIRMS_BASE_URL
    CALLS_PER_SECOND = 0.5
    TIMEOUT_SECONDS  = 45.0

    def fetch_fires(
        self,
        lookback_days: int = 2,
    ) -> tuple[list[FireDetection], IngestionRun]:
        """
        Fetch recent fire detections near all monitored US ports.

        Returns:
            detections: list of FireDetection
            run:        IngestionRun metadata
        """
        run_id     = uuid.uuid4().hex
        started_at = datetime.utcnow()

        # Degrade gracefully if key is missing
        if not settings.NASA_FIRMS_MAP_KEY:
            logger.warning("firms_skipped", reason="NASA_FIRMS_MAP_KEY not set")
            return [], IngestionRun(
                run_id        = run_id,
                source        = "firms",
                started_at    = started_at,
                finished_at   = datetime.utcnow(),
                status        = IngestionStatus.SKIPPED,
                error_message = "NASA_FIRMS_MAP_KEY not configured",
                extra         = {"hint": "Register free at https://firms.modaps.eosdis.nasa.gov/api/"},
            )

        # Build a bounding box covering all monitored ports
        bbox = self._conus_bbox()
        csv_text = self._fetch_csv(bbox, lookback_days)

        if csv_text is None:
            return [], self._failed_run(run_id, started_at, "FIRMS API call failed")

        detections = self._parse_csv(csv_text)

        run = IngestionRun(
            run_id          = run_id,
            source          = "firms",
            started_at      = started_at,
            finished_at     = datetime.utcnow(),
            status          = IngestionStatus.SUCCESS,
            records_fetched = len(detections),
            records_stored  = len(detections),
            extra           = {
                "product":       FIRMS_PRODUCT,
                "lookback_days": lookback_days,
                "bbox":          bbox,
            },
        )
        logger.info("firms_fetch_complete", detections=len(detections), bbox=bbox)
        return detections, run

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fetch_csv(self, bbox: str, days: int) -> str | None:
        """
        Call FIRMS area API.
        URL: /api/area/csv/{MAP_KEY}/{product}/{area}/{days}
        area format: lon_min,lat_min,lon_max,lat_max
        """
        url = (
            f"{FIRMS_BASE_URL}/area/csv"
            f"/{settings.NASA_FIRMS_MAP_KEY}"
            f"/{FIRMS_PRODUCT}"
            f"/{bbox}"
            f"/{days}"
        )
        return self._get_text(url)

    def _parse_csv(self, csv_text: str) -> list[FireDetection]:
        """
        Parse the FIRMS CSV response and filter by proximity to ports.

        FIRMS CSV columns (VIIRS SNPP NRT):
          latitude, longitude, bright_ti4, scan, track, acq_date, acq_time,
          satellite, instrument, confidence, version, bright_ti5, frp, daynight
        """
        detections: list[FireDetection] = []
        radius_km = settings.PORT_HAZARD_RADIUS_KM

        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])

                port, dist_km = nearest_port(lat, lon)
                if dist_km > radius_km:
                    continue

                # Parse acquisition datetime
                acq_date = row.get("acq_date", "")
                acq_time = str(row.get("acq_time", "0000")).zfill(4)
                acquired_at = datetime.strptime(
                    f"{acq_date} {acq_time[:2]}:{acq_time[2:]}", "%Y-%m-%d %H:%M"
                )

                detections.append(
                    FireDetection(
                        lat                  = lat,
                        lon                  = lon,
                        brightness           = float(row.get("bright_ti4", 0) or 0),
                        frp                  = self._safe_float(row.get("frp")),
                        confidence           = str(row.get("confidence", "n")).lower(),
                        satellite            = row.get("satellite", "unknown"),
                        acquired_at          = acquired_at,
                        nearest_port_id      = port.id,
                        distance_to_port_km  = round(dist_km, 2),
                    )
                )
            except Exception as exc:
                logger.warning("firms_row_parse_error", error=str(exc))
                continue

        return detections

    @staticmethod
    def _conus_bbox() -> str:
        """
        Bounding box covering all monitored U.S. ports (CONUS).
        Format: lon_min,lat_min,lon_max,lat_max
        """
        lats = [p.lat for p in US_PORTS]
        lons = [p.lon for p in US_PORTS]
        # Add padding to capture fires approaching from inland/offshore
        pad = 3.0
        return (
            f"{min(lons) - pad:.2f},{min(lats) - pad:.2f}"
            f",{max(lons) + pad:.2f},{max(lats) + pad:.2f}"
        )

    @staticmethod
    def _safe_float(val: str | None) -> float | None:
        try:
            return float(val) if val else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _failed_run(run_id: str, started_at: datetime, msg: str) -> IngestionRun:
        return IngestionRun(
            run_id        = run_id,
            source        = "firms",
            started_at    = started_at,
            finished_at   = datetime.utcnow(),
            status        = IngestionStatus.FAILED,
            error_message = msg,
        )
