"""
USGS Earthquake ingestion client.

Fetches recent earthquakes near each monitored port using the USGS
FDSN Event Web Service.  No API key required.

API docs: https://earthquake.usgs.gov/fdsnws/event/1/
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import structlog

from config import US_PORTS, haversine_km, settings
from src.ingestion.base_client import BaseClient
from src.storage.schemas import EarthquakeEvent, IngestionRun, IngestionStatus

logger = structlog.get_logger(__name__)

USGS_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


class USGSClient(BaseClient):
    """
    Fetches significant earthquakes near monitored ports for the past 7 days.

    Strategy: one query per port with a bounding radius.  The USGS API
    supports radial queries natively (latitude/longitude/maxradiuskm).
    We deduplicate events that appear in multiple port queries.
    """

    BASE_URL         = "https://earthquake.usgs.gov"
    CALLS_PER_SECOND = 1.0
    TIMEOUT_SECONDS  = 20.0

    def fetch_earthquakes(
        self,
        lookback_days: int = 7,
        min_magnitude: float | None = None,
    ) -> tuple[list[EarthquakeEvent], IngestionRun]:
        """
        Fetch earthquakes near all monitored ports for the last `lookback_days`.

        Returns:
            events: list of EarthquakeEvent
            run:    IngestionRun metadata
        """
        run_id     = uuid.uuid4().hex
        started_at = datetime.utcnow()
        min_mag    = min_magnitude or settings.EARTHQUAKE_MIN_MAGNITUDE
        radius_km  = settings.PORT_HAZARD_RADIUS_KM

        end_time   = datetime.utcnow()
        start_time = end_time - timedelta(days=lookback_days)

        events:  list[EarthquakeEvent] = []
        seen_ids: set[str]             = set()
        errors:   list[str]            = []

        for port in US_PORTS:
            port_events = self._fetch_for_port(
                port_id    = port.id,
                lat        = port.lat,
                lon        = port.lon,
                radius_km  = radius_km,
                start_time = start_time,
                end_time   = end_time,
                min_mag    = min_mag,
            )
            if port_events is None:
                errors.append(port.id)
                continue

            for evt in port_events:
                if evt.usgs_id not in seen_ids:
                    seen_ids.add(evt.usgs_id)
                    events.append(evt)

        status = (
            IngestionStatus.SUCCESS if not errors else
            IngestionStatus.PARTIAL if events      else
            IngestionStatus.FAILED
        )

        run = IngestionRun(
            run_id          = run_id,
            source          = "usgs",
            started_at      = started_at,
            finished_at     = datetime.utcnow(),
            status          = status,
            records_fetched = len(events),
            records_stored  = len(events),
            error_message   = f"Failed ports: {errors}" if errors else None,
            extra           = {
                "lookback_days": lookback_days,
                "min_magnitude": min_mag,
                "radius_km":     radius_km,
            },
        )
        logger.info(
            "usgs_fetch_complete",
            events=len(events),
            ports_queried=len(US_PORTS),
            ports_failed=len(errors),
        )
        return events, run

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fetch_for_port(
        self,
        port_id:    str,
        lat:        float,
        lon:        float,
        radius_km:  float,
        start_time: datetime,
        end_time:   datetime,
        min_mag:    float,
    ) -> list[EarthquakeEvent] | None:
        """Query the USGS event service for one port's area."""
        params = {
            "format":        "geojson",
            "starttime":     start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "endtime":       end_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "minmagnitude":  min_mag,
            "latitude":      lat,
            "longitude":     lon,
            "maxradiuskm":   radius_km,
            "orderby":       "time",
            "limit":         100,         # safety cap per port query
        }
        data = self._get(USGS_BASE_URL, params=params)
        if data is None:
            return None

        events = []
        for feature in data.get("features", []):
            evt = self._parse_feature(feature, port_id, lat, lon)
            if evt:
                events.append(evt)
        return events

    def _parse_feature(
        self,
        feature: dict,
        port_id: str,
        port_lat: float,
        port_lon: float,
    ) -> EarthquakeEvent | None:
        try:
            props   = feature["properties"]
            coords  = feature["geometry"]["coordinates"]  # [lon, lat, depth]
            eq_lon, eq_lat, depth_km = float(coords[0]), float(coords[1]), float(coords[2])

            dist_km = haversine_km(port_lat, port_lon, eq_lat, eq_lon)

            return EarthquakeEvent(
                usgs_id              = feature["id"],
                magnitude            = float(props["mag"]),
                mag_type             = props.get("magType", "unknown"),
                depth_km             = depth_km,
                lat                  = eq_lat,
                lon                  = eq_lon,
                place                = props.get("place", ""),
                occurred_at          = datetime.utcfromtimestamp(props["time"] / 1000),
                nearest_port_id      = port_id,
                distance_to_port_km  = round(dist_km, 2),
                tsunami_flag         = bool(props.get("tsunami", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("usgs_parse_error", error=str(exc))
            return None
