"""
NWS (National Weather Service) ingestion client.

Fetches active weather alerts for each monitored port's state.
No API key required. NWS explicitly requests caching — we honour TTLs.

API docs: https://www.weather.gov/documentation/services-web-api
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import structlog

from config import US_PORTS, PortConfig
from src.ingestion.base_client import BaseClient
from src.storage.schemas import IngestionRun, IngestionStatus, WeatherAlert

logger = structlog.get_logger(__name__)

# NWS expects a contact email in the User-Agent for caching/monitoring
NWS_BASE_URL = "https://api.weather.gov"


class NWSClient(BaseClient):
    """
    Fetches active NWS alerts for all monitored port states.

    One call per unique state in US_PORTS (currently ≤ 10 states,
    deduplicated to avoid duplicate calls for CA, which has multiple ports).
    """

    BASE_URL         = NWS_BASE_URL
    CALLS_PER_SECOND = 1.0   # NWS requests ≤ 1 req/s per UA
    TIMEOUT_SECONDS  = 20.0

    # Severity values in descending order of impact
    SEVERITY_RANK = {"Extreme": 4, "Severe": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}

    def fetch_alerts(self) -> tuple[list[WeatherAlert], IngestionRun]:
        """
        Fetch active alerts for all port states and return structured records.

        Returns:
            alerts:  list of WeatherAlert objects
            run:     IngestionRun metadata for this call
        """
        run_id     = uuid.uuid4().hex
        started_at = datetime.utcnow()
        alerts:    list[WeatherAlert] = []
        errors:    list[str] = []

        # Deduplicate states — multiple ports share a state (e.g. CA)
        port_by_state: dict[str, list[PortConfig]] = {}
        for port in US_PORTS:
            port_by_state.setdefault(port.state, []).append(port)

        for state, ports in port_by_state.items():
            state_alerts = self._fetch_state_alerts(state, ports)
            if state_alerts is None:
                errors.append(f"state={state}")
                logger.warning("nws_state_failed", state=state)
            else:
                alerts.extend(state_alerts)
                logger.info("nws_state_ok", state=state, alerts=len(state_alerts))

        status = (
            IngestionStatus.SUCCESS  if not errors else
            IngestionStatus.PARTIAL  if alerts      else
            IngestionStatus.FAILED
        )

        run = IngestionRun(
            run_id          = run_id,
            source          = "nws",
            started_at      = started_at,
            finished_at     = datetime.utcnow(),
            status          = status,
            records_fetched = len(alerts),
            records_stored  = len(alerts),
            error_message   = f"Failed states: {errors}" if errors else None,
            extra           = {"states_queried": len(port_by_state), "states_failed": len(errors)},
        )
        return alerts, run

    # ── Private helpers ────────────────────────────────────────────────────────

    def _fetch_state_alerts(
        self,
        state: str,
        ports: list[PortConfig],
    ) -> list[WeatherAlert] | None:
        """
        Fetch active alerts for a single state and map them to port IDs.
        NWS endpoint: GET /alerts/active?area={state_code}
        """
        url  = f"{NWS_BASE_URL}/alerts/active"
        data = self._get(url, params={"area": state})
        if data is None:
            return None

        features = data.get("features", [])
        parsed   = []

        for feature in features:
            alert = self._parse_alert_feature(feature, state, ports)
            if alert:
                parsed.append(alert)

        # Deduplicate by alert_id (NWS can return the same alert for multiple queries)
        seen: set[str] = set()
        unique = []
        for a in parsed:
            if a.alert_id not in seen:
                seen.add(a.alert_id)
                unique.append(a)

        return unique

    def _parse_alert_feature(
        self,
        feature: dict[str, Any],
        state: str,
        ports: list[PortConfig],
    ) -> WeatherAlert | None:
        """Parse a single GeoJSON Feature from the NWS alerts response."""
        try:
            props      = feature.get("properties", {})
            alert_id   = feature.get("id", "").split("/")[-1]  # strip URL prefix
            event_type = props.get("event", "Unknown")
            severity   = props.get("severity", "Unknown")

            # Only keep alerts that actually matter for port operations
            if self.SEVERITY_RANK.get(severity, 0) < 1:
                return None

            # Map to the closest monitored port in this state
            port_id = ports[0].id if len(ports) == 1 else self._best_port(props, ports)

            effective_raw = props.get("effective") or props.get("onset") or ""
            expires_raw   = props.get("expires") or props.get("ends") or ""

            return WeatherAlert(
                alert_id         = alert_id or uuid.uuid4().hex,
                port_id          = port_id,
                event_type       = event_type,
                severity         = severity,
                certainty        = props.get("certainty", "Unknown"),
                urgency          = props.get("urgency", "Unknown"),
                headline         = props.get("headline") or props.get("event", ""),
                description      = (props.get("description") or "")[:1000],
                area_description = props.get("areaDesc", ""),
                effective        = self._parse_dt(effective_raw),
                expires          = self._parse_dt(expires_raw) if expires_raw else None,
            )
        except Exception as exc:
            logger.warning("nws_parse_error", error=str(exc))
            return None

    def _best_port(self, props: dict[str, Any], ports: list[PortConfig]) -> str:
        """
        Pick the most relevant port from a list when a state has multiple ports.
        We use the alert's area description to match city/county names.
        Falls back to the first port if no match found.
        """
        area = (props.get("areaDesc") or "").lower()
        for port in ports:
            if port.name.split("/")[0].strip().lower() in area:
                return port.id
        return ports[0].id

    @staticmethod
    def _parse_dt(raw: str) -> datetime:
        """Parse ISO-8601 datetime strings from NWS (handles timezone suffix)."""
        if not raw:
            return datetime.utcnow()
        # NWS uses "+00:00" suffix; datetime.fromisoformat handles it in Python 3.11+
        try:
            dt = datetime.fromisoformat(raw)
            # Convert to naive UTC
            if dt.tzinfo is not None:
                dt = dt.astimezone(UTC).replace(tzinfo=None)
            return dt
        except ValueError:
            return datetime.utcnow()
