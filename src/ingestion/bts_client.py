"""
BTS (Bureau of Transportation Statistics) ingestion client.

Fetches monthly TEU (container volume) data from the BTS open data portal
(data.bts.gov) using the Socrata API.

Dataset: rd72-aq8r — Monthly TEU Data by Port (Oct 2019–Sep 2022)
  Wide format: one column per port, one row per month.

Since the original weekly berthing-time dataset (y7iz-hhid) was retired,
we derive a berthing-time proxy from TEU volume:
  berthing_hrs = BASE_HOURS[port] * (teu / rolling_median_teu[port])

This preserves real temporal congestion patterns: the 2021-2022 COVID
container surge correctly produces elevated "berthing times" for the model.

Socrata API:  https://dev.socrata.com/docs/endpoints.html
BTS portal:   https://data.bts.gov
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import structlog

from src.ingestion.base_client import BaseClient
from src.storage.schemas import IngestionRun, IngestionStatus, PortMetric

logger = structlog.get_logger(__name__)

BTS_BASE_URL = "https://data.bts.gov"

# Monthly TEU dataset — wide format, one column per port
TEU_DATASET_ID = "rd72-aq8r"

# Map BTS column name → our port_id
_COLUMN_TO_PORT: dict[str, str] = {
    "los_angeles_ca":         "la_lb",
    "port_of_ny_nj":          "ny_nj",
    "savannah_ga":            "savannah",
    "nwsa_seattle_tacoma_wa": "seattle",
    "houston_tx":             "houston",
    "charleston_sc":          "charleston",
    "port_of_virginia_va":    "norfolk",
    "oakland_ca":             "oakland",
    # long_beach_ca folded into la_lb below
    # miami and baltimore not in this dataset
}

# Approximate normal berthing time per port (hours) — used as the proxy baseline
_BASE_HOURS: dict[str, float] = {
    "la_lb":      60.0,
    "ny_nj":      48.0,
    "savannah":   36.0,
    "seattle":    38.0,
    "houston":    42.0,
    "charleston": 30.0,
    "norfolk":    36.0,
    "oakland":    30.0,
    "miami":      28.0,
    "baltimore":  32.0,
}


class BTSClient(BaseClient):
    """
    Fetches monthly port TEU data from BTS and converts to PortMetric records.

    Wide-format TEU rows are melted to long format (one row per port per month).
    Berthing times are derived from TEU volume relative to each port's median.
    """

    BASE_URL         = BTS_BASE_URL
    CALLS_PER_SECOND = 1.0
    TIMEOUT_SECONDS  = 30.0
    FETCH_LIMIT      = 200  # fetch all available months

    def fetch_port_metrics(self) -> tuple[list[PortMetric], IngestionRun]:
        run_id     = uuid.uuid4().hex
        started_at = datetime.utcnow()

        raw_rows = self._fetch_teu_rows()
        if raw_rows is None:
            return [], self._failed_run(run_id, started_at, "TEU dataset fetch failed")

        metrics = self._convert_to_metrics(raw_rows)
        status  = IngestionStatus.SUCCESS if metrics else IngestionStatus.PARTIAL

        run = IngestionRun(
            run_id          = run_id,
            source          = "bts",
            started_at      = started_at,
            finished_at     = datetime.utcnow(),
            status          = status,
            records_fetched = len(raw_rows),
            records_stored  = len(metrics),
            extra           = {"dataset_id": TEU_DATASET_ID},
        )
        logger.info("bts_fetch_complete", fetched=len(raw_rows), stored=len(metrics))
        return metrics, run

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_teu_rows(self) -> list[dict[str, Any]] | None:
        url  = f"{BTS_BASE_URL}/resource/{TEU_DATASET_ID}.json"
        data = self._get(url, params={"$limit": self.FETCH_LIMIT, "$order": "port ASC"})
        if isinstance(data, list):
            return data
        logger.error("bts_teu_fetch_failed", response=str(data)[:200])
        return None

    def _convert_to_metrics(self, rows: list[dict[str, Any]]) -> list[PortMetric]:
        """
        Melt wide-format TEU rows into long-format PortMetric records.

        Also sums LA + Long Beach TEU into the la_lb bucket.
        Derives berthing_hrs proxy = BASE_HOURS * (teu / median_teu_for_port).
        """
        # First pass: collect (port_id, date, teu) tuples
        raw: dict[str, list[tuple[str, float]]] = {}  # port_id → [(date, teu)]

        for row in rows:
            date_str = str(row.get("port", "")).strip()
            month_end = self._parse_month_date(date_str)
            if not month_end:
                continue

            lb_teu = self._safe_float(row.get("long_beach_ca", 0)) or 0.0

            for col, port_id in _COLUMN_TO_PORT.items():
                teu = self._safe_float(row.get(col)) or 0.0
                if col == "los_angeles_ca":
                    teu += lb_teu  # combine LA + LB
                if teu > 0:
                    raw.setdefault(port_id, []).append((month_end, teu))

        # Second pass: compute per-port median TEU → berthing proxy
        metrics: list[PortMetric] = []
        for port_id, observations in raw.items():
            teus       = [t for _, t in observations]
            median_teu = sorted(teus)[len(teus) // 2] if teus else 1.0
            base_hrs   = _BASE_HOURS.get(port_id, 36.0)

            for week_ending, teu in observations:
                berthing_hrs = round(base_hrs * (teu / median_teu), 2)
                metrics.append(PortMetric(
                    port_id                  = port_id,
                    port_name                = port_id.replace("_", " ").title(),
                    week_ending              = week_ending,
                    median_berthing_time_hrs = berthing_hrs,
                    avg_berthing_time_hrs    = berthing_hrs,
                    vessel_call_count        = None,
                    teu_count                = int(teu),
                    dataset_id               = TEU_DATASET_ID,
                ))

        return metrics

    @staticmethod
    def _parse_month_date(raw: str) -> str | None:
        """Parse BTS date string (e.g. '9/1/2022') to 'YYYY-MM-DD'."""
        for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(raw.split("T")[0], fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    @staticmethod
    def _safe_float(val: Any) -> float | None:
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _failed_run(run_id: str, started_at: datetime, msg: str) -> IngestionRun:
        return IngestionRun(
            run_id        = run_id,
            source        = "bts",
            started_at    = started_at,
            finished_at   = datetime.utcnow(),
            status        = IngestionStatus.FAILED,
            error_message = msg,
        )
