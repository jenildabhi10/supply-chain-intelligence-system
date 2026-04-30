"""
Pydantic v2 data models for every Bronze-layer record type.

These models are the single source of truth for what each ingestion client
must produce. Downstream Silver transforms and feature engineering depend
on these schemas remaining stable.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

# ─── Shared enums ─────────────────────────────────────────────────────────────

class IngestionStatus(str, Enum):
    SUCCESS  = "success"
    PARTIAL  = "partial"    # some records failed, others succeeded
    FAILED   = "failed"
    SKIPPED  = "skipped"    # e.g. API key missing


class RiskTier(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ─── NWS Weather Alerts ───────────────────────────────────────────────────────

class WeatherAlert(BaseModel):
    """One active NWS alert relevant to a monitored port."""

    alert_id:          str
    source:            str = "nws"
    port_id:           str                   # which port this alert was fetched for
    event_type:        str                   # e.g. "Tornado Warning", "High Wind Warning"
    severity:          str                   # "Extreme", "Severe", "Moderate", "Minor"
    certainty:         str                   # "Observed", "Likely", "Possible"
    urgency:           str                   # "Immediate", "Expected", "Future"
    headline:          str
    description:       str
    area_description:  str
    effective:         datetime
    expires:           datetime | None = None
    ingested_at:       datetime = Field(default_factory=datetime.utcnow)

    @field_validator("severity", mode="before")
    @classmethod
    def normalise_severity(cls, v: str) -> str:
        return v.strip().title() if v else "Unknown"


# ─── GDELT Events ─────────────────────────────────────────────────────────────

class GdeltEvent(BaseModel):
    """One GDELT 2.0 event filtered near a monitored port or supply-chain topic."""

    global_event_id:  int
    source:           str = "gdelt"
    sql_date:         str                   # YYYYMMDD string from GDELT
    event_code:       str                   # CAMEO code, e.g. "145"
    event_root_code:  str                   # CAMEO root, e.g. "14"
    goldstein_scale:  float                 # -10.0 to +10.0 (negative = destabilising)
    num_mentions:     int
    num_articles:     int
    avg_tone:         float
    action_geo_name:  str | None   = None
    action_geo_lat:   float | None = None
    action_geo_lon:   float | None = None
    action_geo_country: str | None = None
    nearest_port_id:  str | None   = None  # assigned during filtering
    distance_to_port_km: float | None = None
    source_url:       str
    ingested_at:      datetime = Field(default_factory=datetime.utcnow)


# ─── BTS Port Metrics ─────────────────────────────────────────────────────────

class PortMetric(BaseModel):
    """One weekly port performance record from BTS."""

    source:                    str = "bts"
    port_id:                   str
    port_name:                 str
    week_ending:               str           # ISO date string YYYY-MM-DD
    median_berthing_time_hrs:  float | None = None
    avg_berthing_time_hrs:     float | None = None
    vessel_call_count:         int | None   = None
    teu_count:                 int | None   = None
    dataset_id:                str           # Socrata resource ID used
    ingested_at:               datetime = Field(default_factory=datetime.utcnow)


# ─── USGS Earthquake Events ───────────────────────────────────────────────────

class EarthquakeEvent(BaseModel):
    """One USGS earthquake event near a monitored port."""

    usgs_id:          str
    source:           str = "usgs"
    magnitude:        float
    mag_type:         str                   # "mw", "mb", etc.
    depth_km:         float
    lat:              float
    lon:              float
    place:            str
    occurred_at:      datetime
    nearest_port_id:  str
    distance_to_port_km: float
    tsunami_flag:     bool = False
    ingested_at:      datetime = Field(default_factory=datetime.utcnow)


# ─── NASA FIRMS Fire Detections ───────────────────────────────────────────────

class FireDetection(BaseModel):
    """One VIIRS fire detection near a monitored port or transport corridor."""

    source:            str = "firms"
    lat:               float
    lon:               float
    brightness:        float               # brightness temperature (Kelvin)
    frp:               float | None = None  # Fire Radiative Power (MW)
    confidence:        str                # "low", "nominal", "high"
    satellite:         str                # "NOAA-20", "Suomi NPP"
    acquired_at:       datetime
    nearest_port_id:   str
    distance_to_port_km: float
    ingested_at:       datetime = Field(default_factory=datetime.utcnow)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalise_confidence(cls, v: str) -> str:
        mapping = {"l": "low", "n": "nominal", "h": "high"}
        return mapping.get(str(v).lower(), str(v).lower())


# ─── Ingestion run metadata ───────────────────────────────────────────────────

class IngestionRun(BaseModel):
    """Metadata row written to DuckDB after each ingestion run."""

    run_id:          str
    source:          str
    started_at:      datetime
    finished_at:     datetime | None = None
    status:          IngestionStatus = IngestionStatus.SUCCESS
    records_fetched: int = 0
    records_stored:  int = 0
    bronze_paths:    list[str] = Field(default_factory=list)
    error_message:   str | None = None
    extra:           dict = Field(default_factory=dict)  # source-specific metadata
