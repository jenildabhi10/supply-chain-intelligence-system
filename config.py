"""
Central configuration for the Supply Chain Intelligence System.

All settings are driven by environment variables (or .env file).
Port definitions and domain constants live here so every module
imports from a single source of truth.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Port registry ────────────────────────────────────────────────────────────

class PortConfig:
    """Metadata for a monitored port."""

    __slots__ = ("id", "name", "lat", "lon", "state", "region", "nws_zone")

    def __init__(
        self,
        id: str,
        name: str,
        lat: float,
        lon: float,
        state: str,
        region: str,
        nws_zone: str = "",
    ) -> None:
        self.id = id
        self.name = name
        self.lat = lat
        self.lon = lon
        self.state = state
        self.region = region
        self.nws_zone = nws_zone  # NWS forecast zone code (e.g. "CAZ043")

    def __repr__(self) -> str:
        return f"PortConfig(id={self.id!r}, name={self.name!r})"


# U.S. ports tracked in MVP — expandable for global rollout
US_PORTS: Final[list[PortConfig]] = [
    PortConfig("la_lb",      "Los Angeles / Long Beach", 33.75,  -118.25, "CA", "West Coast",  "CAZ043"),
    PortConfig("ny_nj",      "New York / New Jersey",    40.68,   -74.05, "NJ", "East Coast",  "NJZ006"),
    PortConfig("savannah",   "Savannah",                 32.08,   -81.10, "GA", "East Coast",  "GAZ099"),
    PortConfig("seattle",    "Seattle / Tacoma",         47.59,  -122.35, "WA", "West Coast",  "WAZ551"),
    PortConfig("houston",    "Houston",                  29.75,   -95.08, "TX", "Gulf Coast",  "TXZ163"),
    PortConfig("charleston", "Charleston",               32.78,   -79.94, "SC", "East Coast",  "SCZ048"),
    PortConfig("norfolk",    "Norfolk / Hampton Roads",  36.94,   -76.29, "VA", "East Coast",  "VAZ098"),
    PortConfig("oakland",    "Oakland",                  37.79,  -122.27, "CA", "West Coast",  "CAZ007"),
    PortConfig("miami",      "Miami",                    25.77,   -80.19, "FL", "East Coast",  "FLZ072"),
    PortConfig("baltimore",  "Baltimore",                39.27,   -76.61, "MD", "East Coast",  "MDZ012"),
]

PORT_BY_ID: Final[dict[str, PortConfig]] = {p.id: p for p in US_PORTS}


# ─── GDELT CAMEO event codes relevant to supply chain ─────────────────────────

DISRUPTION_CAMEO_CODES: Final[dict[str, str]] = {
    "14":  "Protest",
    "141": "Demonstrate or rally",
    "144": "Hunger strike",
    "145": "Strike or boycott",
    "146": "Obstruct passage / block",
    "172": "Administrative sanctions",
    "173": "Embargo, boycott, or trade sanctions",
    "174": "Halt negotiations",
    "201": "Impose blockade / restrict movement",
    "20":  "Use of conventional military force",
}

# Root CAMEO codes to match events starting with these prefixes
DISRUPTION_ROOT_CODES: Final[frozenset[str]] = frozenset(["14", "17", "20"])


# ─── Geography helpers ────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres between two (lat, lon) points."""
    R = 6_371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def nearest_port(lat: float, lon: float) -> tuple[PortConfig, float]:
    """Return the closest monitored port and distance in km."""
    best = min(US_PORTS, key=lambda p: haversine_km(lat, lon, p.lat, p.lon))
    dist = haversine_km(lat, lon, best.lat, best.lon)
    return best, dist


# ─── Application settings ─────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage paths
    DATA_DIR: str = "data"
    DB_PATH:  str = "data/supply_chain.duckdb"
    LOG_LEVEL: str = "INFO"

    # API keys (all optional — system degrades gracefully when absent)
    NASA_FIRMS_MAP_KEY:   str = ""
    OPENWEATHER_API_KEY:  str = ""
    GROQ_API_KEY:         str = ""
    ANTHROPIC_API_KEY:    str = ""

    # Scheduling
    INGESTION_INTERVAL_MINUTES: int   = 30
    GDELT_POLL_INTERVAL_MINUTES: int  = 15   # GDELT updates every 15 min

    # Ingestion thresholds
    EARTHQUAKE_MIN_MAGNITUDE:  float = 4.5
    PORT_HAZARD_RADIUS_KM:     float = 200.0
    NEWS_PROXIMITY_RADIUS_KM:  float = 500.0

    # ── Derived paths (not from env) ──────────────────────────────────────────
    @property
    def data_path(self)   -> Path: return Path(self.DATA_DIR)

    @property
    def bronze_path(self) -> Path: return self.data_path / "bronze"

    @property
    def silver_path(self) -> Path: return self.data_path / "silver"

    @property
    def gold_path(self)   -> Path: return self.data_path / "gold"


# Singleton — import this everywhere instead of instantiating Settings() again
settings = Settings()
