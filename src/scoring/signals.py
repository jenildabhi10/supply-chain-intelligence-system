"""
ActiveSignal assembler — reads recent Bronze data and converts it into
the structured ActiveSignal objects that populate RiskCard.active_signals.

Each signal represents one discrete real-world event (a weather alert,
a news event, an earthquake, a fire) that contributed to the risk score.
These become the evidence pack fed to the LLM agent in Phase 5.

Design:
- Reads directly from Bronze JSON / Parquet files (not the Gold feature table).
  The Gold table has already aggregated signals into numeric features; here we
  want the individual raw events so the LLM can cite them.
- Filters to the last `lookback_days` of activity (default 7).
- Returns an empty list when Bronze data is unavailable — never crashes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from src.scoring.risk_card import ActiveSignal, SignalSeverity, SignalSource

log = structlog.get_logger(__name__)

BRONZE_DIR = Path("data/bronze")


# ─── Severity normalisation helpers ──────────────────────────────────────────

_NWS_SEVERITY_MAP: dict[str, SignalSeverity] = {
    "extreme":  SignalSeverity.CRITICAL,
    "severe":   SignalSeverity.HIGH,
    "moderate": SignalSeverity.MEDIUM,
    "minor":    SignalSeverity.LOW,
    "unknown":  SignalSeverity.LOW,
}

_MAGNITUDE_TO_SEVERITY = [
    (7.0, SignalSeverity.CRITICAL),
    (6.0, SignalSeverity.HIGH),
    (5.0, SignalSeverity.MEDIUM),
    (0.0, SignalSeverity.LOW),
]

_GOLDSTEIN_TO_SEVERITY = [
    (-7.0, SignalSeverity.CRITICAL),
    (-4.0, SignalSeverity.HIGH),
    (-1.0, SignalSeverity.MEDIUM),
    (float("-inf"), SignalSeverity.LOW),
]


def _nws_severity(raw: str) -> SignalSeverity:
    return _NWS_SEVERITY_MAP.get(raw.lower(), SignalSeverity.LOW)


def _magnitude_severity(mag: float) -> SignalSeverity:
    for threshold, sev in _MAGNITUDE_TO_SEVERITY:
        if mag >= threshold:
            return sev
    return SignalSeverity.LOW


def _goldstein_severity(score: float) -> SignalSeverity:
    for threshold, sev in _GOLDSTEIN_TO_SEVERITY:
        if score <= threshold:
            return sev
    return SignalSeverity.LOW


# ─── Bronze readers ───────────────────────────────────────────────────────────

def _load_bronze_json(source: str) -> list[dict]:
    """Load all JSON records from a Bronze source directory."""
    source_dir = BRONZE_DIR / source
    if not source_dir.exists():
        return []
    records = []
    for path in sorted(source_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records.extend(data)
            elif isinstance(data, dict):
                records.append(data)
        except Exception as exc:
            log.warning("bronze_read_error", path=str(path), error=str(exc))
    return records


def _cutoff_dt(lookback_days: int) -> datetime:
    return datetime.now(UTC).replace(tzinfo=None) - __import__("datetime").timedelta(days=lookback_days)


# ─── Source-specific assemblers ───────────────────────────────────────────────

def _nws_signals(port_id: str, lookback_days: int) -> list[ActiveSignal]:
    cutoff = _cutoff_dt(lookback_days)
    signals: list[ActiveSignal] = []

    for rec in _load_bronze_json("nws"):
        if rec.get("port_id") != port_id:
            continue
        try:
            effective = datetime.fromisoformat(rec["effective"].replace("Z", "+00:00")).replace(tzinfo=None)
        except (KeyError, ValueError):
            continue
        if effective < cutoff:
            continue

        event_type = rec.get("event_type", "Weather Alert")
        headline   = rec.get("headline", event_type)
        desc       = f"{event_type}: {headline}"

        signals.append(ActiveSignal(
            source       = SignalSource.NWS,
            signal_type  = "weather_alert",
            description  = desc,
            severity     = _nws_severity(rec.get("severity", "unknown")),
            port_id      = port_id,
            observed_at  = effective,
            evidence_url = None,
        ))

    return signals


def _gdelt_signals(port_id: str, lookback_days: int) -> list[ActiveSignal]:
    cutoff = _cutoff_dt(lookback_days)
    signals: list[ActiveSignal] = []

    try:
        import pandas as pd
        gdelt_dir = BRONZE_DIR / "gdelt"
        if not gdelt_dir.exists():
            return []
        parquet_files = sorted(gdelt_dir.glob("*.parquet"))
        if not parquet_files:
            return []
        df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
    except Exception as exc:
        log.warning("gdelt_load_error", error=str(exc))
        return []

    if df.empty or "nearest_port_id" not in df.columns:
        return []

    df = df[df["nearest_port_id"] == port_id].copy()
    if df.empty:
        return []

    for _, row in df.iterrows():
        try:
            sql_date = str(row.get("sql_date", ""))
            observed = datetime.strptime(sql_date[:8], "%Y%m%d") if len(sql_date) >= 8 else None
        except ValueError:
            observed = None
        if observed is None or observed < cutoff:
            continue

        goldstein = float(row.get("goldstein_scale", 0.0))
        event_code = str(row.get("event_code", ""))
        geo_name   = str(row.get("action_geo_name", "")) or "unknown location"
        desc       = f"GDELT event {event_code} near {geo_name} (Goldstein: {goldstein:+.1f})"

        signals.append(ActiveSignal(
            source       = SignalSource.GDELT,
            signal_type  = "supply_chain_news",
            description  = desc,
            severity     = _goldstein_severity(goldstein),
            port_id      = port_id,
            observed_at  = observed,
            evidence_url = str(row.get("source_url", "")) or None,
        ))

    return signals[:20]  # cap at 20 GDELT events per port per window


def _usgs_signals(port_id: str, lookback_days: int) -> list[ActiveSignal]:
    cutoff = _cutoff_dt(lookback_days)
    signals: list[ActiveSignal] = []

    for rec in _load_bronze_json("usgs"):
        if rec.get("nearest_port_id") != port_id:
            continue
        try:
            occurred_at = datetime.fromisoformat(rec["occurred_at"].replace("Z", "+00:00")).replace(tzinfo=None)
        except (KeyError, ValueError):
            continue
        if occurred_at < cutoff:
            continue

        mag   = float(rec.get("magnitude", 0.0))
        place = rec.get("place", "unknown location")
        dist  = rec.get("distance_to_port_km", "?")
        tsunami = rec.get("tsunami_flag", False)
        desc  = f"M{mag:.1f} earthquake near {place} ({dist} km from port)"
        if tsunami:
            desc += " — TSUNAMI WARNING"

        signals.append(ActiveSignal(
            source       = SignalSource.USGS,
            signal_type  = "earthquake",
            description  = desc,
            severity     = _magnitude_severity(mag),
            port_id      = port_id,
            observed_at  = occurred_at,
            evidence_url = None,
        ))

    return signals


def _firms_signals(port_id: str, lookback_days: int) -> list[ActiveSignal]:
    cutoff = _cutoff_dt(lookback_days)
    signals: list[ActiveSignal] = []

    for rec in _load_bronze_json("firms"):
        if rec.get("nearest_port_id") != port_id:
            continue
        try:
            acquired_at = datetime.fromisoformat(rec["acquired_at"].replace("Z", "+00:00")).replace(tzinfo=None)
        except (KeyError, ValueError):
            continue
        if acquired_at < cutoff:
            continue

        lat  = rec.get("lat", "?")
        lon  = rec.get("lon", "?")
        frp  = rec.get("frp")
        dist = rec.get("distance_to_port_km", "?")
        desc = f"Fire detection at ({lat}, {lon}), {dist} km from port"
        if frp:
            desc += f", FRP={frp:.0f} MW"

        conf_str = rec.get("confidence", "nominal")
        if conf_str == "high":
            sev = SignalSeverity.HIGH
        elif conf_str == "nominal":
            sev = SignalSeverity.MEDIUM
        else:
            sev = SignalSeverity.LOW

        signals.append(ActiveSignal(
            source       = SignalSource.FIRMS,
            signal_type  = "fire",
            description  = desc,
            severity     = sev,
            port_id      = port_id,
            observed_at  = acquired_at,
            evidence_url = None,
        ))

    return signals


# ─── Public API ───────────────────────────────────────────────────────────────

def assemble_signals(
    port_id:       str,
    lookback_days: int = 7,
    bronze_dir:    Path | None = None,
) -> list[ActiveSignal]:
    """
    Collect all active signals for a port from the last `lookback_days`.

    Returns an empty list if Bronze data is unavailable — caller decides
    whether to treat that as a confidence flag.
    """
    global BRONZE_DIR
    if bronze_dir is not None:
        original = BRONZE_DIR
        BRONZE_DIR = bronze_dir

    try:
        signals: list[ActiveSignal] = []
        signals.extend(_nws_signals(port_id, lookback_days))
        signals.extend(_gdelt_signals(port_id, lookback_days))
        signals.extend(_usgs_signals(port_id, lookback_days))
        signals.extend(_firms_signals(port_id, lookback_days))
    finally:
        if bronze_dir is not None:
            BRONZE_DIR = original  # type: ignore[possibly-undefined]

    # Sort by severity desc, then recency desc
    _sev_order = {SignalSeverity.CRITICAL: 3, SignalSeverity.HIGH: 2, SignalSeverity.MEDIUM: 1, SignalSeverity.LOW: 0}
    signals.sort(key=lambda s: (_sev_order[s.severity], s.observed_at), reverse=True)
    return signals
