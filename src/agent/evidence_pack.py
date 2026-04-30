"""
Evidence Pack — structured snapshot of risk data assembled for an LLM query.

The EvidencePack contains everything the LLM should know before generating
an explanation: risk cards, active signals, and port metadata.

It is serialised to compact JSON for the LLM system prompt.
The LLM is instructed to ONLY cite information present in the pack.

Also useful standalone: Phase 6 API can return the pack as a structured
JSON payload alongside the LLM's prose explanation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import structlog
from pydantic import BaseModel

from config import PORT_BY_ID, US_PORTS
from src.agent.tools import get_active_signals, get_all_port_scores, get_risk_card

log = structlog.get_logger(__name__)


class EvidencePack(BaseModel):
    """
    All evidence gathered for one LLM query.

    Passed to the LLM as context; the LLM must only cite this content.
    """

    query_ports:      list[str]            # port IDs the query is about
    generated_at:     str                  # UTC ISO timestamp
    risk_cards:       list[dict]           # full RiskCard dicts
    active_signals:   list[dict]           # ActiveSignal dicts
    port_metadata:    list[dict]           # name, region, lat/lon per port
    all_port_summary: list[dict] | None = None  # short summaries for every port

    def to_prompt_context(self) -> str:
        """Compact JSON string for insertion into an LLM system prompt."""
        payload: dict = {
            "evidence_generated_at": self.generated_at,
            "risk_assessments":      self.risk_cards,
            "active_signals":        self.active_signals,
            "port_metadata":         self.port_metadata,
        }
        if self.all_port_summary:
            payload["all_ports_overview"] = self.all_port_summary
        return json.dumps(payload, default=str, separators=(",", ":"))

    def is_empty(self) -> bool:
        return not self.risk_cards and not self.active_signals

    def signal_count(self) -> int:
        return len(self.active_signals)

    def highest_risk_score(self) -> float | None:
        if not self.risk_cards:
            return None
        return max(c.get("risk_score", 0.0) for c in self.risk_cards)


def build_evidence_pack(
    port_ids:            list[str] | None = None,
    lookback_days:       int = 7,
    include_all_summary: bool = False,
) -> EvidencePack:
    """
    Assemble an EvidencePack for the given port IDs.

    Args:
        port_ids:            Ports to fetch full detail for. None = all ports.
        lookback_days:       Signal lookback window in days.
        include_all_summary: Also include a short summary for every monitored
                             port (useful for "which ports are worst?" queries).

    Returns:
        Populated EvidencePack. Never raises.
    """
    if port_ids is None:
        port_ids = [p.id for p in US_PORTS]

    # Drop unknown IDs silently
    port_ids = [p for p in port_ids if p in PORT_BY_ID]

    risk_cards:     list[dict] = []
    active_signals: list[dict] = []
    port_metadata:  list[dict] = []

    for pid in port_ids:
        card = get_risk_card(pid)
        if "error" not in card:
            risk_cards.append(card)

        signals_result = get_active_signals(pid, lookback_days=lookback_days)
        if "signals" in signals_result:
            active_signals.extend(signals_result["signals"])

        port = PORT_BY_ID[pid]
        port_metadata.append({
            "port_id": port.id,
            "name":    port.name,
            "region":  port.region,
            "state":   port.state,
            "lat":     port.lat,
            "lon":     port.lon,
        })

    all_port_summary: list[dict] | None = None
    if include_all_summary:
        try:
            result = get_all_port_scores()
            all_port_summary = result.get("port_summaries")
        except Exception as exc:
            log.warning("evidence_pack_all_summary_error", error=str(exc))

    return EvidencePack(
        query_ports      = port_ids,
        generated_at     = datetime.now(UTC).isoformat(),
        risk_cards       = risk_cards,
        active_signals   = active_signals,
        port_metadata    = port_metadata,
        all_port_summary = all_port_summary,
    )
