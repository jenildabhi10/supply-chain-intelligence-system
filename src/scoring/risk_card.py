"""
RiskCard — the core output object of the entire system.

Every downstream consumer (API, dashboard, LLM agent, alerts) works with RiskCards.
Defining this schema first forces every other component to produce structured,
typed output rather than raw dicts or prose strings.

Design rules:
  - All fields are typed and validated by Pydantic.
  - Every float is in a documented range.
  - Confidence flags explain WHY confidence is low, not just that it is.
  - The card is self-contained: it carries everything needed to explain it.
  - JSON-serialisable via .model_dump(mode="json").
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

# ─── Enums ────────────────────────────────────────────────────────────────────

class RiskTier(str, Enum):
    LOW      = "low"       # score  0–30
    MEDIUM   = "medium"    # score 30–60
    HIGH     = "high"      # score 60–80
    CRITICAL = "critical"  # score 80–100


class SignalSource(str, Enum):
    NWS   = "nws"
    GDELT = "gdelt"
    USGS  = "usgs"
    FIRMS = "firms"
    BTS   = "bts"


class SignalSeverity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ─── Sub-models ───────────────────────────────────────────────────────────────

class ActiveSignal(BaseModel):
    """
    One discrete real-world signal contributing to the risk assessment.
    These become evidence items in the LLM agent's Evidence Pack (Phase 5).
    """
    signal_id:    str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    source:       SignalSource
    signal_type:  str          # "weather_alert", "strike_news", "earthquake", "fire"
    description:  str          # short human-readable description (≤ 200 chars)
    severity:     SignalSeverity = SignalSeverity.LOW
    port_id:      str
    observed_at:  datetime
    evidence_url: str | None = None   # source URL for citation

    @field_validator("description")
    @classmethod
    def truncate_description(cls, v: str) -> str:
        return v[:200] if len(v) > 200 else v


class SHAPDriver(BaseModel):
    """One feature's SHAP contribution — the model's 'reason' for a prediction."""
    feature:     str
    value:       float   # actual feature value in this row
    shap_impact: float   # positive = pushed probability up
    direction:   str     # "increases_risk" or "decreases_risk"


class ModelInfo(BaseModel):
    """Metadata about which model version produced this card."""
    version:           str
    trained_at:        str | None    = None
    cv_pr_auc:         float | None  = None
    n_training_rows:   int | None    = None


# ─── Risk tier helpers ────────────────────────────────────────────────────────

def score_to_tier(score: float) -> RiskTier:
    if score >= 80:
        return RiskTier.CRITICAL
    if score >= 60:
        return RiskTier.HIGH
    if score >= 30:
        return RiskTier.MEDIUM
    return RiskTier.LOW


RECOMMENDED_ACTIONS: dict[RiskTier, list[str]] = {
    RiskTier.LOW: [
        "Monitor normal operations and scheduled shipments.",
        "Review weekly BTS port performance updates.",
        "No immediate contingency action required.",
    ],
    RiskTier.MEDIUM: [
        "Increase safety stock by 10–15% for goods transiting this port.",
        "Review carrier commitments and request status confirmations.",
        "Identify backup routing options to alternative ports.",
        "Brief procurement team on elevated risk level.",
    ],
    RiskTier.HIGH: [
        "Activate contingency supply plan immediately.",
        "Re-route time-critical shipments to alternative ports where feasible.",
        "Engage logistics providers for real-time status updates.",
        "Notify affected business units and key customers of potential delays.",
        "Increase monitoring frequency to daily.",
    ],
    RiskTier.CRITICAL: [
        "Escalate to VP Supply Chain — immediate leadership awareness required.",
        "Authorise emergency sourcing from alternative suppliers.",
        "Suspend new bookings through this port until situation stabilises.",
        "Notify key customers of confirmed or likely delays.",
        "Convene crisis response team within 24 hours.",
    ],
}


# ─── RiskCard ────────────────────────────────────────────────────────────────

class RiskCard(BaseModel):
    """
    The primary output object of the Supply Chain Intelligence System.

    One RiskCard per (port, forecast_horizon).
    Contains everything needed to understand, explain, and act on the risk.
    """

    # ── Identifiers ──────────────────────────────────────────────────────────
    card_id:              str = Field(default_factory=lambda: uuid.uuid4().hex)
    port_id:              str
    port_name:            str
    region:               str

    # ── Risk assessment ───────────────────────────────────────────────────────
    risk_score:           float   # 0–100 (higher = more risk)
    risk_tier:            RiskTier
    p_disruption:         float   # calibrated probability [0, 1]
    anomaly_score:        float   # IsolationForest score [0, 1]
    impact_index:         float   # port trade-exposure weight [0, 1]

    # ── Confidence ────────────────────────────────────────────────────────────
    confidence_score:     float         # 0–100
    confidence_flags:     list[str] = Field(default_factory=list)
    data_completeness:    float         # 0–1 fraction of sources contributing

    # ── Forecast ─────────────────────────────────────────────────────────────
    forecast_horizon_days: int
    features_week_ending:  str | None = None   # ISO date of features used

    # ── Explanation ──────────────────────────────────────────────────────────
    top_shap_drivers:     list[SHAPDriver]  = Field(default_factory=list)
    active_signals:       list[ActiveSignal] = Field(default_factory=list)

    # ── Actions ───────────────────────────────────────────────────────────────
    recommended_actions:  list[str]  = Field(default_factory=list)

    # ── Metadata ─────────────────────────────────────────────────────────────
    model_info:           ModelInfo | None = None
    generated_at:         datetime = Field(default_factory=datetime.utcnow)

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("risk_score")
    @classmethod
    def clamp_risk_score(cls, v: float) -> float:
        return max(0.0, min(100.0, round(v, 2)))

    @field_validator("confidence_score")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(100.0, round(v, 2)))

    @field_validator("p_disruption", "anomaly_score", "impact_index", "data_completeness")
    @classmethod
    def clamp_unit_interval(cls, v: float) -> float:
        return max(0.0, min(1.0, round(v, 4)))

    # ── Convenience ──────────────────────────────────────────────────────────
    def signal_count_by_source(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.active_signals:
            counts[s.source.value] = counts.get(s.source.value, 0) + 1
        return counts

    def top_driver_names(self) -> list[str]:
        return [d.feature for d in self.top_shap_drivers]

    def is_high_confidence(self) -> bool:
        return self.confidence_score >= 70.0

    def to_summary(self) -> dict:
        """Compact dict for logging and API list endpoints."""
        return {
            "port_id":          self.port_id,
            "port_name":        self.port_name,
            "risk_score":       self.risk_score,
            "risk_tier":        self.risk_tier.value,
            "p_disruption":     self.p_disruption,
            "anomaly_score":    self.anomaly_score,
            "confidence_score": self.confidence_score,
            "n_signals":        len(self.active_signals),
            "top_driver":       self.top_shap_drivers[0].feature if self.top_shap_drivers else None,
            "generated_at":     self.generated_at.isoformat(),
        }
