"""
Supply Chain Intelligence API — FastAPI backend.

Endpoints:
  GET  /health            — liveness check
  GET  /ports             — list all monitored ports
  GET  /risk              — risk summaries for all ports (sorted by score)
  GET  /risk/{port_id}    — full RiskCard for one port
  GET  /alerts/{port_id}  — active signals for a port
  POST /explain           — ask the LLM agent a natural-language question
  POST /simulate          — what-if: override p_disruption / anomaly_score

All endpoints degrade gracefully — no 500s from missing data.
"""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from config import PORT_BY_ID, US_PORTS
from src.agent.agent import AgentResponse, SupplyChainAgent
from src.agent.tools import get_active_signals as _get_active_signals
from src.scoring.scorer import RiskScorer

log = structlog.get_logger(__name__)

# ─── Shared app state ─────────────────────────────────────────────────────────

_scorer: RiskScorer | None = None
_agent:  SupplyChainAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scorer, _agent
    try:
        from src.models.registry import ModelRegistry
        _scorer = RiskScorer(registry=ModelRegistry())
        log.info("api_scorer_loaded")
    except Exception as exc:
        log.warning("api_scorer_fallback", error=str(exc))
        _scorer = RiskScorer()
    _agent = SupplyChainAgent()
    log.info("api_agent_loaded", llm_available=_agent._client is not None)
    yield


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Supply Chain Intelligence API",
    description = "Evidence-grounded port disruption risk assessment for U.S. ports.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)


# ─── Response / request models ────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str = "ok"
    model_loaded: bool
    llm_available: bool
    timestamp:    str


class PortInfo(BaseModel):
    port_id: str
    name:    str
    region:  str
    state:   str
    lat:     float
    lon:     float


class PortListResponse(BaseModel):
    ports: list[PortInfo]
    total: int


class RiskSummary(BaseModel):
    port_id:          str
    port_name:        str
    risk_score:       float
    risk_tier:        str
    p_disruption:     float
    anomaly_score:    float
    confidence_score: float
    n_signals:        int
    top_driver:       str | None
    generated_at:     str


class AllRiskResponse(BaseModel):
    port_summaries: list[RiskSummary]
    total_ports:    int
    generated_at:   str


class AlertsResponse(BaseModel):
    port_id:      str
    port_name:    str
    lookback_days: int
    signal_count: int
    signals:      list[dict]


class ExplainRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class ExplainResponse(BaseModel):
    answer:          str
    tool_calls_made: list[str]
    llm_used:        bool
    model:           str | None
    question:        str


class SimulateRequest(BaseModel):
    port_id:      str
    p_disruption: float = Field(..., ge=0.0, le=1.0,
                                description="Override disruption probability [0, 1]")
    anomaly_score: float = Field(0.5, ge=0.0, le=1.0,
                                 description="Override anomaly score [0, 1]")

    @field_validator("port_id")
    @classmethod
    def validate_port(cls, v: str) -> str:
        if v not in PORT_BY_ID:
            raise ValueError(f"Unknown port_id '{v}'")
        return v


class SimulateResponse(BaseModel):
    port_id:       str
    port_name:     str
    scenario:      dict       # the inputs used
    risk_score:    float
    risk_tier:     str
    delta_score:   float      # simulated − baseline
    baseline_score: float
    recommended_actions: list[str]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _scorer_or_raise() -> RiskScorer:
    if _scorer is None:
        raise HTTPException(503, "Scoring engine not ready — try again shortly.")
    return _scorer


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _compute_score(p: float, a: float, i: float) -> float:
    raw = 0.5 * (p - 0.5) * 4 + 0.3 * (a - 0.5) * 4 + 0.2 * (i - 0.5) * 4
    return round(100.0 * _sigmoid(raw), 2)


def _tier(score: float) -> str:
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 30:
        return "medium"
    return "low"


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    return HealthResponse(
        model_loaded  = _scorer is not None and _scorer._bundle is not None,
        llm_available = _agent is not None and _agent._client is not None,
        timestamp     = datetime.now(UTC).isoformat(),
    )


@app.get("/ports", response_model=PortListResponse, tags=["ports"])
def list_ports():
    ports = [
        PortInfo(
            port_id = p.id,
            name    = p.name,
            region  = p.region,
            state   = p.state,
            lat     = p.lat,
            lon     = p.lon,
        )
        for p in US_PORTS
    ]
    return PortListResponse(ports=ports, total=len(ports))


@app.get("/risk", response_model=AllRiskResponse, tags=["risk"])
def all_risk():
    scorer = _scorer_or_raise()
    try:
        cards = scorer.score_all_ports()
    except Exception as exc:
        log.error("api_all_risk_error", error=str(exc))
        raise HTTPException(500, f"Scoring error: {exc}") from exc

    summaries = [
        RiskSummary(
            port_id          = c.port_id,
            port_name        = c.port_name,
            risk_score       = c.risk_score,
            risk_tier        = c.risk_tier.value,
            p_disruption     = c.p_disruption,
            anomaly_score    = c.anomaly_score,
            confidence_score = c.confidence_score,
            n_signals        = len(c.active_signals),
            top_driver       = c.top_shap_drivers[0].feature if c.top_shap_drivers else None,
            generated_at     = c.generated_at.isoformat(),
        )
        for c in cards
    ]
    return AllRiskResponse(
        port_summaries = summaries,
        total_ports    = len(summaries),
        generated_at   = datetime.now(UTC).isoformat(),
    )


@app.get("/risk/{port_id}", tags=["risk"])
def port_risk(port_id: str):
    if port_id not in PORT_BY_ID:
        raise HTTPException(404, f"Port '{port_id}' not found. GET /ports for valid IDs.")
    scorer = _scorer_or_raise()
    try:
        card = scorer.score(port_id)
        return card.model_dump(mode="json")
    except Exception as exc:
        log.error("api_port_risk_error", port_id=port_id, error=str(exc))
        raise HTTPException(500, f"Scoring error: {exc}") from exc


@app.get("/alerts/{port_id}", response_model=AlertsResponse, tags=["alerts"])
def port_alerts(
    port_id:      str,
    lookback_days: int = Query(default=7, ge=1, le=90),
):
    if port_id not in PORT_BY_ID:
        raise HTTPException(404, f"Port '{port_id}' not found.")
    result = _get_active_signals(port_id=port_id, lookback_days=lookback_days)
    if "error" in result:
        raise HTTPException(500, result["error"])
    return AlertsResponse(
        port_id       = port_id,
        port_name     = PORT_BY_ID[port_id].name,
        lookback_days = lookback_days,
        signal_count  = result["signal_count"],
        signals       = result["signals"],
    )


@app.post("/explain", response_model=ExplainResponse, tags=["agent"])
def explain(body: ExplainRequest):
    if _agent is None:
        raise HTTPException(503, "Agent not ready.")
    try:
        resp: AgentResponse = _agent.chat(body.question)
        return ExplainResponse(
            answer          = resp.answer,
            tool_calls_made = resp.tool_calls_made,
            llm_used        = resp.llm_used,
            model           = resp.model,
            question        = body.question,
        )
    except Exception as exc:
        log.error("api_explain_error", error=str(exc))
        raise HTTPException(500, f"Agent error: {exc}") from exc


@app.post("/simulate", response_model=SimulateResponse, tags=["risk"])
def simulate(body: SimulateRequest):
    scorer = _scorer_or_raise()
    port   = PORT_BY_ID[body.port_id]

    from src.scoring.impact import get_impact_index
    from src.scoring.risk_card import RECOMMENDED_ACTIONS, score_to_tier

    impact  = get_impact_index(body.port_id)
    sim_score = _compute_score(body.p_disruption, body.anomaly_score, impact)
    sim_tier  = score_to_tier(sim_score)

    # Baseline from live scorer
    try:
        baseline_card = scorer.score(body.port_id)
        baseline = baseline_card.risk_score
    except Exception:
        baseline = _compute_score(0.5, 0.5, impact)

    return SimulateResponse(
        port_id    = body.port_id,
        port_name  = port.name,
        scenario   = {
            "p_disruption":  body.p_disruption,
            "anomaly_score": body.anomaly_score,
            "impact_index":  round(impact, 4),
        },
        risk_score          = sim_score,
        risk_tier           = sim_tier.value,
        delta_score         = round(sim_score - baseline, 2),
        baseline_score      = baseline,
        recommended_actions = RECOMMENDED_ACTIONS[sim_tier],
    )
