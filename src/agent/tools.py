"""
Agent tool functions — callable by the LLM agent to query the scoring system.

Each function:
  - Accepts typed Python arguments
  - Returns a JSON-serialisable dict
  - Never raises — returns {"error": "..."} on failure

TOOL_SCHEMAS: Groq/OpenAI-compatible function definitions for tool calling.
dispatch():   routes tool_name → function call, returns JSON string.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from config import PORT_BY_ID, US_PORTS
from src.scoring.scorer import RiskScorer
from src.scoring.signals import assemble_signals

log = structlog.get_logger(__name__)

# Lazy singleton scorer — avoids loading model on import
_scorer: RiskScorer | None = None


def _get_scorer() -> RiskScorer:
    global _scorer
    if _scorer is None:
        try:
            from src.models.registry import ModelRegistry
            _scorer = RiskScorer(registry=ModelRegistry())
        except Exception:
            _scorer = RiskScorer()
    return _scorer


# ─── Tool functions ───────────────────────────────────────────────────────────

def list_ports() -> dict:
    """Return all monitored US ports."""
    return {
        "ports": [
            {"port_id": p.id, "name": p.name, "region": p.region, "state": p.state}
            for p in US_PORTS
        ]
    }


def get_risk_card(port_id: str) -> dict:
    """Score a single port and return its full RiskCard as a dict."""
    if port_id not in PORT_BY_ID:
        return {"error": f"Unknown port_id '{port_id}'. Use list_ports() to see valid IDs."}
    try:
        card = _get_scorer().score(port_id)
        return card.model_dump(mode="json")
    except Exception as exc:
        log.warning("tool_get_risk_card_error", port_id=port_id, error=str(exc))
        return {"error": str(exc)}


def get_all_port_scores() -> dict:
    """Score all monitored ports and return summaries sorted by risk (highest first)."""
    try:
        cards = _get_scorer().score_all_ports()
        return {
            "port_summaries": [c.to_summary() for c in cards],
            "total_ports":    len(cards),
        }
    except Exception as exc:
        log.warning("tool_get_all_port_scores_error", error=str(exc))
        return {"error": str(exc)}


def get_active_signals(port_id: str, lookback_days: int = 7) -> dict:
    """Return active signals (weather, news, seismic, fire) for a port."""
    if port_id not in PORT_BY_ID:
        return {"error": f"Unknown port_id '{port_id}'."}
    try:
        signals = assemble_signals(port_id=port_id, lookback_days=lookback_days)
        return {
            "port_id":      port_id,
            "lookback_days": lookback_days,
            "signal_count": len(signals),
            "signals":      [s.model_dump(mode="json") for s in signals],
        }
    except Exception as exc:
        log.warning("tool_get_active_signals_error", port_id=port_id, error=str(exc))
        return {"error": str(exc)}


def get_top_risk_ports(n: int = 3) -> dict:
    """Return the top N ports by current risk score."""
    n = max(1, min(n, 10))
    try:
        cards = _get_scorer().score_all_ports()
        return {
            "top_ports":   [c.to_summary() for c in cards[:n]],
            "n_requested": n,
        }
    except Exception as exc:
        log.warning("tool_get_top_risk_ports_error", error=str(exc))
        return {"error": str(exc)}


# ─── Groq / OpenAI-compatible tool schemas ────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_ports",
            "description": "List all 10 monitored US ports with their IDs, names, and regions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_card",
            "description": (
                "Score a single US port and return its full risk assessment: "
                "risk score (0–100), risk tier (low/medium/high/critical), "
                "disruption probability, anomaly score, top SHAP feature drivers, "
                "active signals, and recommended actions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "port_id": {
                        "type": "string",
                        "description": "Port identifier, e.g. 'la_lb', 'ny_nj', 'savannah'. Use list_ports() if unsure.",
                    }
                },
                "required": ["port_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_port_scores",
            "description": (
                "Score all 10 monitored ports and return a summary list sorted by risk score "
                "(highest first). Use this for overview queries like 'which ports are most at risk?'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_signals",
            "description": (
                "Return raw active signals (weather alerts, news events, earthquakes, fires) "
                "for a specific port over the past N days. Use when the user wants details "
                "about what is happening at a port."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "port_id": {
                        "type": "string",
                        "description": "Port identifier.",
                    },
                    "lookback_days": {
                        "type": "integer",
                        "description": "Days back to look for signals (1-30). Omit to use default of 7.",
                    },
                },
                "required": ["port_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_risk_ports",
            "description": "Return the top N ports with the highest current risk scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of ports to return (1-10). Omit to use default of 3.",
                    }
                },
                "required": [],
            },
        },
    },
]

# ─── Dispatcher ───────────────────────────────────────────────────────────────

_TOOL_FN_MAP: dict[str, Any] = {
    "list_ports":          lambda args: list_ports(),
    "get_risk_card":       lambda args: get_risk_card(**args),
    "get_all_port_scores": lambda args: get_all_port_scores(),
    "get_active_signals":  lambda args: get_active_signals(**args),
    "get_top_risk_ports":  lambda args: get_top_risk_ports(**args),
}


def dispatch(tool_name: str, tool_args: dict) -> str:
    """Execute a named tool and return its result as a JSON string."""
    fn = _TOOL_FN_MAP.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{tool_name}'"})
    try:
        result = fn(tool_args)
        return json.dumps(result, default=str)
    except Exception as exc:
        log.warning("tool_dispatch_error", tool=tool_name, error=str(exc))
        return json.dumps({"error": str(exc)})
