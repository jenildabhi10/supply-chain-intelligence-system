"""
Unit tests for Phase 6 — FastAPI backend.

Uses FastAPI's TestClient (synchronous, no running server needed).
The scorer and agent are initialised via the lifespan on test client startup.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


# Single client for all tests — lifespan runs once
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ─── /health ──────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_has_timestamp(self, client):
        data = client.get("/health").json()
        assert "timestamp" in data
        assert "T" in data["timestamp"]  # ISO format

    def test_has_model_loaded_flag(self, client):
        data = client.get("/health").json()
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)

    def test_has_llm_available_flag(self, client):
        data = client.get("/health").json()
        assert "llm_available" in data


# ─── /ports ───────────────────────────────────────────────────────────────────

class TestPorts:
    def test_returns_200(self, client):
        assert client.get("/ports").status_code == 200

    def test_returns_10_ports(self, client):
        data = client.get("/ports").json()
        assert data["total"] == 10
        assert len(data["ports"]) == 10

    def test_each_port_has_required_fields(self, client):
        ports = client.get("/ports").json()["ports"]
        for p in ports:
            assert "port_id" in p
            assert "name" in p
            assert "region" in p
            assert "lat" in p
            assert "lon" in p

    def test_la_lb_present(self, client):
        ids = [p["port_id"] for p in client.get("/ports").json()["ports"]]
        assert "la_lb" in ids


# ─── /risk ────────────────────────────────────────────────────────────────────

class TestAllRisk:
    def test_returns_200(self, client):
        assert client.get("/risk").status_code == 200

    def test_returns_10_summaries(self, client):
        data = client.get("/risk").json()
        assert data["total_ports"] == 10
        assert len(data["port_summaries"]) == 10

    def test_sorted_by_risk_desc(self, client):
        summaries = client.get("/risk").json()["port_summaries"]
        scores = [s["risk_score"] for s in summaries]
        assert scores == sorted(scores, reverse=True)

    def test_each_summary_has_risk_tier(self, client):
        for s in client.get("/risk").json()["port_summaries"]:
            assert s["risk_tier"] in ("low", "medium", "high", "critical")

    def test_has_generated_at(self, client):
        data = client.get("/risk").json()
        assert "generated_at" in data


class TestPortRisk:
    def test_valid_port_returns_200(self, client):
        assert client.get("/risk/la_lb").status_code == 200

    def test_valid_port_has_risk_score(self, client):
        data = client.get("/risk/la_lb").json()
        assert "risk_score" in data
        assert 0.0 <= data["risk_score"] <= 100.0

    def test_valid_port_has_risk_tier(self, client):
        data = client.get("/risk/la_lb").json()
        assert data["risk_tier"] in ("low", "medium", "high", "critical")

    def test_valid_port_has_confidence(self, client):
        data = client.get("/risk/la_lb").json()
        assert "confidence_score" in data

    def test_valid_port_has_active_signals_list(self, client):
        data = client.get("/risk/la_lb").json()
        assert "active_signals" in data
        assert isinstance(data["active_signals"], list)

    def test_valid_port_has_recommended_actions(self, client):
        data = client.get("/risk/la_lb").json()
        assert "recommended_actions" in data
        assert len(data["recommended_actions"]) > 0

    def test_unknown_port_returns_404(self, client):
        assert client.get("/risk/not_a_port").status_code == 404

    def test_all_10_ports_return_200(self, client):
        ports = ["la_lb", "ny_nj", "savannah", "seattle", "houston",
                 "charleston", "norfolk", "oakland", "miami", "baltimore"]
        for pid in ports:
            assert client.get(f"/risk/{pid}").status_code == 200, f"{pid} failed"


# ─── /alerts ─────────────────────────────────────────────────────────────────

class TestAlerts:
    def test_valid_port_returns_200(self, client):
        assert client.get("/alerts/la_lb").status_code == 200

    def test_response_has_signal_count(self, client):
        data = client.get("/alerts/la_lb").json()
        assert "signal_count" in data
        assert data["signal_count"] == len(data["signals"])

    def test_response_has_port_name(self, client):
        data = client.get("/alerts/la_lb").json()
        assert data["port_name"] == "Los Angeles / Long Beach"

    def test_lookback_param_accepted(self, client):
        r = client.get("/alerts/houston?lookback_days=14")
        assert r.status_code == 200
        assert r.json()["lookback_days"] == 14

    def test_lookback_too_large_rejected(self, client):
        assert client.get("/alerts/la_lb?lookback_days=999").status_code == 422

    def test_unknown_port_returns_404(self, client):
        assert client.get("/alerts/fake_port").status_code == 404


# ─── /explain ────────────────────────────────────────────────────────────────

class TestExplain:
    def test_valid_question_returns_200(self, client):
        r = client.post("/explain", json={"question": "What is the risk at LA/LB?"})
        assert r.status_code == 200

    def test_response_has_answer(self, client):
        r = client.post("/explain", json={"question": "What is the risk at LA/LB?"})
        data = r.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_response_has_llm_used_flag(self, client):
        r = client.post("/explain", json={"question": "List all ports."})
        assert "llm_used" in r.json()

    def test_response_echoes_question(self, client):
        q = "Which port has the highest risk?"
        r = client.post("/explain", json={"question": q})
        assert r.json()["question"] == q

    def test_response_has_tool_calls_list(self, client):
        r = client.post("/explain", json={"question": "What is risk at savannah?"})
        assert isinstance(r.json()["tool_calls_made"], list)

    def test_empty_question_rejected(self, client):
        r = client.post("/explain", json={"question": ""})
        assert r.status_code == 422

    def test_too_short_question_rejected(self, client):
        r = client.post("/explain", json={"question": "hi"})
        assert r.status_code == 422


# ─── /simulate ───────────────────────────────────────────────────────────────

class TestSimulate:
    def _sim(self, client, port_id="la_lb", p=0.8, a=0.7):
        return client.post("/simulate", json={
            "port_id":      port_id,
            "p_disruption": p,
            "anomaly_score": a,
        })

    def test_valid_request_returns_200(self, client):
        assert self._sim(client).status_code == 200

    def test_high_p_disruption_raises_score(self, client):
        r = self._sim(client, p=0.9, a=0.5)
        data = r.json()
        assert data["risk_score"] > 50.0

    def test_low_inputs_give_low_score(self, client):
        r = self._sim(client, p=0.1, a=0.1)
        data = r.json()
        assert data["risk_score"] < 50.0

    def test_response_has_delta_score(self, client):
        data = self._sim(client).json()
        assert "delta_score" in data

    def test_response_has_baseline_score(self, client):
        data = self._sim(client).json()
        assert "baseline_score" in data
        assert data["baseline_score"] >= 0.0

    def test_response_has_recommended_actions(self, client):
        data = self._sim(client).json()
        assert "recommended_actions" in data
        assert len(data["recommended_actions"]) > 0

    def test_response_has_scenario(self, client):
        data = self._sim(client, p=0.75, a=0.6).json()
        assert data["scenario"]["p_disruption"] == 0.75
        assert data["scenario"]["anomaly_score"] == 0.6

    def test_unknown_port_returns_422(self, client):
        r = client.post("/simulate", json={
            "port_id": "bad_port", "p_disruption": 0.5, "anomaly_score": 0.5,
        })
        assert r.status_code == 422

    def test_p_out_of_range_rejected(self, client):
        r = client.post("/simulate", json={
            "port_id": "la_lb", "p_disruption": 1.5, "anomaly_score": 0.5,
        })
        assert r.status_code == 422

    def test_risk_score_in_valid_range(self, client):
        data = self._sim(client).json()
        assert 0.0 <= data["risk_score"] <= 100.0

    def test_risk_tier_valid(self, client):
        data = self._sim(client).json()
        assert data["risk_tier"] in ("low", "medium", "high", "critical")
