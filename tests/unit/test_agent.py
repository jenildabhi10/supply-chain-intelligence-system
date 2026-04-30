"""
Unit tests for Phase 5 — LLM Agent.

Coverage:
  - tools.py       : list_ports, get_risk_card, get_active_signals,
                     get_top_risk_ports, get_all_port_scores, dispatch
  - evidence_pack.py : build_evidence_pack, EvidencePack helpers
  - agent.py       : SupplyChainAgent fallback mode, AgentResponse structure
"""

from __future__ import annotations

import json

# ─── tools.py ─────────────────────────────────────────────────────────────────
from src.agent.tools import (
    TOOL_SCHEMAS,
    dispatch,
    get_active_signals,
    get_all_port_scores,
    get_risk_card,
    get_top_risk_ports,
    list_ports,
)


class TestListPorts:
    def test_returns_dict_with_ports_key(self):
        result = list_ports()
        assert "ports" in result

    def test_returns_all_10_ports(self):
        result = list_ports()
        assert len(result["ports"]) == 10

    def test_each_port_has_required_keys(self):
        for port in list_ports()["ports"]:
            assert "port_id" in port
            assert "name" in port
            assert "region" in port
            assert "state" in port

    def test_la_lb_present(self):
        ids = [p["port_id"] for p in list_ports()["ports"]]
        assert "la_lb" in ids

    def test_all_expected_ports_present(self):
        ids = {p["port_id"] for p in list_ports()["ports"]}
        expected = {
            "la_lb", "ny_nj", "savannah", "seattle", "houston",
            "charleston", "norfolk", "oakland", "miami", "baltimore",
        }
        assert expected == ids


class TestGetRiskCard:
    def test_valid_port_returns_dict(self):
        result = get_risk_card("la_lb")
        assert isinstance(result, dict)

    def test_valid_port_no_error_key(self):
        result = get_risk_card("la_lb")
        assert "error" not in result

    def test_valid_port_has_risk_score(self):
        result = get_risk_card("la_lb")
        assert "risk_score" in result
        assert 0.0 <= result["risk_score"] <= 100.0

    def test_valid_port_has_risk_tier(self):
        result = get_risk_card("la_lb")
        assert result["risk_tier"] in ("low", "medium", "high", "critical")

    def test_valid_port_has_confidence_score(self):
        result = get_risk_card("la_lb")
        assert "confidence_score" in result
        assert 0.0 <= result["confidence_score"] <= 100.0

    def test_unknown_port_returns_error(self):
        result = get_risk_card("not_a_real_port")
        assert "error" in result

    def test_unknown_port_error_message_helpful(self):
        result = get_risk_card("xyz")
        assert "xyz" in result["error"]

    def test_valid_port_has_port_name(self):
        result = get_risk_card("ny_nj")
        assert result.get("port_name") == "New York / New Jersey"

    def test_all_10_ports_score_without_error(self):
        ports = [p["port_id"] for p in list_ports()["ports"]]
        for pid in ports:
            result = get_risk_card(pid)
            assert "error" not in result, f"{pid} returned error: {result['error']}"


class TestGetActiveSignals:
    def test_valid_port_returns_dict(self):
        result = get_active_signals("la_lb")
        assert isinstance(result, dict)

    def test_valid_port_has_signals_key(self):
        result = get_active_signals("la_lb")
        assert "signals" in result

    def test_valid_port_has_signal_count(self):
        result = get_active_signals("la_lb")
        assert "signal_count" in result
        assert result["signal_count"] == len(result["signals"])

    def test_valid_port_lookback_reflected(self):
        result = get_active_signals("houston", lookback_days=14)
        assert result["lookback_days"] == 14

    def test_unknown_port_returns_error(self):
        result = get_active_signals("fake_port")
        assert "error" in result

    def test_signals_is_list(self):
        result = get_active_signals("savannah")
        assert isinstance(result["signals"], list)


class TestGetTopRiskPorts:
    def test_returns_dict_with_top_ports(self):
        result = get_top_risk_ports(3)
        assert "top_ports" in result

    def test_returns_correct_n(self):
        result = get_top_risk_ports(3)
        assert len(result["top_ports"]) == 3

    def test_n_1_returns_one_port(self):
        result = get_top_risk_ports(1)
        assert len(result["top_ports"]) == 1

    def test_n_clamped_at_10(self):
        result = get_top_risk_ports(99)
        assert len(result["top_ports"]) <= 10

    def test_n_clamped_at_1(self):
        result = get_top_risk_ports(0)
        assert len(result["top_ports"]) >= 1

    def test_ports_sorted_by_risk_desc(self):
        result = get_top_risk_ports(10)
        scores = [p["risk_score"] for p in result["top_ports"]]
        assert scores == sorted(scores, reverse=True)


class TestGetAllPortScores:
    def test_returns_dict(self):
        result = get_all_port_scores()
        assert isinstance(result, dict)

    def test_has_port_summaries(self):
        result = get_all_port_scores()
        assert "port_summaries" in result

    def test_total_ports_is_10(self):
        result = get_all_port_scores()
        assert result["total_ports"] == 10

    def test_summaries_sorted_by_risk_desc(self):
        result = get_all_port_scores()
        scores = [s["risk_score"] for s in result["port_summaries"]]
        assert scores == sorted(scores, reverse=True)

    def test_each_summary_has_required_keys(self):
        result = get_all_port_scores()
        required = {"port_id", "port_name", "risk_score", "risk_tier", "confidence_score"}
        for s in result["port_summaries"]:
            assert required.issubset(s.keys())


class TestToolSchemas:
    def test_five_schemas_defined(self):
        assert len(TOOL_SCHEMAS) == 5

    def test_all_schemas_have_type_function(self):
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"

    def test_all_schemas_have_name(self):
        names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        expected = {
            "list_ports", "get_risk_card", "get_all_port_scores",
            "get_active_signals", "get_top_risk_ports",
        }
        assert names == expected

    def test_get_risk_card_schema_requires_port_id(self):
        schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "get_risk_card")
        assert "port_id" in schema["function"]["parameters"]["required"]


class TestDispatch:
    def test_list_ports_returns_json_string(self):
        result = dispatch("list_ports", {})
        parsed = json.loads(result)
        assert "ports" in parsed

    def test_get_risk_card_returns_json_string(self):
        result = dispatch("get_risk_card", {"port_id": "la_lb"})
        parsed = json.loads(result)
        assert "risk_score" in parsed

    def test_get_risk_card_unknown_port(self):
        result = dispatch("get_risk_card", {"port_id": "unknown"})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_unknown_tool_returns_error(self):
        result = dispatch("nonexistent_tool", {})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_get_active_signals_dispatch(self):
        result = dispatch("get_active_signals", {"port_id": "miami"})
        parsed = json.loads(result)
        assert "signals" in parsed

    def test_get_top_risk_ports_dispatch(self):
        result = dispatch("get_top_risk_ports", {"n": 2})
        parsed = json.loads(result)
        assert "top_ports" in parsed
        assert len(parsed["top_ports"]) == 2

    def test_get_all_port_scores_dispatch(self):
        result = dispatch("get_all_port_scores", {})
        parsed = json.loads(result)
        assert "port_summaries" in parsed


# ─── evidence_pack.py ─────────────────────────────────────────────────────────

from src.agent.evidence_pack import EvidencePack, build_evidence_pack


class TestBuildEvidencePack:
    def test_returns_evidence_pack(self):
        pack = build_evidence_pack(port_ids=["la_lb"])
        assert isinstance(pack, EvidencePack)

    def test_single_port_pack(self):
        pack = build_evidence_pack(port_ids=["la_lb"])
        assert "la_lb" in pack.query_ports

    def test_unknown_port_ids_filtered(self):
        pack = build_evidence_pack(port_ids=["la_lb", "nonexistent_port"])
        assert "nonexistent_port" not in pack.query_ports

    def test_risk_cards_populated(self):
        pack = build_evidence_pack(port_ids=["la_lb"])
        assert len(pack.risk_cards) == 1

    def test_risk_card_has_risk_score(self):
        pack = build_evidence_pack(port_ids=["ny_nj"])
        assert pack.risk_cards[0]["risk_score"] >= 0.0

    def test_port_metadata_populated(self):
        pack = build_evidence_pack(port_ids=["savannah"])
        assert len(pack.port_metadata) == 1
        assert pack.port_metadata[0]["port_id"] == "savannah"

    def test_port_metadata_has_lat_lon(self):
        pack = build_evidence_pack(port_ids=["houston"])
        meta = pack.port_metadata[0]
        assert "lat" in meta and "lon" in meta

    def test_active_signals_is_list(self):
        pack = build_evidence_pack(port_ids=["seattle"])
        assert isinstance(pack.active_signals, list)

    def test_generated_at_is_string(self):
        pack = build_evidence_pack(port_ids=["la_lb"])
        assert isinstance(pack.generated_at, str)

    def test_all_summary_none_by_default(self):
        pack = build_evidence_pack(port_ids=["la_lb"], include_all_summary=False)
        assert pack.all_port_summary is None

    def test_all_summary_populated_when_requested(self):
        pack = build_evidence_pack(port_ids=["la_lb"], include_all_summary=True)
        assert pack.all_port_summary is not None
        assert len(pack.all_port_summary) == 10

    def test_empty_port_ids_uses_all_ports(self):
        pack = build_evidence_pack(port_ids=None)
        assert len(pack.query_ports) == 10

    def test_multiple_ports(self):
        pack = build_evidence_pack(port_ids=["la_lb", "ny_nj", "houston"])
        assert len(pack.risk_cards) == 3


class TestEvidencePackHelpers:
    def test_to_prompt_context_is_valid_json(self):
        pack = build_evidence_pack(port_ids=["la_lb"])
        context = pack.to_prompt_context()
        parsed = json.loads(context)
        assert "risk_assessments" in parsed

    def test_to_prompt_context_includes_signals(self):
        pack = build_evidence_pack(port_ids=["la_lb"])
        context = pack.to_prompt_context()
        parsed = json.loads(context)
        assert "active_signals" in parsed

    def test_to_prompt_context_compact(self):
        pack = build_evidence_pack(port_ids=["la_lb"])
        context = pack.to_prompt_context()
        assert "\n" not in context  # compact JSON, no newlines

    def test_is_empty_false_when_has_cards(self):
        pack = build_evidence_pack(port_ids=["la_lb"])
        assert not pack.is_empty()

    def test_signal_count_matches(self):
        pack = build_evidence_pack(port_ids=["la_lb"])
        assert pack.signal_count() == len(pack.active_signals)

    def test_highest_risk_score_is_float(self):
        pack = build_evidence_pack(port_ids=["la_lb", "ny_nj"])
        score = pack.highest_risk_score()
        assert score is not None
        assert isinstance(score, float)

    def test_highest_risk_score_none_when_no_cards(self):
        pack = EvidencePack(
            query_ports=[], generated_at="2024-01-01T00:00:00",
            risk_cards=[], active_signals=[], port_metadata=[],
        )
        assert pack.highest_risk_score() is None


# ─── agent.py ────────────────────────────────────────────────────────────────

from src.agent.agent import AgentResponse, SupplyChainAgent


class TestAgentResponse:
    def test_default_llm_used_true(self):
        r = AgentResponse(answer="hello")
        assert r.llm_used is True

    def test_tool_calls_defaults_empty(self):
        r = AgentResponse(answer="hello")
        assert r.tool_calls_made == []

    def test_error_defaults_none(self):
        r = AgentResponse(answer="hello")
        assert r.error is None

    def test_model_defaults_none(self):
        r = AgentResponse(answer="hello")
        assert r.model is None


class TestSupplyChainAgentFallback:
    """Tests that run without a Groq API key (fallback mode)."""

    def _make_agent(self) -> SupplyChainAgent:
        return SupplyChainAgent(api_key="")  # force no-key path

    def test_agent_creates_without_key(self):
        agent = self._make_agent()
        assert agent._client is None

    def test_fallback_returns_agent_response(self):
        agent = self._make_agent()
        resp = agent.chat("What is the risk at LA?")
        assert isinstance(resp, AgentResponse)

    def test_fallback_llm_used_false(self):
        agent = self._make_agent()
        resp = agent.chat("What is the risk at LA?")
        assert resp.llm_used is False

    def test_fallback_answer_is_string(self):
        agent = self._make_agent()
        resp = agent.chat("Any question")
        assert isinstance(resp.answer, str)
        assert len(resp.answer) > 0

    def test_fallback_answer_mentions_risk_summary(self):
        agent = self._make_agent()
        resp = agent.chat("show me port risks")
        # Should contain risk data or a recognisable summary header
        assert "risk" in resp.answer.lower() or "port" in resp.answer.lower()

    def test_fallback_tool_calls_empty(self):
        agent = self._make_agent()
        resp = agent.chat("what's happening?")
        assert resp.tool_calls_made == []

    def test_fallback_includes_port_names(self):
        agent = self._make_agent()
        resp = agent.chat("show risk")
        # Should include at least one port name
        port_names = [
            "Los Angeles", "New York", "Savannah", "Seattle", "Houston",
            "Charleston", "Norfolk", "Oakland", "Miami", "Baltimore",
        ]
        assert any(name in resp.answer for name in port_names)
