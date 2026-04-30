"""
Unit tests for Phase 4 — Risk Scoring Engine.

Coverage:
  - impact.py     : IMPACT_INDEX values, get_impact_index fallback
  - risk_card.py  : RiskCard construction, validators, helpers
  - scorer.py     : _compute_risk_score, _confidence, RiskScorer without model
  - signals.py    : assemble_signals returns list, no crash on empty Bronze dir
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

# ─── impact.py ────────────────────────────────────────────────────────────────
from src.scoring.impact import IMPACT_INDEX, get_impact_index


class TestImpactIndex:
    def test_la_lb_is_max(self):
        assert IMPACT_INDEX["la_lb"] == pytest.approx(1.0, abs=1e-4)

    def test_all_ports_in_range(self):
        for port_id, idx in IMPACT_INDEX.items():
            assert 0.0 < idx <= 1.0, f"{port_id} impact {idx} out of range"

    def test_all_10_ports_present(self):
        expected = {
            "la_lb", "ny_nj", "savannah", "seattle", "houston",
            "charleston", "norfolk", "oakland", "miami", "baltimore",
        }
        assert expected == set(IMPACT_INDEX.keys())

    def test_la_lb_larger_than_ny_nj(self):
        assert IMPACT_INDEX["la_lb"] > IMPACT_INDEX["ny_nj"]

    def test_ny_nj_larger_than_baltimore(self):
        assert IMPACT_INDEX["ny_nj"] > IMPACT_INDEX["baltimore"]

    def test_unknown_port_returns_default(self):
        result = get_impact_index("unknown_port_xyz")
        assert 0.0 < result <= 0.1

    def test_known_port_lookup(self):
        assert get_impact_index("houston") == IMPACT_INDEX["houston"]


# ─── risk_card.py ─────────────────────────────────────────────────────────────

from src.scoring.risk_card import (
    RECOMMENDED_ACTIONS,
    ActiveSignal,
    ModelInfo,
    RiskCard,
    RiskTier,
    SHAPDriver,
    SignalSeverity,
    SignalSource,
    score_to_tier,
)


class TestScoreToTier:
    @pytest.mark.parametrize("score,expected", [
        (0.0,   RiskTier.LOW),
        (15.0,  RiskTier.LOW),
        (29.99, RiskTier.LOW),
        (30.0,  RiskTier.MEDIUM),
        (50.0,  RiskTier.MEDIUM),
        (59.99, RiskTier.MEDIUM),
        (60.0,  RiskTier.HIGH),
        (75.0,  RiskTier.HIGH),
        (79.99, RiskTier.HIGH),
        (80.0,  RiskTier.CRITICAL),
        (100.0, RiskTier.CRITICAL),
    ])
    def test_tier_boundaries(self, score, expected):
        assert score_to_tier(score) == expected


class TestRecommendedActions:
    def test_all_tiers_have_actions(self):
        for tier in RiskTier:
            assert tier in RECOMMENDED_ACTIONS
            assert len(RECOMMENDED_ACTIONS[tier]) >= 2

    def test_critical_has_escalation_action(self):
        actions = RECOMMENDED_ACTIONS[RiskTier.CRITICAL]
        combined = " ".join(actions).lower()
        assert "escalat" in combined or "leadership" in combined or "vp" in combined


class TestActiveSignal:
    def test_description_truncated_at_200(self):
        long_desc = "X" * 300
        sig = ActiveSignal(
            source      = SignalSource.NWS,
            signal_type = "weather_alert",
            description = long_desc,
            port_id     = "la_lb",
            observed_at = datetime(2024, 1, 1),
        )
        assert len(sig.description) == 200

    def test_short_description_unchanged(self):
        sig = ActiveSignal(
            source      = SignalSource.USGS,
            signal_type = "earthquake",
            description = "M5.2 earthquake near Los Angeles",
            port_id     = "la_lb",
            observed_at = datetime(2024, 1, 1),
        )
        assert sig.description == "M5.2 earthquake near Los Angeles"

    def test_default_severity_is_low(self):
        sig = ActiveSignal(
            source      = SignalSource.GDELT,
            signal_type = "supply_chain_news",
            description = "Strike near port",
            port_id     = "ny_nj",
            observed_at = datetime(2024, 1, 1),
        )
        assert sig.severity == SignalSeverity.LOW

    def test_signal_id_auto_generated(self):
        sig = ActiveSignal(
            source      = SignalSource.FIRMS,
            signal_type = "fire",
            description = "Fire detection",
            port_id     = "houston",
            observed_at = datetime(2024, 1, 1),
        )
        assert len(sig.signal_id) == 8


class TestRiskCard:
    def _make_card(self, **kwargs) -> RiskCard:
        defaults = dict(
            port_id               = "la_lb",
            port_name             = "Los Angeles / Long Beach",
            region                = "West Coast",
            risk_score            = 55.0,
            risk_tier             = RiskTier.MEDIUM,
            p_disruption          = 0.45,
            anomaly_score         = 0.6,
            impact_index          = 1.0,
            confidence_score      = 80.0,
            data_completeness     = 0.8,
            forecast_horizon_days = 7,
        )
        defaults.update(kwargs)
        return RiskCard(**defaults)

    def test_basic_construction(self):
        card = self._make_card()
        assert card.port_id == "la_lb"
        assert card.risk_tier == RiskTier.MEDIUM

    def test_risk_score_clamped_above_100(self):
        card = self._make_card(risk_score=150.0)
        assert card.risk_score == 100.0

    def test_risk_score_clamped_below_0(self):
        card = self._make_card(risk_score=-10.0)
        assert card.risk_score == 0.0

    def test_confidence_score_clamped(self):
        card = self._make_card(confidence_score=200.0)
        assert card.confidence_score == 100.0

    def test_p_disruption_clamped(self):
        card = self._make_card(p_disruption=1.5)
        assert card.p_disruption == 1.0

    def test_anomaly_score_clamped(self):
        card = self._make_card(anomaly_score=-0.2)
        assert card.anomaly_score == 0.0

    def test_impact_index_clamped(self):
        card = self._make_card(impact_index=2.0)
        assert card.impact_index == 1.0

    def test_generated_at_is_datetime(self):
        card = self._make_card()
        assert isinstance(card.generated_at, datetime)

    def test_card_id_auto_generated(self):
        card = self._make_card()
        assert len(card.card_id) == 32

    def test_is_high_confidence_true(self):
        card = self._make_card(confidence_score=70.0)
        assert card.is_high_confidence() is True

    def test_is_high_confidence_false(self):
        card = self._make_card(confidence_score=69.9)
        assert card.is_high_confidence() is False

    def test_to_summary_keys(self):
        card = self._make_card()
        summary = card.to_summary()
        for key in ("port_id", "port_name", "risk_score", "risk_tier", "p_disruption",
                    "anomaly_score", "confidence_score", "n_signals", "generated_at"):
            assert key in summary

    def test_signal_count_by_source(self):
        sig1 = ActiveSignal(source=SignalSource.NWS,  signal_type="weather_alert",
                            description="Wind warning", port_id="la_lb",
                            observed_at=datetime(2024, 1, 1))
        sig2 = ActiveSignal(source=SignalSource.NWS,  signal_type="weather_alert",
                            description="Rain warning", port_id="la_lb",
                            observed_at=datetime(2024, 1, 2))
        sig3 = ActiveSignal(source=SignalSource.USGS, signal_type="earthquake",
                            description="M4.5 quake", port_id="la_lb",
                            observed_at=datetime(2024, 1, 3))
        card = self._make_card(active_signals=[sig1, sig2, sig3])
        counts = card.signal_count_by_source()
        assert counts["nws"] == 2
        assert counts["usgs"] == 1

    def test_top_driver_names(self):
        driver = SHAPDriver(feature="berthing_time_z_score_52w",
                            value=2.1, shap_impact=0.15,
                            direction="increases_risk")
        card = self._make_card(top_shap_drivers=[driver])
        assert card.top_driver_names() == ["berthing_time_z_score_52w"]

    def test_json_serialisable(self):
        card = self._make_card()
        data = card.model_dump(mode="json")
        # Should not raise
        json.dumps(data)

    def test_with_model_info(self):
        info = ModelInfo(version="v20240101_120000", cv_pr_auc=0.72)
        card = self._make_card(model_info=info)
        assert card.model_info.cv_pr_auc == pytest.approx(0.72)


# ─── scorer.py internals ─────────────────────────────────────────────────────

from src.scoring.scorer import _compute_risk_score, _confidence


class TestComputeRiskScore:
    def test_neutral_inputs_near_50(self):
        score = _compute_risk_score(0.5, 0.5, 0.5)
        assert 45.0 < score < 55.0

    def test_high_p_raises_score(self):
        low  = _compute_risk_score(0.3, 0.5, 0.5)
        high = _compute_risk_score(0.9, 0.5, 0.5)
        assert high > low

    def test_high_anomaly_raises_score(self):
        low  = _compute_risk_score(0.5, 0.2, 0.5)
        high = _compute_risk_score(0.5, 0.9, 0.5)
        assert high > low

    def test_high_impact_raises_score(self):
        low  = _compute_risk_score(0.5, 0.5, 0.1)
        high = _compute_risk_score(0.5, 0.5, 0.9)
        assert high > low

    def test_output_in_0_100(self):
        for p, a, i in [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5)]:
            score = _compute_risk_score(p, a, i)
            assert 0.0 <= score <= 100.0

    def test_p_dominates_anomaly(self):
        # w_p=0.5 > w_a=0.3, so swinging p has larger effect
        delta_p = _compute_risk_score(0.9, 0.5, 0.5) - _compute_risk_score(0.1, 0.5, 0.5)
        delta_a = _compute_risk_score(0.5, 0.9, 0.5) - _compute_risk_score(0.5, 0.1, 0.5)
        assert delta_p > delta_a

    def test_extreme_inputs(self):
        full_risk = _compute_risk_score(1.0, 1.0, 1.0)
        no_risk   = _compute_risk_score(0.0, 0.0, 0.0)
        assert full_risk > 85.0   # sigmoid(2.0) ≈ 88% — theoretical max with these weights
        assert no_risk   < 15.0


class TestConfidence:
    def test_full_data_high_signals_gives_100(self):
        score, flags = _confidence(
            data_completeness = 1.0,
            n_signals         = 5,
            n_shap_drivers    = 3,
            anomaly_score     = 0.7,
            model_loaded      = True,
        )
        assert score == pytest.approx(100.0)
        assert flags == []

    def test_no_model_deducts_40(self):
        score, flags = _confidence(
            data_completeness = 1.0,
            n_signals         = 5,
            n_shap_drivers    = 3,
            anomaly_score     = 0.7,
            model_loaded      = False,
        )
        assert score == pytest.approx(60.0)
        assert any("no_model_loaded" in f for f in flags)

    def test_low_completeness_deducts_30(self):
        score, flags = _confidence(
            data_completeness = 0.3,
            n_signals         = 5,
            n_shap_drivers    = 3,
            anomaly_score     = 0.7,
            model_loaded      = True,
        )
        assert score == pytest.approx(70.0)
        assert any("low_data_completeness" in f for f in flags)

    def test_partial_completeness_deducts_15(self):
        score, flags = _confidence(
            data_completeness = 0.6,
            n_signals         = 5,
            n_shap_drivers    = 3,
            anomaly_score     = 0.7,
            model_loaded      = True,
        )
        assert score == pytest.approx(85.0)
        assert any("partial_data" in f for f in flags)

    def test_few_signals_deducts_10(self):
        score, flags = _confidence(
            data_completeness = 1.0,
            n_signals         = 1,
            n_shap_drivers    = 3,
            anomaly_score     = 0.7,
            model_loaded      = True,
        )
        assert score == pytest.approx(90.0)
        assert any("few_signals" in f for f in flags)

    def test_no_shap_deducts_10(self):
        score, flags = _confidence(
            data_completeness = 1.0,
            n_signals         = 5,
            n_shap_drivers    = 0,
            anomaly_score     = 0.7,
            model_loaded      = True,
        )
        assert score == pytest.approx(90.0)
        assert any("no_shap_drivers" in f for f in flags)

    def test_neutral_anomaly_deducts_5(self):
        score, flags = _confidence(
            data_completeness = 1.0,
            n_signals         = 5,
            n_shap_drivers    = 3,
            anomaly_score     = 0.5,
            model_loaded      = True,
        )
        assert score == pytest.approx(95.0)
        assert any("anomaly_default" in f for f in flags)

    def test_score_never_below_0(self):
        score, _ = _confidence(
            data_completeness = 0.0,
            n_signals         = 0,
            n_shap_drivers    = 0,
            anomaly_score     = 0.5,
            model_loaded      = False,
        )
        assert score >= 0.0

    def test_multiple_flags_accumulate(self):
        score, flags = _confidence(
            data_completeness = 0.3,
            n_signals         = 0,
            n_shap_drivers    = 0,
            anomaly_score     = 0.5,
            model_loaded      = False,
        )
        assert len(flags) >= 4
        assert score <= 10.0


# ─── scorer.py integration (no model, no Bronze data) ────────────────────────

from src.scoring.scorer import RiskScorer


class TestRiskScorerNoModel:
    """Scorer must produce a valid RiskCard even with no model and no Bronze data."""

    def test_score_returns_risk_card(self):
        scorer = RiskScorer(registry=None)
        card   = scorer.score("la_lb")
        assert isinstance(card, RiskCard)

    def test_card_has_correct_port(self):
        scorer = RiskScorer(registry=None)
        card   = scorer.score("ny_nj")
        assert card.port_id   == "ny_nj"
        assert card.port_name == "New York / New Jersey"

    def test_score_in_range(self):
        scorer = RiskScorer(registry=None)
        card   = scorer.score("houston")
        assert 0.0 <= card.risk_score <= 100.0

    def test_confidence_flags_contain_no_model(self):
        scorer = RiskScorer(registry=None)
        card   = scorer.score("savannah")
        assert any("no_model_loaded" in f for f in card.confidence_flags)

    def test_recommended_actions_populated(self):
        scorer = RiskScorer(registry=None)
        card   = scorer.score("la_lb")
        assert len(card.recommended_actions) >= 2

    def test_risk_tier_consistent_with_score(self):
        scorer = RiskScorer(registry=None)
        card   = scorer.score("la_lb")
        assert card.risk_tier == score_to_tier(card.risk_score)

    def test_impact_index_matches_lookup(self):
        scorer = RiskScorer(registry=None)
        card   = scorer.score("la_lb")
        assert card.impact_index == pytest.approx(get_impact_index("la_lb"), abs=1e-4)

    def test_score_all_ports_returns_10_cards(self):
        scorer = RiskScorer(registry=None)
        cards  = scorer.score_all_ports()
        assert len(cards) == 10

    def test_score_all_ports_sorted_descending(self):
        scorer = RiskScorer(registry=None)
        cards  = scorer.score_all_ports()
        scores = [c.risk_score for c in cards]
        assert scores == sorted(scores, reverse=True)

    def test_card_is_json_serialisable(self):
        scorer = RiskScorer(registry=None)
        card   = scorer.score("oakland")
        data   = card.model_dump(mode="json")
        json.dumps(data)  # must not raise


# ─── signals.py ───────────────────────────────────────────────────────────────

from src.scoring.signals import assemble_signals


class TestAssembleSignals:
    def test_returns_list_when_no_bronze(self, tmp_path):
        signals = assemble_signals("la_lb", lookback_days=7, bronze_dir=tmp_path)
        assert isinstance(signals, list)

    def test_nws_signals_parsed(self, tmp_path):
        nws_dir = tmp_path / "nws"
        nws_dir.mkdir()
        record = {
            "port_id":    "la_lb",
            "event_type": "High Wind Warning",
            "severity":   "severe",
            "headline":   "Winds up to 70 mph",
            "effective":  (datetime.utcnow() - timedelta(hours=1)).isoformat(),
        }
        (nws_dir / "alerts.json").write_text(json.dumps([record]), encoding="utf-8")
        signals = assemble_signals("la_lb", lookback_days=7, bronze_dir=tmp_path)
        nws_sigs = [s for s in signals if s.source.value == "nws"]
        assert len(nws_sigs) == 1
        assert nws_sigs[0].severity.value == "high"

    def test_usgs_signals_parsed(self, tmp_path):
        usgs_dir = tmp_path / "usgs"
        usgs_dir.mkdir()
        record = {
            "nearest_port_id":   "la_lb",
            "magnitude":         6.5,
            "place":             "20km SW of Los Angeles",
            "distance_to_port_km": 25.0,
            "occurred_at":       (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            "tsunami_flag":      False,
        }
        (usgs_dir / "quakes.json").write_text(json.dumps([record]), encoding="utf-8")
        signals = assemble_signals("la_lb", lookback_days=7, bronze_dir=tmp_path)
        usgs_sigs = [s for s in signals if s.source.value == "usgs"]
        assert len(usgs_sigs) == 1
        assert usgs_sigs[0].severity == SignalSeverity.HIGH

    def test_old_signals_excluded(self, tmp_path):
        nws_dir = tmp_path / "nws"
        nws_dir.mkdir()
        old_record = {
            "port_id":    "la_lb",
            "event_type": "Flood Warning",
            "severity":   "moderate",
            "headline":   "Flooding expected",
            "effective":  (datetime.utcnow() - timedelta(days=30)).isoformat(),
        }
        (nws_dir / "old.json").write_text(json.dumps([old_record]), encoding="utf-8")
        signals = assemble_signals("la_lb", lookback_days=7, bronze_dir=tmp_path)
        assert len(signals) == 0

    def test_wrong_port_excluded(self, tmp_path):
        nws_dir = tmp_path / "nws"
        nws_dir.mkdir()
        record = {
            "port_id":    "ny_nj",
            "event_type": "Winter Storm",
            "severity":   "severe",
            "headline":   "Heavy snow",
            "effective":  (datetime.utcnow() - timedelta(hours=1)).isoformat(),
        }
        (nws_dir / "alerts.json").write_text(json.dumps([record]), encoding="utf-8")
        signals = assemble_signals("la_lb", lookback_days=7, bronze_dir=tmp_path)
        assert len(signals) == 0

    def test_signals_sorted_by_severity(self, tmp_path):
        usgs_dir = tmp_path / "usgs"
        nws_dir  = tmp_path / "nws"
        usgs_dir.mkdir()
        nws_dir.mkdir()

        critical_quake = {
            "nearest_port_id":   "la_lb",
            "magnitude":         7.5,
            "place":             "Near LA",
            "distance_to_port_km": 10.0,
            "occurred_at":       (datetime.utcnow() - timedelta(hours=3)).isoformat(),
            "tsunami_flag":      False,
        }
        low_weather = {
            "port_id":    "la_lb",
            "event_type": "Fog Advisory",
            "severity":   "minor",
            "headline":   "Dense fog",
            "effective":  (datetime.utcnow() - timedelta(hours=1)).isoformat(),
        }
        (usgs_dir / "q.json").write_text(json.dumps([critical_quake]), encoding="utf-8")
        (nws_dir  / "w.json").write_text(json.dumps([low_weather]),    encoding="utf-8")

        signals = assemble_signals("la_lb", lookback_days=7, bronze_dir=tmp_path)
        assert signals[0].severity == SignalSeverity.CRITICAL
