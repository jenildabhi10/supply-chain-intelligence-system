"""
RiskScorer — assembles a complete RiskCard from model outputs + live signals.

Formula:
    p_norm  = (p_disruption  - 0.5) * 4     # centred, scaled
    a_norm  = (anomaly_score - 0.5) * 4
    i_norm  = (impact_index  - 0.5) * 4
    raw     = 0.5·p_norm + 0.3·a_norm + 0.2·i_norm
    score   = 100 × sigmoid(raw)             → [0, 100]

Confidence computation:
    Starts at 100. Deductions applied per flag:
      - No model loaded              → −40
      - Data completeness < 0.5      → −30
      - Data completeness 0.5–0.75   → −15
      - Fewer than 2 signals         → −10
      - No SHAP drivers              → −10
      - Anomaly score is exactly 0.5 → −5  (default/no anomaly data)

The scorer is stateless — create once, call score() repeatedly.
"""

from __future__ import annotations

import math
from pathlib import Path

import structlog

from config import US_PORTS
from src.scoring.impact import get_impact_index
from src.scoring.risk_card import (
    RECOMMENDED_ACTIONS,
    ActiveSignal,
    ModelInfo,
    RiskCard,
    SHAPDriver,
    score_to_tier,
)
from src.scoring.signals import assemble_signals

log = structlog.get_logger(__name__)

# Composite weight vector — must sum to 1.0
_W_P  = 0.5   # calibrated disruption probability
_W_A  = 0.3   # anomaly score
_W_I  = 0.2   # port impact index


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _compute_risk_score(
    p_disruption: float,
    anomaly_score: float,
    impact_index:  float,
) -> float:
    """Map three [0,1] inputs to a [0,100] risk score via weighted sigmoid."""
    p_norm = (p_disruption  - 0.5) * 4
    a_norm = (anomaly_score - 0.5) * 4
    i_norm = (impact_index  - 0.5) * 4
    raw    = _W_P * p_norm + _W_A * a_norm + _W_I * i_norm
    return round(100.0 * _sigmoid(raw), 2)


def _confidence(
    data_completeness: float,
    n_signals:         int,
    n_shap_drivers:    int,
    anomaly_score:     float,
    model_loaded:      bool,
) -> tuple[float, list[str]]:
    """Return (confidence_score 0–100, list_of_flag_strings)."""
    score  = 100.0
    flags: list[str] = []

    if not model_loaded:
        score -= 40.0
        flags.append("no_model_loaded: scoring used fallback heuristics only")

    if data_completeness < 0.5:
        score -= 30.0
        flags.append(f"low_data_completeness: only {data_completeness:.0%} of sources contributed")
    elif data_completeness < 0.75:
        score -= 15.0
        flags.append(f"partial_data: {data_completeness:.0%} of sources contributed")

    if n_signals < 2:
        score -= 10.0
        flags.append("few_signals: fewer than 2 live signals found in lookback window")

    if n_shap_drivers == 0:
        score -= 10.0
        flags.append("no_shap_drivers: model explanation unavailable")

    if abs(anomaly_score - 0.5) < 1e-6:
        score -= 5.0
        flags.append("anomaly_default: anomaly detector returned neutral score (may lack data)")

    return max(0.0, score), flags


# ─── Port metadata helpers ────────────────────────────────────────────────────

_PORT_LOOKUP = {p.id: p for p in US_PORTS}


def _port_name(port_id: str) -> str:
    return _PORT_LOOKUP[port_id].name if port_id in _PORT_LOOKUP else port_id


def _port_region(port_id: str) -> str:
    return _PORT_LOOKUP[port_id].region if port_id in _PORT_LOOKUP else "Unknown"


# ─── SHAPDriver conversion ───────────────────────────────────────────────────

def _shap_drivers_from_attributions(attributions: list) -> list[SHAPDriver]:
    """Convert SHAPAttribution objects (from Phase 3 explainer) to SHAPDriver."""
    drivers = []
    for attr in attributions:
        drivers.append(SHAPDriver(
            feature     = attr.feature_name,
            value       = float(attr.feature_value),
            shap_impact = float(attr.shap_impact),
            direction   = attr.direction(),
        ))
    return drivers


# ─── RiskScorer ──────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Converts model outputs into a fully populated RiskCard.

    Usage:
        scorer = RiskScorer(registry=ModelRegistry())
        card   = scorer.score("la_lb", forecast_horizon_days=7)
    """

    def __init__(
        self,
        registry=None,
        bronze_dir:    Path | None = None,
        lookback_days: int = 7,
    ) -> None:
        self._registry     = registry
        self._bronze_dir   = bronze_dir
        self._lookback     = lookback_days
        self._bundle: dict | None = None
        self._gold_df      = None

        if registry is not None:
            try:
                self._bundle = registry.load_champion()
                log.info("scorer_loaded_champion", version=self._bundle.get("version"))
            except RuntimeError:
                log.warning("scorer_no_champion", msg="Will use heuristic scoring")

    def _load_gold(self):
        if self._gold_df is not None:
            return self._gold_df
        import pandas as pd
        gold_path = Path("data/gold/port_week_features.parquet")
        if gold_path.exists():
            self._gold_df = pd.read_parquet(gold_path)
        return self._gold_df

    def score(
        self,
        port_id:               str,
        forecast_horizon_days: int = 7,
        features_week_ending:  str | None = None,
    ) -> RiskCard:
        """
        Build a complete RiskCard for the given port.

        Args:
            port_id:               Port identifier (e.g. "la_lb").
            forecast_horizon_days: How many days ahead the score covers.
            features_week_ending:  ISO date of the feature snapshot used.
                                   Auto-detected from Gold table if None.

        Returns:
            Fully populated RiskCard.
        """
        model_loaded   = self._bundle is not None
        p_disruption   = 0.5   # neutral prior
        anomaly_score  = 0.5
        shap_drivers:  list[SHAPDriver] = []
        model_info:    ModelInfo | None = None

        # ── 1. Model inference ────────────────────────────────────────────────
        gold_df = self._load_gold()
        if model_loaded and gold_df is not None:
            try:
                from src.models.features import get_feature_matrix
                port_df = gold_df[gold_df["port_id"] == port_id].copy()

                if not port_df.empty:
                    # Use the most recent week
                    port_df = port_df.sort_values("week_ending")
                    latest_row = port_df.iloc[[-1]]

                    if features_week_ending is None:
                        features_week_ending = str(latest_row["week_ending"].iloc[0])

                    X = get_feature_matrix(latest_row)

                    trainer    = self._bundle.get("trainer")
                    calibrator = self._bundle.get("calibrator")
                    anomaly_d  = self._bundle.get("anomaly_detector")
                    explainer  = self._bundle.get("explainer")

                    if calibrator is not None:
                        proba = calibrator.predict_proba(X)
                        p_disruption = float(proba[0])
                    elif trainer is not None:
                        p_disruption = float(trainer.predict_proba(X)[0])

                    if anomaly_d is not None:
                        anomaly_score = float(anomaly_d.score(X)[0])

                    if explainer is not None:
                        attributions = explainer.explain_row(X)
                        shap_drivers = _shap_drivers_from_attributions(attributions)

                    metrics = self._bundle.get("metrics", {})
                    model_info = ModelInfo(
                        version         = self._bundle.get("version", "unknown"),
                        cv_pr_auc       = metrics.get("tuned_cv_pr_auc", metrics.get("cv_pr_auc")),
                        n_training_rows = metrics.get("n_train_rows"),
                    )
            except Exception as exc:
                log.warning("scorer_inference_error", port_id=port_id, error=str(exc))
                model_loaded = False

        # ── 2. Impact index ───────────────────────────────────────────────────
        impact_index = get_impact_index(port_id)

        # ── 3. Risk score ─────────────────────────────────────────────────────
        risk_score = _compute_risk_score(p_disruption, anomaly_score, impact_index)
        risk_tier  = score_to_tier(risk_score)

        # ── 4. Live signals ───────────────────────────────────────────────────
        active_signals: list[ActiveSignal] = []
        try:
            active_signals = assemble_signals(
                port_id       = port_id,
                lookback_days = self._lookback,
                bronze_dir    = self._bronze_dir,
            )
        except Exception as exc:
            log.warning("scorer_signals_error", port_id=port_id, error=str(exc))

        # ── 5. Data completeness ──────────────────────────────────────────────
        if gold_df is not None and not gold_df[gold_df["port_id"] == port_id].empty:
            port_df = gold_df[gold_df["port_id"] == port_id]
            dc_col = "data_completeness_score"
            if dc_col in port_df.columns:
                data_completeness = float(port_df.sort_values("week_ending")[dc_col].iloc[-1])
            else:
                data_completeness = 0.5
        else:
            data_completeness = 0.0

        # ── 6. Confidence ─────────────────────────────────────────────────────
        confidence_score, confidence_flags = _confidence(
            data_completeness = data_completeness,
            n_signals         = len(active_signals),
            n_shap_drivers    = len(shap_drivers),
            anomaly_score     = anomaly_score,
            model_loaded      = model_loaded,
        )

        # ── 7. Assemble card ─────────────────────────────────────────────────
        card = RiskCard(
            port_id               = port_id,
            port_name             = _port_name(port_id),
            region                = _port_region(port_id),
            risk_score            = risk_score,
            risk_tier             = risk_tier,
            p_disruption          = p_disruption,
            anomaly_score         = anomaly_score,
            impact_index          = impact_index,
            confidence_score      = confidence_score,
            confidence_flags      = confidence_flags,
            data_completeness     = data_completeness,
            forecast_horizon_days = forecast_horizon_days,
            features_week_ending  = features_week_ending,
            top_shap_drivers      = shap_drivers,
            active_signals        = active_signals,
            recommended_actions   = RECOMMENDED_ACTIONS[risk_tier],
            model_info            = model_info,
        )

        log.info(
            "risk_card_generated",
            port_id    = port_id,
            risk_score = card.risk_score,
            risk_tier  = card.risk_tier.value,
            confidence = card.confidence_score,
            n_signals  = len(active_signals),
        )
        return card

    def score_all_ports(self, forecast_horizon_days: int = 7) -> list[RiskCard]:
        """Score every monitored port and return cards sorted by risk_score desc."""
        cards = []
        for port in US_PORTS:
            try:
                card = self.score(port.id, forecast_horizon_days=forecast_horizon_days)
                cards.append(card)
            except Exception as exc:
                log.error("scorer_port_failed", port_id=port.id, error=str(exc))
        cards.sort(key=lambda c: c.risk_score, reverse=True)
        return cards
