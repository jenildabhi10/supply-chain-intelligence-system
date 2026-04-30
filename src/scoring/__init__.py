# Phase 4: Risk scoring engine — produces structured RiskCard objects

from src.scoring.impact import IMPACT_INDEX, get_impact_index
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
from src.scoring.scorer import RiskScorer
from src.scoring.signals import assemble_signals

__all__ = [
    "ActiveSignal",
    "IMPACT_INDEX",
    "ModelInfo",
    "RECOMMENDED_ACTIONS",
    "RiskCard",
    "RiskScorer",
    "RiskTier",
    "SHAPDriver",
    "SignalSeverity",
    "SignalSource",
    "assemble_signals",
    "get_impact_index",
    "score_to_tier",
]
