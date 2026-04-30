"""
AnomalyDetector — Isolation Forest for early-warning port anomaly detection.

This runs as a PARALLEL pipeline to the supervised XGBoost predictor.
Its purpose is different: it flags "this looks statistically unusual"
even before the XGBoost model has enough evidence to raise a high probability.

Key design decisions:
  - Fit on ALL historical rows (not just normal weeks) — IsolationForest is
    unsupervised and isolates anomalies via random partitioning. Fitting on
    normal-only data is an option but risks excluding some anomaly patterns
    from the reference distribution.
  - Output is normalised to [0, 1] where 1 = most anomalous.
    Raw IsolationForest scores are negative and unbounded; normalisation
    makes them composable with the XGBoost probability in the risk score.
  - contamination=0.05 means IsolationForest expects ~5% of training rows
    to be anomalies. Adjust if your disruption base rate is very different.

Reference: Liu, Fei Tony, Ting, Kai Ming, Zhou, Zhi-Hua. "Isolation Forest."
ICDM 2008. https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/tkdd11.pdf
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from sklearn.ensemble import IsolationForest

from src.models.features import ANOMALY_FEATURE_COLUMNS

log = structlog.get_logger(__name__)


class AnomalyDetector:
    """
    Isolation Forest wrapper with [0, 1] score normalisation.

    Usage:
        detector = AnomalyDetector()
        detector.fit(gold_df)
        scores = detector.score(gold_df)  # array of floats in [0, 1]
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200) -> None:
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self._model: IsolationForest | None = None
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._feature_columns: list[str] | None = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> AnomalyDetector:
        """
        Fit the Isolation Forest on all available feature rows.

        Args:
            df: Gold feature table.  Rows with all-zero features are skipped.
        """
        X = self._extract_features(df)
        if X.empty:
            log.warning("anomaly_fit_skipped", reason="empty feature matrix")
            return self

        self._model = IsolationForest(
            n_estimators    = self.n_estimators,
            contamination   = self.contamination,
            random_state    = 42,
            n_jobs          = -1,
        )
        self._model.fit(X)
        self._feature_columns = list(X.columns)

        # Compute normalisation bounds from training data
        raw_scores         = self._model.score_samples(X)
        self._score_min    = float(raw_scores.min())
        self._score_max    = float(raw_scores.max())

        n_anomalies = (self._model.predict(X) == -1).sum()
        log.info(
            "anomaly_detector_fitted",
            rows        = len(X),
            n_anomalies = int(n_anomalies),
            anomaly_pct = round(n_anomalies / len(X) * 100, 1),
        )
        return self

    # ── Score ─────────────────────────────────────────────────────────────────

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return anomaly scores in [0, 1] for each row in df.
        Higher = more anomalous (opposite of IsolationForest's raw convention).

        Rows that cannot be scored (missing features) get score 0.0.
        """
        if self._model is None:
            log.warning("anomaly_score_called_before_fit")
            return np.zeros(len(df))

        X = self._extract_features(df)
        if X.empty:
            return np.zeros(len(df))

        X = X.reindex(columns=self._feature_columns, fill_value=0)
        raw = self._model.score_samples(X)
        return self._normalise(raw)

    def is_anomaly(self, df: pd.DataFrame, threshold: float = 0.7) -> np.ndarray:
        """
        Return boolean array: True where anomaly_score > threshold.
        Default threshold 0.7 = top ~10% most anomalous rows.
        """
        return self.score(df) > threshold

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and fill anomaly feature columns from the Gold DataFrame."""
        available = [c for c in ANOMALY_FEATURE_COLUMNS if c in df.columns]
        if not available:
            return pd.DataFrame()
        X = df[available].copy().fillna(0).astype(float)
        return X

    def _normalise(self, raw_scores: np.ndarray) -> np.ndarray:
        """
        Map raw IsolationForest scores to [0, 1].
        Raw scores are negative; lower = more anomalous.
        We invert so that higher = more anomalous.
        """
        span = self._score_max - self._score_min
        if span < 1e-9:
            return np.zeros_like(raw_scores)
        normalised = (raw_scores - self._score_min) / span   # [0, 1], higher = more normal
        inverted   = 1.0 - normalised                          # [0, 1], higher = more anomalous
        return np.clip(inverted, 0.0, 1.0)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None
