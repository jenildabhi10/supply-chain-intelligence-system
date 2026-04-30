"""
ProbabilityCalibrator — Platt scaling for XGBoost output probabilities.

Why calibrate?
  XGBoost probabilities are not inherently well-calibrated.  A raw output of
  0.7 does not necessarily mean "disruption occurs 70% of the time in similar
  conditions."  Calibration maps raw scores to reliable probabilities, which
  is critical for the risk score formula and for honest communication to
  decision-makers ("we are 70% confident" must mean something).

We use Platt scaling (sigmoid calibration) implemented via sklearn's
CalibratedClassifierCV with cv="prefit":
  - "prefit" means we calibrate a model that is already trained
  - We fit the calibrator on a held-out validation set (last N weeks of data)
  - This avoids using the same data for both training and calibration

After calibration, use `calibrated_proba()` instead of model.predict_proba().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
from sklearn.calibration import CalibratedClassifierCV

from src.models.features import TARGET_COLUMN, get_feature_matrix

log = structlog.get_logger(__name__)


class ProbabilityCalibrator:
    """
    Platt-scaling calibrator for the XGBoost disruption predictor.

    Fit on a held-out validation set after the base model is trained.
    """

    def __init__(self, method: str = "sigmoid") -> None:
        self.method    = method    # "sigmoid" (Platt) or "isotonic"
        self._cal_model: CalibratedClassifierCV | None = None

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        base_model,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> ProbabilityCalibrator:
        """
        Fit the calibrator on held-out validation data.

        Args:
            base_model: fitted XGBClassifier (with .predict_proba)
            X_val:      feature matrix for the validation set
            y_val:      true binary labels for the validation set
        """
        if len(y_val) < 20 or y_val.nunique() < 2:
            log.warning(
                "calibrator_fit_skipped",
                reason="too few validation samples or only one class",
                n_samples=len(y_val),
            )
            return self

        self._cal_model = CalibratedClassifierCV(
            estimator = base_model,
            method    = self.method,
            cv        = "prefit",      # base_model is already fitted
        )
        self._cal_model.fit(X_val, y_val.astype(int))
        log.info("calibrator_fitted", n_val=len(y_val), method=self.method)
        return self

    def fit_from_df(
        self,
        base_model,
        df:            pd.DataFrame,
        feature_cols:  list[str],
        val_frac:      float = 0.2,
    ) -> ProbabilityCalibrator:
        """
        Convenience: automatically hold out the last `val_frac` weeks as calibration set.
        Assumes df is sorted by week_ending.
        """
        df_labeled = df.dropna(subset=[TARGET_COLUMN]).copy()
        df_labeled = df_labeled.sort_values("week_ending")

        n_val      = max(int(len(df_labeled) * val_frac), 10)
        val_df     = df_labeled.iloc[-n_val:]
        X_val      = get_feature_matrix(val_df).reindex(columns=feature_cols, fill_value=0)
        y_val      = val_df[TARGET_COLUMN].astype(int)
        return self.fit(base_model, X_val, y_val)

    # ── Predict ───────────────────────────────────────────────────────────────

    def calibrated_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return calibrated probabilities.  Falls back to base model if
        calibrator is not fitted (e.g. not enough validation data).

        Args:
            X: feature matrix (same columns as training)

        Returns:
            np.ndarray of shape (n,) with values in [0, 1].
        """
        if self._cal_model is None:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        return self._cal_model.predict_proba(X)[:, 1]

    @property
    def is_fitted(self) -> bool:
        return self._cal_model is not None
