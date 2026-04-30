"""
SHAPExplainer — per-prediction feature attributions using SHAP TreeExplainer.

Why SHAP?
  TreeExplainer computes exact Shapley values for tree-based models efficiently.
  Each attribution is mathematically grounded: it represents that feature's
  average marginal contribution to the prediction across all feature orderings.
  This gives us a defensible "why did the model score this port 74/100?"
  answer — not a post-hoc rationalisation.

The explainer's output is the "truth anchor" for the LLM agent in Phase 5:
  the agent is only allowed to cite SHAP top drivers + retrieved evidence.
  It cannot invent causes. This is what separates the system from a chatbot.

Reference: Lundberg & Lee. "A Unified Approach to Interpreting Model Predictions."
NeurIPS 2017. https://arxiv.org/abs/1705.07874
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger(__name__)


@dataclass
class SHAPAttribution:
    """One feature's contribution to a single prediction."""

    feature_name:  str
    feature_value: float          # actual value of the feature in this row
    shap_impact:   float          # SHAP value: positive = pushed probability up
    abs_impact:    float = field(init=False)

    def __post_init__(self) -> None:
        self.abs_impact = abs(self.shap_impact)

    def direction(self) -> str:
        return "increases_risk" if self.shap_impact > 0 else "decreases_risk"

    def to_dict(self) -> dict:
        return {
            "feature":     self.feature_name,
            "value":       round(float(self.feature_value), 4),
            "shap_impact": round(float(self.shap_impact), 4),
            "direction":   self.direction(),
        }


class SHAPExplainer:
    """
    SHAP TreeExplainer wrapper for the trained XGBoost disruption model.

    Usage:
        explainer = SHAPExplainer(trainer.model, trainer.feature_columns)
        top_drivers = explainer.explain_row(X.iloc[0])
        # Returns list of SHAPAttribution sorted by |shap_impact| descending
    """

    TOP_N = 5   # number of top drivers to return per prediction

    def __init__(self, model, feature_columns: list[str]) -> None:
        import shap
        self._explainer       = shap.TreeExplainer(model)
        self._feature_columns = feature_columns
        self._background_mean: float | None = None  # E[f(X)] from training set

    # ── Row-level explanation ─────────────────────────────────────────────────

    def explain_row(self, X_row: pd.Series | pd.DataFrame) -> list[SHAPAttribution]:
        """
        Explain a single prediction.

        Args:
            X_row: one row as a Series (index = feature names) or a 1-row DataFrame.

        Returns:
            Top-N SHAPAttribution objects sorted by abs_impact descending.
        """
        if isinstance(X_row, pd.Series):
            X_row = X_row.to_frame().T

        X_aligned = X_row.reindex(columns=self._feature_columns, fill_value=0)

        try:
            shap_values = self._explainer.shap_values(X_aligned)
            # For binary XGBoost, shap_values can be a list [neg_class, pos_class]
            # or a 2D array. We want the positive class contributions.
            if isinstance(shap_values, list):
                sv = np.array(shap_values[1]).flatten()
            else:
                sv = np.array(shap_values).flatten()

            fv = X_aligned.values.flatten()
            attributions = [
                SHAPAttribution(
                    feature_name  = self._feature_columns[i],
                    feature_value = float(fv[i]),
                    shap_impact   = float(sv[i]),
                )
                for i in range(len(self._feature_columns))
            ]
            return sorted(attributions, key=lambda a: a.abs_impact, reverse=True)[: self.TOP_N]

        except Exception as exc:
            log.warning("shap_explain_row_error", error=str(exc))
            return []

    # ── Batch explanation ─────────────────────────────────────────────────────

    def explain_batch(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SHAP values for all rows in X.

        Returns:
            DataFrame with same index as X, columns = feature names,
            values = SHAP impact per feature per row.
        """
        X_aligned = X.reindex(columns=self._feature_columns, fill_value=0)
        try:
            shap_values = self._explainer.shap_values(X_aligned)
            if isinstance(shap_values, list):
                sv = np.array(shap_values[1])
            else:
                sv = np.array(shap_values)

            return pd.DataFrame(sv, index=X.index, columns=self._feature_columns)
        except Exception as exc:
            log.warning("shap_explain_batch_error", error=str(exc))
            return pd.DataFrame(index=X.index, columns=self._feature_columns)

    # ── Global feature importance ─────────────────────────────────────────────

    def global_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute mean |SHAP| per feature across all rows in X.
        Useful for the dashboard's "global model explanation" panel.

        Returns:
            DataFrame with columns [feature, mean_abs_shap] sorted descending.
        """
        shap_df = self.explain_batch(X)
        importance = (
            shap_df.abs()
            .mean()
            .reset_index()
            .rename(columns={"index": "feature", 0: "mean_abs_shap"})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        return importance

    # ── Convenience: evidence-ready format ───────────────────────────────────

    def top_drivers_as_evidence(self, X_row: pd.Series | pd.DataFrame) -> list[dict]:
        """
        Return top-N drivers in a dict format ready to include in the Evidence Pack.
        Used by the LLM agent (Phase 5) to anchor its explanation.
        """
        return [a.to_dict() for a in self.explain_row(X_row)]
