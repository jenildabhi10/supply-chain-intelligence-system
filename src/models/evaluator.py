"""
Model evaluation utilities — PR-AUC, walk-forward backtest, calibration curves.

All functions are pure (take arrays/DataFrames, return DataFrames/dicts).
No side effects — suitable for both offline evaluation and CI tests.

Why PR-AUC as the primary metric?
  Disruption weeks are a minority class (~20-25%).  In imbalanced settings,
  AUC-ROC can be misleadingly optimistic because it counts true negatives
  heavily.  PR-AUC (Average Precision) focuses only on the positive class:
  it answers "how useful are the model's alerts?" rather than "how often
  is the model right on the easy cases?"

Walk-forward backtest:
  Unlike k-fold CV which splits randomly, walk-forward CV splits on time.
  This is the only correct evaluation strategy for time-series classification.
  Each fold's test window comes AFTER its training window — exactly as in
  production where you predict the future, not random held-out samples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.models.features import TARGET_COLUMN, get_feature_matrix
from src.models.trainer import walk_forward_splits

log = structlog.get_logger(__name__)


# ─── Core metrics ─────────────────────────────────────────────────────────────

def compute_pr_auc(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    """Average Precision (area under PR curve) for the positive class."""
    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    if yt.sum() == 0 or yt.sum() == len(yt):
        return 0.0
    return float(average_precision_score(yt, ys))


def compute_roc_auc(y_true: np.ndarray | pd.Series, y_score: np.ndarray | pd.Series) -> float:
    yt = np.asarray(y_true, dtype=int)
    ys = np.asarray(y_score, dtype=float)
    if yt.nunique() < 2 if hasattr(yt, "nunique") else np.unique(yt).size < 2:
        return 0.5
    return float(roc_auc_score(yt, ys))


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    """Precision among the top-k highest-scored predictions."""
    top_k_idx = np.argsort(y_score)[::-1][:k]
    return float(np.mean(np.asarray(y_true)[top_k_idx]))


# ─── Walk-forward backtest ────────────────────────────────────────────────────

def walk_forward_report(
    df:              pd.DataFrame,
    model,
    feature_columns: list[str],
    n_splits:        int = 5,
    test_weeks:      int = 4,
) -> pd.DataFrame:
    """
    Run a full walk-forward backtest and return per-fold metrics.

    Each fold trains strictly on past data and evaluates on the immediately
    following `test_weeks` window.

    Returns:
        DataFrame with columns:
            fold, train_start, train_end, test_start, test_end,
            n_train, n_test, pos_rate_test, pr_auc, roc_auc, precision_at_10
    """
    df = df.copy()
    df["week_ending"] = pd.to_datetime(df["week_ending"])

    labeled = df.dropna(subset=[TARGET_COLUMN]).copy()
    labeled[TARGET_COLUMN] = labeled[TARGET_COLUMN].astype(int)

    if len(labeled) < 40:
        log.warning("backtest_insufficient_data", n=len(labeled))
        return pd.DataFrame()

    splits = walk_forward_splits(labeled, n_splits=n_splits, test_weeks=test_weeks)
    rows: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        train_df = labeled.loc[train_idx]
        test_df  = labeled.loc[test_idx]

        if test_df[TARGET_COLUMN].nunique() < 2:
            continue

        X_train = get_feature_matrix(train_df).reindex(columns=feature_columns, fill_value=0)
        y_train = train_df[TARGET_COLUMN]
        X_test  = get_feature_matrix(test_df).reindex(columns=feature_columns, fill_value=0)
        y_test  = test_df[TARGET_COLUMN]

        try:
            import xgboost as xgb
            fold_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                use_label_encoder=False, eval_metric="aucpr",
                random_state=42, verbosity=0, n_jobs=-1,
            )
            fold_model.fit(X_train, y_train, verbose=False)
            y_prob = fold_model.predict_proba(X_test)[:, 1]

            we = pd.to_datetime(labeled["week_ending"])
            rows.append({
                "fold":           fold_idx + 1,
                "train_start":    we.loc[train_idx].min().strftime("%Y-%m-%d"),
                "train_end":      we.loc[train_idx].max().strftime("%Y-%m-%d"),
                "test_start":     we.loc[test_idx].min().strftime("%Y-%m-%d"),
                "test_end":       we.loc[test_idx].max().strftime("%Y-%m-%d"),
                "n_train":        len(train_df),
                "n_test":         len(test_df),
                "pos_rate_test":  round(float(y_test.mean()), 3),
                "pr_auc":         round(compute_pr_auc(y_test, y_prob), 4),
                "roc_auc":        round(compute_roc_auc(y_test, y_prob), 4),
                "precision_at_10": round(precision_at_k(y_test.values, y_prob, k=min(10, len(y_test))), 3),
            })
        except Exception as exc:
            log.warning("backtest_fold_error", fold=fold_idx + 1, error=str(exc))

    report = pd.DataFrame(rows)
    if not report.empty:
        log.info(
            "backtest_complete",
            n_folds       = len(report),
            mean_pr_auc   = round(report["pr_auc"].mean(), 4),
            mean_roc_auc  = round(report["roc_auc"].mean(), 4),
        )
    return report


# ─── Calibration curve data ───────────────────────────────────────────────────

def calibration_data(
    y_true:  np.ndarray | pd.Series,
    y_prob:  np.ndarray | pd.Series,
    n_bins:  int = 10,
) -> pd.DataFrame:
    """
    Compute reliability diagram data (calibration curve).

    A well-calibrated model should have predicted_prob ≈ actual_freq.
    Points on the diagonal = perfect calibration.

    Returns:
        DataFrame with columns [mean_predicted_prob, fraction_of_positives, count]
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)

    if len(yt) < n_bins:
        return pd.DataFrame()

    frac_pos, mean_pred = calibration_curve(yt, yp, n_bins=n_bins, strategy="uniform")

    # Count samples per bin
    bins = np.linspace(0, 1, n_bins + 1)
    counts = np.histogram(yp, bins=bins)[0]

    return pd.DataFrame({
        "mean_predicted_prob":    mean_pred,
        "fraction_of_positives":  frac_pos,
        "count":                  counts[: len(mean_pred)],
    })


# ─── Precision-Recall curve data ─────────────────────────────────────────────

def pr_curve_data(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """
    Compute Precision-Recall curve data for plotting.

    Returns:
        DataFrame with columns [precision, recall, threshold]
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_prob, dtype=float)

    precision, recall, thresholds = precision_recall_curve(yt, yp)
    return pd.DataFrame({
        "precision": precision[:-1],
        "recall":    recall[:-1],
        "threshold": thresholds,
    })


# ─── Feature drift detection ──────────────────────────────────────────────────

def feature_drift_report(
    reference_df: pd.DataFrame,
    current_df:   pd.DataFrame,
    feature_cols: list[str],
    z_threshold:  float = 2.0,
) -> pd.DataFrame:
    """
    Detect features that have drifted significantly from the training distribution.

    Uses Population Stability Index (PSI) approximation:
    compares the mean and std of each feature in current vs. reference.
    A z-score > z_threshold flags potential drift.

    Returns:
        DataFrame with columns [feature, ref_mean, cur_mean, ref_std, cur_std, z_score, drifted]
    """
    rows = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = reference_df[col].dropna()
        cur_vals = current_df[col].dropna()

        if ref_vals.empty or cur_vals.empty:
            continue

        ref_mean, ref_std = ref_vals.mean(), ref_vals.std()
        cur_mean = cur_vals.mean()

        z_score = abs(cur_mean - ref_mean) / max(ref_std, 1e-6)
        rows.append({
            "feature":  col,
            "ref_mean": round(ref_mean, 4),
            "cur_mean": round(cur_mean, 4),
            "ref_std":  round(ref_std, 4),
            "cur_std":  round(cur_vals.std(), 4),
            "z_score":  round(z_score, 2),
            "drifted":  z_score > z_threshold,
        })

    report = pd.DataFrame(rows).sort_values("z_score", ascending=False)
    n_drifted = report["drifted"].sum() if not report.empty else 0
    if n_drifted:
        log.warning("feature_drift_detected", n_drifted=int(n_drifted))
    return report
