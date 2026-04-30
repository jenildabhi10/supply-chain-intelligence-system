"""
DisruptionTrainer — XGBoost training pipeline for port disruption prediction.

Pipeline:
  1. Load Gold feature table
  2. Walk-forward cross-validation to estimate PR-AUC
  3. Optuna hyperparameter search (objective = mean PR-AUC across CV folds)
  4. Train final model on all labeled data with best hyperparameters
  5. Log everything to MLflow (local mlruns/ directory, no server needed)
  6. Return trained model + CV metrics

Why PR-AUC, not AUC-ROC?
  Disruption weeks are ~20-25% of all weeks — a minority class.
  PR-AUC focuses on precision-recall trade-off for the positive class,
  which is what an operations analyst actually cares about:
  "When the model raises an alert, how often is it right?"

Why walk-forward CV, not random k-fold?
  Random splits allow future data to inform past predictions (leakage).
  Walk-forward splits train strictly on the past and validate on the future,
  matching how the model is used in production.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import structlog
import xgboost as xgb
from sklearn.metrics import average_precision_score

from src.models.features import (
    TARGET_COLUMN,
    get_feature_matrix,
)

log = structlog.get_logger(__name__)

# Suppress Optuna and XGBoost verbosity
warnings.filterwarnings("ignore", category=UserWarning)


# ─── Walk-forward split ───────────────────────────────────────────────────────

def walk_forward_splits(
    df: pd.DataFrame,
    n_splits:       int = 5,
    test_weeks:     int = 4,
    min_train_weeks: int = 26,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward (expanding window) train/test index pairs.

    Splits are based on `week_ending` dates, not row indices, so that
    the multi-port dataset is handled correctly — all ports share the
    same time axis.

    Each test window covers exactly `test_weeks` consecutive Sundays.
    Train window expands from the beginning through the Sunday before
    each test window.

    Returns:
        List of (train_indices, test_indices) tuples.
        May be shorter than n_splits if the dataset doesn't have enough history.
    """
    weeks       = sorted(df["week_ending"].unique())
    total_weeks = len(weeks)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for fold in range(n_splits, 0, -1):
        test_end_idx   = total_weeks - (fold - 1) * test_weeks
        test_start_idx = test_end_idx - test_weeks

        if test_start_idx < min_train_weeks:
            continue   # not enough history to train meaningfully

        train_weeks = set(weeks[:test_start_idx])
        test_weeks_ = set(weeks[test_start_idx:test_end_idx])

        train_idx = df.index[df["week_ending"].isin(train_weeks)].to_numpy()
        test_idx  = df.index[df["week_ending"].isin(test_weeks_)].to_numpy()

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


# ─── XGBoost training helpers ─────────────────────────────────────────────────

def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Handle class imbalance: weight = negative_count / positive_count."""
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return float(neg / pos) if pos > 0 else 1.0


def _train_xgb(X: pd.DataFrame, y: pd.Series, params: dict[str, Any]) -> xgb.XGBClassifier:
    """Train a single XGBoost model on (X, y) with the given params."""
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight = _compute_scale_pos_weight(y),
        use_label_encoder = False,
        eval_metric        = "aucpr",
        random_state       = 42,
        n_jobs             = -1,
        verbosity          = 0,
    )
    model.fit(X, y, verbose=False)
    return model


def _cv_pr_auc(
    df_labeled:  pd.DataFrame,
    params:      dict[str, Any],
    splits:      list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """
    Run walk-forward CV with the given params and return mean PR-AUC.
    Rows without a label (NaN) are excluded from both train and validation.
    """
    scores: list[float] = []

    for train_idx, test_idx in splits:
        train_df = df_labeled.loc[train_idx].dropna(subset=[TARGET_COLUMN])
        test_df  = df_labeled.loc[test_idx].dropna(subset=[TARGET_COLUMN])

        if len(train_df) < 20 or test_df[TARGET_COLUMN].nunique() < 2:
            continue   # degenerate fold — skip rather than crash

        X_train = get_feature_matrix(train_df)
        y_train = train_df[TARGET_COLUMN].astype(int)
        X_test  = get_feature_matrix(test_df)
        y_test  = test_df[TARGET_COLUMN].astype(int)

        # Align columns (in case a fold lacks some feature columns)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        try:
            model  = _train_xgb(X_train, y_train, params)
            y_prob = model.predict_proba(X_test)[:, 1]
            scores.append(average_precision_score(y_test, y_prob))
        except Exception as exc:
            log.warning("cv_fold_error", error=str(exc))

    return float(np.mean(scores)) if scores else 0.0


# ─── Optuna hyperparameter search ────────────────────────────────────────────

def _optuna_search(
    df_labeled: pd.DataFrame,
    splits:     list[tuple[np.ndarray, np.ndarray]],
    n_trials:   int = 30,
) -> dict[str, Any]:
    """
    Use Optuna to find the best XGBoost hyperparameters.
    Objective: maximise mean walk-forward CV PR-AUC.
    Returns the best parameter dict.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        }
        return _cv_pr_auc(df_labeled, params, splits)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    log.info(
        "optuna_complete",
        best_pr_auc = round(study.best_value, 4),
        best_params = study.best_params,
        n_trials    = n_trials,
    )
    return study.best_params


# ─── Main trainer class ───────────────────────────────────────────────────────

class DisruptionTrainer:
    """
    End-to-end XGBoost training pipeline with walk-forward CV and MLflow tracking.

    Usage:
        trainer = DisruptionTrainer()
        metrics = trainer.train(gold_df)
        # trainer.model is now the fitted XGBClassifier
        # trainer.feature_columns is the column list used
    """

    def __init__(
        self,
        n_cv_folds:     int = 5,
        test_weeks:     int = 4,
        n_optuna_trials: int = 30,
        use_mlflow:     bool = True,
    ) -> None:
        self.n_cv_folds      = n_cv_folds
        self.test_weeks      = test_weeks
        self.n_optuna_trials = n_optuna_trials
        self.use_mlflow      = use_mlflow

        self.model:           xgb.XGBClassifier | None = None
        self.feature_columns: list[str] | None         = None
        self.cv_metrics:      dict[str, float]         = {}
        self.best_params:     dict[str, Any]           = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Full training pipeline.

        Args:
            df: Gold feature table (port_week_features.parquet)

        Returns:
            metrics dict: cv_pr_auc, holdout_pr_auc, n_train_rows, n_labeled_rows, etc.
        """
        df = df.copy()
        df["week_ending"] = pd.to_datetime(df["week_ending"])

        # Filter to rows with valid labels
        labeled = df.dropna(subset=[TARGET_COLUMN]).copy()
        labeled[TARGET_COLUMN] = labeled[TARGET_COLUMN].astype(int)

        log.info("trainer_start", total_rows=len(df), labeled_rows=len(labeled))

        if len(labeled) < 40:
            log.warning(
                "insufficient_data",
                labeled_rows=len(labeled),
                msg="Need at least 40 labeled rows for meaningful training. "
                    "Run the scheduler for several weeks to collect more BTS data.",
            )
            # Still train — produces a model, but metrics will be unreliable
            if len(labeled) < 10:
                raise ValueError(f"Too few labeled rows ({len(labeled)}) to train. Run ingestion first.")

        # Build walk-forward splits
        splits = walk_forward_splits(
            labeled,
            n_splits        = self.n_cv_folds,
            test_weeks      = self.test_weeks,
            min_train_weeks = 26,
        )
        log.info("cv_splits", n_splits=len(splits))

        run_fn = self._train_with_mlflow if self.use_mlflow else self._train_core
        return run_fn(labeled, splits)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict disruption probability for each row in df.

        Returns:
            np.ndarray of shape (n,) with calibrated probabilities in [0, 1].
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X = get_feature_matrix(df).reindex(columns=self.feature_columns, fill_value=0)
        return self.model.predict_proba(X)[:, 1]

    # ── Internal training logic ────────────────────────────────────────────────

    def _train_core(
        self,
        labeled: pd.DataFrame,
        splits:  list[tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, float]:
        """Train without MLflow (used when use_mlflow=False or as a fallback)."""
        # Step 1: CV with default params to get baseline PR-AUC
        default_params = {
            "n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
        }
        baseline_pr_auc = _cv_pr_auc(labeled, default_params, splits) if splits else 0.0
        log.info("baseline_cv_pr_auc", value=round(baseline_pr_auc, 4))

        # Step 2: Optuna search (only if we have enough data for meaningful CV)
        if splits and len(labeled) >= 80:
            best_params = _optuna_search(labeled, splits, n_trials=self.n_optuna_trials)
        else:
            log.warning("optuna_skipped", reason="too few splits or data — using default params")
            best_params = default_params

        self.best_params = best_params

        # Step 3: Re-evaluate with best params
        tuned_pr_auc = _cv_pr_auc(labeled, best_params, splits) if splits else 0.0

        # Step 4: Train final model on all labeled data
        X_all = get_feature_matrix(labeled)
        y_all = labeled[TARGET_COLUMN].astype(int)

        self.model           = _train_xgb(X_all, y_all, best_params)
        self.feature_columns = list(X_all.columns)

        self.cv_metrics = {
            "baseline_cv_pr_auc": round(baseline_pr_auc, 4),
            "tuned_cv_pr_auc":    round(tuned_pr_auc, 4),
            "n_cv_folds":         len(splits),
            "n_train_rows":       len(labeled),
            "n_features":         len(self.feature_columns),
            "pos_rate":           round(labeled[TARGET_COLUMN].mean(), 4),
            "scale_pos_weight":   round(_compute_scale_pos_weight(y_all), 2),
        }
        log.info("training_complete", **self.cv_metrics)
        return self.cv_metrics

    def _train_with_mlflow(
        self,
        labeled: pd.DataFrame,
        splits:  list[tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, float]:
        """Train with full MLflow experiment tracking."""
        try:
            import mlflow
            import mlflow.xgboost

            mlflow.set_tracking_uri("mlruns")
            mlflow.set_experiment("supply_chain_disruption")

            with mlflow.start_run(run_name=f"train_{datetime.utcnow():%Y%m%d_%H%M%S}"):
                metrics = self._train_core(labeled, splits)

                # Log params + metrics to MLflow
                if self.best_params:
                    mlflow.log_params(self.best_params)
                mlflow.log_metrics(metrics)
                mlflow.log_param("n_cv_folds", self.n_cv_folds)
                mlflow.log_param("n_optuna_trials", self.n_optuna_trials)

                # Log model artifact
                if self.model is not None:
                    mlflow.xgboost.log_model(self.model, artifact_path="xgboost_model")

                mlflow.set_tag("phase", "phase3_ml")
                run_id = mlflow.active_run().info.run_id
                log.info("mlflow_run_complete", run_id=run_id)

            return metrics
        except ImportError:
            log.warning("mlflow_not_available", fallback="training without tracking")
            return self._train_core(labeled, splits)
        except Exception as exc:
            log.warning("mlflow_error", error=str(exc), fallback="training without tracking")
            return self._train_core(labeled, splits)
