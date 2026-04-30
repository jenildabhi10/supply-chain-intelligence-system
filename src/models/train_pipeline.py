"""
Full ML training pipeline — entry point for Phase 3.

Loads the Gold feature table, trains XGBoost + IsolationForest,
calibrates, explains, evaluates, and saves to the model registry.

Run directly:
    python -m src.models.train_pipeline

Or call from the scheduler (optionally after BTS ingestion):
    from src.models.train_pipeline import run_training_pipeline
    run_training_pipeline()
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import structlog

from config import settings
from src.models.anomaly import AnomalyDetector
from src.models.calibrator import ProbabilityCalibrator
from src.models.evaluator import walk_forward_report
from src.models.explainer import SHAPExplainer
from src.models.registry import ModelRegistry
from src.models.trainer import DisruptionTrainer

log = structlog.get_logger(__name__)


def run_training_pipeline(
    gold_path:      Path | None = None,
    n_optuna_trials: int = 30,
    use_mlflow:     bool = True,
) -> dict:
    """
    End-to-end training pipeline.

    Args:
        gold_path:       path to port_week_features.parquet (defaults to Gold layer)
        n_optuna_trials: Optuna search budget (reduce to 10 for quick testing)
        use_mlflow:      whether to log to MLflow

    Returns:
        dict with keys: version, metrics, backtest_report
    """
    # ── Load Gold ─────────────────────────────────────────────────────────────
    path = gold_path or (settings.gold_path / "port_week_features.parquet")
    if not path.exists():
        log.error("gold_not_found", path=str(path), hint="Run `python scheduler.py` first")
        raise FileNotFoundError(
            f"Gold table not found at {path}. "
            "Run the ingestion scheduler first to populate Bronze data, "
            "then run the feature pipeline: python -m src.features.pipeline"
        )

    df = pd.read_parquet(path)
    df["week_ending"] = pd.to_datetime(df["week_ending"])
    log.info("gold_loaded", rows=len(df), ports=df["port_id"].nunique())

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    trainer = DisruptionTrainer(n_optuna_trials=n_optuna_trials, use_mlflow=use_mlflow)
    metrics = trainer.train(df)

    # ── Fit Anomaly Detector ──────────────────────────────────────────────────
    detector = AnomalyDetector()
    detector.fit(df)

    # ── Calibrate Probabilities ───────────────────────────────────────────────
    calibrator = ProbabilityCalibrator()
    if trainer.model is not None and trainer.feature_columns:
        calibrator.fit_from_df(trainer.model, df, trainer.feature_columns)
        metrics["calibrator_fitted"] = calibrator.is_fitted
    else:
        log.warning("calibration_skipped", reason="model not trained")

    # ── Build SHAP Explainer ──────────────────────────────────────────────────
    explainer = None
    if trainer.model is not None and trainer.feature_columns:
        try:
            explainer = SHAPExplainer(trainer.model, trainer.feature_columns)
            log.info("shap_explainer_ready")
        except Exception as exc:
            log.warning("shap_explainer_failed", error=str(exc))

    # ── Backtest report ───────────────────────────────────────────────────────
    backtest_df = pd.DataFrame()
    if trainer.model is not None and trainer.feature_columns:
        backtest_df = walk_forward_report(df, trainer.model, trainer.feature_columns)
        if not backtest_df.empty:
            metrics["backtest_mean_pr_auc"] = round(backtest_df["pr_auc"].mean(), 4)
            metrics["backtest_mean_roc_auc"] = round(backtest_df["roc_auc"].mean(), 4)
            _print_backtest_report(backtest_df)

    # ── Save to registry ──────────────────────────────────────────────────────
    registry = ModelRegistry()
    version  = registry.save(
        trainer          = trainer,
        anomaly_detector = detector,
        calibrator       = calibrator,
        explainer        = explainer,
        metrics          = metrics,
    )

    log.info("pipeline_complete", version=version, metrics=metrics)
    return {"version": version, "metrics": metrics, "backtest_report": backtest_df}


# ─── CLI helpers ─────────────────────────────────────────────────────────────

def _print_backtest_report(report: pd.DataFrame) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table   = Table(title="Walk-Forward Backtest Results", show_lines=True)

    display_cols = [c for c in ["fold", "train_start", "test_start", "test_end",
                                 "n_train", "n_test", "pos_rate_test", "pr_auc", "roc_auc"]
                    if c in report.columns]
    for col in display_cols:
        table.add_column(col, style="cyan" if col == "pr_auc" else "white")

    for _, row in report.iterrows():
        table.add_row(*[str(row.get(c, "")) for c in display_cols])

    console.print(table)
    console.print(
        f"\n[bold]Mean PR-AUC:[/bold] {report['pr_auc'].mean():.4f}  |  "
        f"[bold]Mean ROC-AUC:[/bold] {report['roc_auc'].mean():.4f}\n"
    )


if __name__ == "__main__":
    import logging

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    result = run_training_pipeline(n_optuna_trials=20)
    print(f"\nVersion: {result['version']}")
    print(f"Metrics: {result['metrics']}")
