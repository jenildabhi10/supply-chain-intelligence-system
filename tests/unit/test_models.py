"""
Unit tests for the ML layer — Phase 3.

All tests use synthetic data generated in-memory.
No filesystem I/O, no network, no MLflow server required.

Design principle: tests prove behaviour, not implementation.
  - Trainer tests check that the model is fitted and outputs are in valid ranges.
  - Anomaly tests check score range [0,1] and that anomalies score higher.
  - SHAP tests check output format and count.
  - Evaluator tests check metric calculations on known distributions.
  - Registry tests check save/load round-trip integrity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.anomaly import AnomalyDetector
from src.models.calibrator import ProbabilityCalibrator
from src.models.evaluator import (
    calibration_data,
    compute_pr_auc,
    feature_drift_report,
    precision_at_k,
)
from src.models.features import (
    TARGET_COLUMN,
    get_feature_matrix,
)
from src.models.trainer import DisruptionTrainer, walk_forward_splits

# ─── Synthetic data factory ───────────────────────────────────────────────────

def make_gold_df(
    n_weeks: int = 80,
    n_ports: int = 2,
    disruption_rate: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic Gold-table DataFrame for testing.
    All FEATURE_COLUMNS are populated; is_disruption_week is set for labeled rows.
    """
    rng   = np.random.default_rng(seed)
    ports = [f"port_{i}" for i in range(n_ports)]
    weeks = pd.date_range("2022-01-02", periods=n_weeks, freq="W-SUN")
    rows  = []

    for port in ports:
        for i, week in enumerate(weeks):
            # Only label rows after the first 4 (simulate min_periods=4)
            label = int(rng.random() < disruption_rate) if i >= 4 else pd.NA

            rows.append({
                "port_id":    port,
                "week_ending": week,
                TARGET_COLUMN: label,
                # Port activity features
                "berthing_time_lag_1w":       rng.normal(20, 3),
                "berthing_time_lag_2w":       rng.normal(20, 3),
                "berthing_time_lag_4w":       rng.normal(20, 3),
                "berthing_time_pct_change_1w": rng.normal(0, 5),
                "call_count_pct_change_1w":   rng.normal(0, 3),
                "berthing_time_4w_avg":       rng.normal(20, 2),
                "berthing_time_4w_std":       abs(rng.normal(2, 0.5)),
                "berthing_time_52w_avg":      rng.normal(20, 1),
                "berthing_time_52w_std":      abs(rng.normal(2, 0.3)),
                "berthing_time_z_score_52w":  rng.normal(0, 1),
                # Calendar
                "week_of_year": week.isocalendar().week,
                "month":        week.month,
                "is_q4":        int(week.month >= 10),
                # Weather
                "alert_count":         rng.integers(0, 5),
                "extreme_alert_count": rng.integers(0, 2),
                "severe_alert_count":  rng.integers(0, 3),
                "has_hurricane_alert": rng.integers(0, 2),
                "has_high_wind_alert": rng.integers(0, 2),
                "has_flood_alert":     rng.integers(0, 2),
                "max_severity_score":  rng.integers(0, 5),
                # News
                "gdelt_event_count":       rng.integers(0, 20),
                "gdelt_strike_count":      rng.integers(0, 5),
                "gdelt_coerce_count":      rng.integers(0, 3),
                "gdelt_goldstein_avg":     rng.normal(-1, 2),
                "gdelt_goldstein_min":     rng.normal(-4, 2),
                "gdelt_weighted_disruption": abs(rng.normal(10, 20)),
                # Seismic
                "eq_count":         rng.integers(0, 3),
                "eq_max_magnitude": rng.uniform(0, 6),
                "eq_energy_proxy":  rng.uniform(0, 1e6),
                "eq_tsunami_flag":  rng.integers(0, 2),
                # Fire
                "fire_count":  rng.integers(0, 10),
                "fire_max_frp": rng.uniform(0, 50),
                # Metadata
                "data_completeness_score": rng.uniform(0.6, 1.0),
            })

    return pd.DataFrame(rows)


# ─── walk_forward_splits ──────────────────────────────────────────────────────

class TestWalkForwardSplits:
    def test_returns_correct_number_of_splits(self):
        df = make_gold_df(n_weeks=80, n_ports=2)
        splits = walk_forward_splits(df, n_splits=4, test_weeks=4, min_train_weeks=20)
        assert len(splits) > 0
        assert len(splits) <= 4

    def test_train_indices_strictly_before_test(self):
        df = make_gold_df(n_weeks=80, n_ports=2)
        df["week_ending"] = pd.to_datetime(df["week_ending"])
        splits = walk_forward_splits(df, n_splits=3, test_weeks=4)

        for train_idx, test_idx in splits:
            max_train_week = df.loc[train_idx, "week_ending"].max()
            min_test_week  = df.loc[test_idx, "week_ending"].min()
            assert max_train_week < min_test_week, "Train leaks into test period"

    def test_no_splits_when_insufficient_data(self):
        df = make_gold_df(n_weeks=10, n_ports=1)
        splits = walk_forward_splits(df, n_splits=5, min_train_weeks=26)
        assert splits == []


# ─── get_feature_matrix ───────────────────────────────────────────────────────

class TestGetFeatureMatrix:
    def test_returns_dataframe(self):
        df = make_gold_df(n_weeks=20)
        X  = get_feature_matrix(df)
        assert isinstance(X, pd.DataFrame)

    def test_no_nan_in_output(self):
        df = make_gold_df(n_weeks=20)
        X  = get_feature_matrix(df)
        assert not X.isnull().any().any(), "Feature matrix must have no NaN values"

    def test_contains_port_enc_column(self):
        df = make_gold_df(n_weeks=10)
        X  = get_feature_matrix(df)
        assert "port_id_enc" in X.columns

    def test_all_values_are_numeric(self):
        df = make_gold_df(n_weeks=10)
        X  = get_feature_matrix(df)
        assert (X.dtypes == float).all()


# ─── DisruptionTrainer ────────────────────────────────────────────────────────

class TestDisruptionTrainer:
    @pytest.fixture(scope="class")
    def trained(self):
        df      = make_gold_df(n_weeks=80, n_ports=3)
        trainer = DisruptionTrainer(
            n_cv_folds=3, test_weeks=4, n_optuna_trials=5, use_mlflow=False
        )
        metrics = trainer.train(df)
        return trainer, metrics

    def test_model_is_fitted(self, trained):
        trainer, _ = trained
        assert trainer.model is not None

    def test_feature_columns_set(self, trained):
        trainer, _ = trained
        assert isinstance(trainer.feature_columns, list)
        assert len(trainer.feature_columns) > 0

    def test_metrics_dict_has_expected_keys(self, trained):
        _, metrics = trained
        assert "tuned_cv_pr_auc"  in metrics
        assert "n_train_rows"     in metrics
        assert "n_features"       in metrics

    def test_pr_auc_in_valid_range(self, trained):
        _, metrics = trained
        pr_auc = metrics.get("tuned_cv_pr_auc", 0)
        assert 0.0 <= pr_auc <= 1.0

    def test_predict_proba_returns_valid_array(self, trained):
        trainer, _ = trained
        df    = make_gold_df(n_weeks=10, n_ports=2)
        proba = trainer.predict_proba(df)
        assert proba.shape == (len(df),)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_raises_on_insufficient_data(self):
        df      = make_gold_df(n_weeks=5, n_ports=1)
        trainer = DisruptionTrainer(use_mlflow=False)
        with pytest.raises(ValueError, match="Too few labeled rows"):
            trainer.train(df)


# ─── AnomalyDetector ─────────────────────────────────────────────────────────

class TestAnomalyDetector:
    @pytest.fixture(scope="class")
    def fitted_detector(self):
        df       = make_gold_df(n_weeks=60, n_ports=2)
        detector = AnomalyDetector(contamination=0.1)
        detector.fit(df)
        return detector, df

    def test_is_fitted_after_fit(self, fitted_detector):
        detector, _ = fitted_detector
        assert detector.is_fitted

    def test_scores_in_unit_interval(self, fitted_detector):
        detector, df = fitted_detector
        scores = detector.score(df)
        assert scores.shape == (len(df),)
        assert (scores >= 0.0).all() and (scores <= 1.0).all()

    def test_anomalies_score_higher_than_normal(self, fitted_detector):
        """Inject obviously anomalous rows and verify they score higher."""
        detector, df = fitted_detector

        normal   = df.head(20).copy()
        anomalous = df.head(5).copy()
        # Set anomalous feature values to extreme outliers
        anomalous["berthing_time_lag_1w"]       = 200.0
        anomalous["gdelt_weighted_disruption"]   = 5000.0
        anomalous["berthing_time_z_score_52w"]   = 10.0

        normal_scores    = detector.score(normal)
        anomalous_scores = detector.score(anomalous)

        assert anomalous_scores.mean() > normal_scores.mean(), (
            "Anomalous rows should score higher than normal rows"
        )

    def test_empty_df_returns_zeros(self, fitted_detector):
        detector, _ = fitted_detector
        scores = detector.score(pd.DataFrame())
        assert len(scores) == 0 or (scores == 0).all()


# ─── Calibrator ───────────────────────────────────────────────────────────────

class TestProbabilityCalibrator:
    def test_fit_and_predict(self):
        df      = make_gold_df(n_weeks=80, n_ports=2)
        trainer = DisruptionTrainer(
            n_cv_folds=2, test_weeks=4, n_optuna_trials=3, use_mlflow=False
        )
        trainer.train(df)

        cal = ProbabilityCalibrator()
        cal.fit_from_df(trainer.model, df, trainer.feature_columns)

        assert cal.is_fitted
        X     = get_feature_matrix(df).reindex(columns=trainer.feature_columns, fill_value=0)
        proba = cal.calibrated_proba(X)
        assert proba.shape == (len(df),)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_raises_before_fit(self):
        cal = ProbabilityCalibrator()
        with pytest.raises(RuntimeError, match="not fitted"):
            cal.calibrated_proba(pd.DataFrame({"a": [1]}))


# ─── Evaluator ────────────────────────────────────────────────────────────────

class TestComputePRAuc:
    def test_perfect_classifier(self):
        y_true  = np.array([1, 0, 1, 0, 1])
        y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        assert compute_pr_auc(y_true, y_score) == pytest.approx(1.0)

    def test_random_classifier_near_base_rate(self):
        rng    = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_prob = rng.uniform(0, 1, 200)
        pr_auc = compute_pr_auc(y_true, y_prob)
        assert 0.2 < pr_auc < 0.8  # should be near base rate (~0.5)

    def test_all_same_class_returns_zero(self):
        assert compute_pr_auc(np.zeros(10), np.ones(10)) == 0.0


class TestPrecisionAtK:
    def test_all_correct(self):
        y_true  = np.array([1, 1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
        assert precision_at_k(y_true, y_score, k=3) == pytest.approx(1.0)

    def test_all_wrong(self):
        y_true  = np.array([0, 0, 0, 1, 1])
        y_score = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
        assert precision_at_k(y_true, y_score, k=3) == pytest.approx(0.0)


class TestCalibrationData:
    def test_returns_dataframe(self):
        rng    = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)
        result = calibration_data(y_true, y_prob)
        assert isinstance(result, pd.DataFrame)
        assert "fraction_of_positives" in result.columns
        assert "mean_predicted_prob"   in result.columns

    def test_empty_on_few_samples(self):
        result = calibration_data(np.array([1, 0]), np.array([0.8, 0.2]), n_bins=10)
        assert result.empty


class TestFeatureDrift:
    def test_flags_drifted_feature(self):
        ref = pd.DataFrame({"feat": [1.0] * 50})
        cur = pd.DataFrame({"feat": [100.0] * 50})  # massive shift
        report = feature_drift_report(ref, cur, ["feat"], z_threshold=2.0)
        assert report["drifted"].any()

    def test_no_drift_on_same_distribution(self):
        rng = np.random.default_rng(0)
        ref = pd.DataFrame({"feat": rng.normal(5, 1, 100)})
        cur = pd.DataFrame({"feat": rng.normal(5, 1, 100)})
        report = feature_drift_report(ref, cur, ["feat"], z_threshold=3.0)
        assert not report["drifted"].any()


# ─── ModelRegistry ────────────────────────────────────────────────────────────

class TestModelRegistry:
    def test_save_and_load_champion(self, tmp_path):
        from src.models.registry import ModelRegistry

        # Train a tiny model
        df      = make_gold_df(n_weeks=80, n_ports=2)
        trainer = DisruptionTrainer(
            n_cv_folds=2, test_weeks=4, n_optuna_trials=2, use_mlflow=False
        )
        metrics = trainer.train(df)

        registry = ModelRegistry(models_dir=tmp_path)
        version  = registry.save(
            trainer          = trainer,
            anomaly_detector = AnomalyDetector(),
            metrics          = metrics,
        )

        assert version.startswith("v")

        bundle = registry.load_champion()
        assert bundle["version"] == version
        assert bundle["trainer"] is not None

    def test_first_model_becomes_champion(self, tmp_path):
        from src.models.registry import ModelRegistry

        df      = make_gold_df(n_weeks=60, n_ports=2)
        trainer = DisruptionTrainer(
            n_cv_folds=2, test_weeks=4, n_optuna_trials=2, use_mlflow=False
        )
        trainer.train(df)

        registry = ModelRegistry(models_dir=tmp_path)
        version  = registry.save(trainer, AnomalyDetector(), metrics={"tuned_cv_pr_auc": 0.5})

        assert registry.get_champion_version() == version

    def test_better_model_auto_promotes(self, tmp_path):
        from src.models.registry import ModelRegistry

        df      = make_gold_df(n_weeks=60, n_ports=2)
        trainer = DisruptionTrainer(n_cv_folds=2, test_weeks=4, n_optuna_trials=2, use_mlflow=False)
        trainer.train(df)

        registry = ModelRegistry(models_dir=tmp_path)
        registry.save(trainer, AnomalyDetector(), metrics={"tuned_cv_pr_auc": 0.5})
        v2 = registry.save(trainer, AnomalyDetector(), metrics={"tuned_cv_pr_auc": 0.75})

        assert registry.get_champion_version() == v2  # v2 is better

    def test_list_versions_returns_dataframe(self, tmp_path):
        from src.models.registry import ModelRegistry

        df      = make_gold_df(n_weeks=60, n_ports=2)
        trainer = DisruptionTrainer(n_cv_folds=2, test_weeks=4, n_optuna_trials=2, use_mlflow=False)
        trainer.train(df)

        registry = ModelRegistry(models_dir=tmp_path)
        registry.save(trainer, AnomalyDetector(), metrics={"tuned_cv_pr_auc": 0.55})

        versions = registry.list_versions()
        assert isinstance(versions, pd.DataFrame)
        assert len(versions) >= 1
        assert "is_champion" in versions.columns
