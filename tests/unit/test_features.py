"""
Unit tests for the feature engineering pipeline.

All tests use synthetic DataFrames — no filesystem, no network.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.features.hazard_features import compute_earthquake_features, compute_fire_features
from src.features.news_features import compute_news_features
from src.features.port_features import compute_port_features
from src.features.utils import (
    build_port_week_spine,
    compute_data_completeness,
    gdelt_sqldate_to_timestamp,
    pct_change_safe,
    rolling_zscore,
    week_ending_sunday,
)
from src.features.weather_features import compute_weather_features

# ─── utils ────────────────────────────────────────────────────────────────────

class TestWeekEndingSunday:
    def test_thursday_snaps_forward(self):
        result = week_ending_sunday("2024-04-11")  # Thursday
        assert result == pd.Timestamp("2024-04-14")  # Sunday

    def test_sunday_stays(self):
        result = week_ending_sunday("2024-04-14")
        assert result == pd.Timestamp("2024-04-14")

    def test_monday_snaps_forward_6_days(self):
        result = week_ending_sunday("2024-04-08")  # Monday
        assert result == pd.Timestamp("2024-04-14")

    def test_saturday_snaps_forward_1_day(self):
        result = week_ending_sunday("2024-04-13")  # Saturday
        assert result == pd.Timestamp("2024-04-14")


class TestBuildSpine:
    def test_correct_row_count(self):
        spine = build_port_week_spine(
            port_ids   = ["la_lb", "ny_nj"],
            start_date = "2024-01-01",
            end_date   = "2024-01-28",   # 4 full weeks
        )
        # 2 ports × 4 Sundays = 8 rows
        assert len(spine) == 8

    def test_all_ports_present(self):
        ports = ["la_lb", "ny_nj", "houston"]
        spine = build_port_week_spine(ports, "2024-01-01", "2024-01-07")
        assert set(spine["port_id"]) == set(ports)

    def test_week_ending_is_sunday(self):
        spine = build_port_week_spine(["la_lb"], "2024-04-08", "2024-04-14")
        for we in spine["week_ending"]:
            assert pd.Timestamp(we).day_of_week == 6  # Sunday


class TestPctChangeSafe:
    def test_normal_increase(self):
        s = pd.Series([100.0, 120.0])
        result = pct_change_safe(s)
        assert result.iloc[1] == pytest.approx(20.0)

    def test_division_by_zero_returns_zero(self):
        s = pd.Series([0.0, 10.0])
        result = pct_change_safe(s)
        assert result.iloc[1] == pytest.approx(0.0)

    def test_first_element_is_zero(self):
        s = pd.Series([50.0, 60.0])
        result = pct_change_safe(s)
        assert result.iloc[0] == pytest.approx(0.0)


class TestRollingZscore:
    def test_mean_value_gives_zero_zscore(self):
        s = pd.Series([10.0] * 10)
        z = rolling_zscore(s, window=5, min_periods=4)
        # When all values are equal, std=0 but we clip to 1e-6, so (10 - 10)/1e-6 = 0
        assert z.dropna().abs().max() < 1.0

    def test_high_outlier_gives_positive_zscore(self):
        s = pd.Series([10.0] * 10 + [100.0])
        z = rolling_zscore(s, window=10, min_periods=4)
        assert z.iloc[-1] > 2.0


class TestDataCompleteness:
    def test_all_sources_present(self):
        df = pd.DataFrame({
            "median_berthing_time_hrs": [5.0],
            "alert_count":              [2],
            "gdelt_event_count":        [3],
            "eq_count":                 [0],
            "fire_count":               [1],
        })
        indicators = {
            "bts":   "median_berthing_time_hrs",
            "nws":   "alert_count",
            "gdelt": "gdelt_event_count",
            "usgs":  "eq_count",
            "firms": "fire_count",
        }
        score = compute_data_completeness(df, indicators)
        assert score.iloc[0] == pytest.approx(1.0)

    def test_partial_sources(self):
        df = pd.DataFrame({
            "median_berthing_time_hrs": [5.0],
            "alert_count":              [np.nan],
            "gdelt_event_count":        [3],
            "eq_count":                 [np.nan],
            "fire_count":               [np.nan],
        })
        indicators = {
            "bts": "median_berthing_time_hrs",
            "nws": "alert_count",
            "gdelt": "gdelt_event_count",
            "usgs": "eq_count",
            "firms": "fire_count",
        }
        score = compute_data_completeness(df, indicators)
        assert score.iloc[0] == pytest.approx(0.4)  # 2/5


class TestGdeltSqldate:
    def test_valid_date(self):
        s = pd.Series(["20240415"])
        result = gdelt_sqldate_to_timestamp(s)
        assert result.iloc[0] == pd.Timestamp("2024-04-15")

    def test_invalid_date_coerces_to_nat(self):
        s = pd.Series(["invalid"])
        result = gdelt_sqldate_to_timestamp(s)
        assert pd.isna(result.iloc[0])


# ─── weather_features ─────────────────────────────────────────────────────────

def _make_alerts_df(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "alert_id":    [f"a{i}" for i in range(n)],
        "port_id":     ["la_lb"] * n,
        "event_type":  ["High Wind Warning", "Hurricane Warning", "Flood Advisory", "Dense Fog Advisory"],
        "severity":    ["Severe", "Extreme", "Moderate", "Minor"],
        "effective":   [pd.Timestamp("2024-04-10")] * n,
    })


class TestWeatherFeatures:
    def test_returns_expected_columns(self):
        result = compute_weather_features(_make_alerts_df())
        expected = {"port_id", "week_ending", "alert_count", "extreme_alert_count",
                    "has_hurricane_alert", "has_high_wind_alert", "has_flood_alert"}
        assert expected.issubset(set(result.columns))

    def test_correct_alert_count(self):
        result = compute_weather_features(_make_alerts_df())
        assert result["alert_count"].iloc[0] == 4

    def test_flags_are_correct(self):
        result = compute_weather_features(_make_alerts_df())
        row = result.iloc[0]
        assert row["has_hurricane_alert"] == 1
        assert row["has_high_wind_alert"] == 1
        assert row["has_flood_alert"] == 1
        assert row["has_fog_alert"] == 1

    def test_empty_input_returns_empty(self):
        result = compute_weather_features(pd.DataFrame())
        assert result.empty

    def test_extreme_count_is_correct(self):
        result = compute_weather_features(_make_alerts_df())
        assert result["extreme_alert_count"].iloc[0] == 1


# ─── news_features ────────────────────────────────────────────────────────────

def _make_gdelt_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "global_event_id": list(range(n)),
        "nearest_port_id": ["la_lb"] * n,
        "sql_date":        ["20240410"] * n,
        "event_root_code": ["14", "14", "17", "20", "14"],
        "goldstein_scale": [-5.0, -3.0, -2.0, -8.0, -1.0],
        "num_mentions":    [10, 5, 3, 20, 2],
        "num_articles":    [4, 2, 1, 8, 1],
        "avg_tone":        [-2.5, -1.5, -0.5, -4.0, -1.0],
    })


class TestNewsFeatures:
    def test_returns_expected_columns(self):
        result = compute_news_features(_make_gdelt_df())
        expected = {"port_id", "week_ending", "gdelt_event_count",
                    "gdelt_strike_count", "gdelt_goldstein_avg"}
        assert expected.issubset(set(result.columns))

    def test_event_count_correct(self):
        result = compute_news_features(_make_gdelt_df())
        assert result["gdelt_event_count"].iloc[0] == 5

    def test_strike_count_correct(self):
        result = compute_news_features(_make_gdelt_df())
        assert result["gdelt_strike_count"].iloc[0] == 3  # root codes "14"

    def test_goldstein_is_negative(self):
        result = compute_news_features(_make_gdelt_df())
        assert result["gdelt_goldstein_avg"].iloc[0] < 0

    def test_empty_input_returns_empty(self):
        assert compute_news_features(pd.DataFrame()).empty


# ─── port_features ────────────────────────────────────────────────────────────

def _make_bts_df(n_weeks: int = 60, port_id: str = "la_lb") -> pd.DataFrame:
    """Synthetic weekly BTS records starting from 2023-01-01."""
    weeks = pd.date_range("2023-01-01", periods=n_weeks, freq="W-SUN")
    np.random.seed(42)
    berthing = 20.0 + np.random.randn(n_weeks) * 3   # mean ~20h, std ~3h
    # Inject a spike at week 50 for disruption testing
    if n_weeks > 50:
        berthing[50] = 35.0

    return pd.DataFrame({
        "port_id":                 [port_id] * n_weeks,
        "port_name":               ["Los Angeles"] * n_weeks,
        "week_ending":             weeks,
        "median_berthing_time_hrs": berthing,
        "avg_berthing_time_hrs":    berthing + 2.0,
        "vessel_call_count":        np.random.randint(40, 80, n_weeks),
        "ingested_at":              [datetime.utcnow()] * n_weeks,
        "dataset_id":               ["y7iz-hhid"] * n_weeks,
    })


class TestPortFeatures:
    def test_returns_expected_columns(self):
        result = compute_port_features(_make_bts_df())
        expected = {
            "port_id", "week_ending", "median_berthing_time_hrs",
            "berthing_time_lag_1w", "is_disruption_week",
            "week_of_year", "berthing_time_z_score_52w",
        }
        assert expected.issubset(set(result.columns))

    def test_sorted_by_port_and_week(self):
        result = compute_port_features(_make_bts_df())
        assert result["week_ending"].is_monotonic_increasing

    def test_disruption_label_at_spike_week(self):
        result = compute_port_features(_make_bts_df(n_weeks=60))
        # Week index 50 has a spike to 35h — should be labelled as disruption
        spike_rows = result[result["median_berthing_time_hrs"] > 30]
        assert len(spike_rows) > 0
        assert spike_rows["is_disruption_week"].iloc[0] == 1

    def test_early_rows_have_null_labels(self):
        """First few rows can't have labels because rolling baseline not available."""
        result = compute_port_features(_make_bts_df(n_weeks=60))
        # After min_periods=4, rows should start getting labels
        assert result["is_disruption_week"].isna().sum() > 0

    def test_lag_1w_is_shifted_by_one(self):
        result = compute_port_features(_make_bts_df(n_weeks=20))
        # lag_1w[i] should equal median_berthing_time[i-1]
        for i in range(1, 5):
            lag  = result["berthing_time_lag_1w"].iloc[i]
            prev = result["median_berthing_time_hrs"].iloc[i - 1]
            if pd.notna(lag) and pd.notna(prev):
                assert lag == pytest.approx(prev, abs=1e-9)

    def test_empty_input_returns_empty(self):
        assert compute_port_features(pd.DataFrame()).empty

    def test_calendar_features(self):
        result = compute_port_features(_make_bts_df(n_weeks=10))
        assert result["week_of_year"].between(1, 53).all()
        assert result["month"].between(1, 12).all()
        assert result["is_q4"].isin([0, 1]).all()

    def test_multiple_ports_independent_labels(self):
        """Labels for Port A should not be contaminated by Port B data."""
        bts_a = _make_bts_df(n_weeks=60, port_id="la_lb")
        bts_b = _make_bts_df(n_weeks=60, port_id="ny_nj")
        # Give Port B a much higher baseline
        bts_b["median_berthing_time_hrs"] *= 3

        combined = pd.concat([bts_a, bts_b], ignore_index=True)
        result   = compute_port_features(combined)

        la_labels = result[result["port_id"] == "la_lb"]["is_disruption_week"].dropna()
        ny_labels = result[result["port_id"] == "ny_nj"]["is_disruption_week"].dropna()

        # Both ports should have ~25% disruption rate (75th pct baseline)
        assert 0.1 < la_labels.mean() < 0.6
        assert 0.1 < ny_labels.mean() < 0.6


# ─── hazard_features ──────────────────────────────────────────────────────────

def _make_usgs_df() -> pd.DataFrame:
    return pd.DataFrame({
        "usgs_id":             ["us001", "us002", "us003"],
        "nearest_port_id":     ["la_lb", "la_lb", "seattle"],
        "magnitude":           [5.2, 4.8, 6.1],
        "depth_km":            [10.0, 25.0, 5.0],
        "tsunami_flag":        [False, False, True],
        "occurred_at":         [pd.Timestamp("2024-04-10")] * 3,
        "distance_to_port_km": [45.0, 80.0, 120.0],
    })


def _make_firms_df() -> pd.DataFrame:
    return pd.DataFrame({
        "nearest_port_id":     ["la_lb", "la_lb", "la_lb"],
        "lat":                 [34.1, 34.2, 34.3],
        "lon":                 [-118.1, -118.2, -118.3],
        "brightness":          [320.0, 340.0, 310.0],
        "frp":                 [15.0, 25.0, 8.0],
        "confidence":          ["high", "nominal", "high"],
        "satellite":           ["NOAA-20"] * 3,
        "acquired_at":         [pd.Timestamp("2024-04-10")] * 3,
        "distance_to_port_km": [60.0, 75.0, 90.0],
    })


class TestEarthquakeFeatures:
    def test_correct_count(self):
        result = compute_earthquake_features(_make_usgs_df())
        la_row = result[result["port_id"] == "la_lb"]
        assert la_row["eq_count"].iloc[0] == 2

    def test_max_magnitude(self):
        result = compute_earthquake_features(_make_usgs_df())
        la_row = result[result["port_id"] == "la_lb"]
        assert la_row["eq_max_magnitude"].iloc[0] == pytest.approx(5.2)

    def test_tsunami_flag_set(self):
        result = compute_earthquake_features(_make_usgs_df())
        sea_row = result[result["port_id"] == "seattle"]
        assert sea_row["eq_tsunami_flag"].iloc[0] == 1

    def test_energy_proxy_is_positive(self):
        result = compute_earthquake_features(_make_usgs_df())
        assert (result["eq_energy_proxy"] > 0).all()

    def test_empty_returns_empty(self):
        assert compute_earthquake_features(pd.DataFrame()).empty


class TestFireFeatures:
    def test_correct_count(self):
        result = compute_fire_features(_make_firms_df())
        assert result["fire_count"].iloc[0] == 3

    def test_high_confidence_count(self):
        result = compute_fire_features(_make_firms_df())
        assert result["fire_high_confidence_count"].iloc[0] == 2

    def test_max_frp(self):
        result = compute_fire_features(_make_firms_df())
        assert result["fire_max_frp"].iloc[0] == pytest.approx(25.0)

    def test_empty_returns_empty(self):
        assert compute_fire_features(pd.DataFrame()).empty
