"""Unit tests for Phase 6 feature extraction logic and safeguards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.config import ExtractionConfig
from src.feature_engineering.feature_extraction import (
    compute_directional_timing,
    compute_timing_statistics,
    extract_participant_features,
    extract_participant_quality,
)
from src.feature_engineering.feature_schema import ALLOWED_DIRECTIONS, ALLOWED_HANDS, get_predictor_columns


@pytest.fixture
def extraction_config() -> ExtractionConfig:
    return ExtractionConfig(
        min_obs_skewness=20,
        min_obs_kurtosis=30,
        min_obs_cv=10,
        cv_mean_epsilon=1e-6,
    )


@pytest.fixture
def synthetic_events_df() -> pd.DataFrame:
    """Create a controlled synthetic event dataframe for one participant."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "user_key": ["TESTUSER01"] * n,
        "event_time": pd.date_range("2017-01-01 10:00:00", periods=n, freq="s"),
        "hand": np.random.choice(["L", "R", "S"], size=n, p=[0.5, 0.35, 0.15]),
        "hold_ms": np.random.normal(120.0, 20.0, size=n).clip(min=10.0),
        "direction": np.random.choice(["LL", "LR", "RL", "RR"], size=n),
        "latency_ms": np.random.normal(350.0, 50.0, size=n).clip(min=50.0),
        "flight_ms": np.random.normal(100.0, 30.0, size=n),
        "warning_codes": [""] * (n - 2) + ["W01_BACKWARD_TIMESTAMP", "W02_DUPLICATE_RECORD"],
    })


def test_compute_timing_statistics_normal(extraction_config):
    """Test timing statistics computation on a known normal series."""
    s = pd.Series([100.0, 110.0, 120.0, 130.0, 140.0] * 10)  # 50 observations
    stats = compute_timing_statistics(s, "hold", extraction_config)

    assert stats["hold_mean"] == pytest.approx(120.0, abs=0.01)
    assert stats["hold_median"] == pytest.approx(120.0, abs=0.01)
    assert stats["hold_min"] == 100.0
    assert stats["hold_max"] == 140.0
    assert stats["hold_iqr"] > 0
    assert not np.isnan(stats["hold_skew"])
    assert not np.isnan(stats["hold_kurt"])
    assert not np.isnan(stats["hold_cv"])


def test_compute_timing_statistics_insufficient_sample(extraction_config):
    """Verify that sample size < threshold returns NaN for skewness, kurtosis, CV."""
    s = pd.Series([100.0, 110.0, 120.0])  # Only 3 observations
    stats = compute_timing_statistics(s, "hold", extraction_config)

    assert stats["hold_mean"] == pytest.approx(110.0, abs=0.01)
    assert np.isnan(stats["hold_skew"])
    assert np.isnan(stats["hold_kurt"])
    assert np.isnan(stats["hold_cv"])


def test_compute_timing_statistics_empty(extraction_config):
    """Verify that empty series safely returns all NaN statistics."""
    s = pd.Series([], dtype=float)
    stats = compute_timing_statistics(s, "hold", extraction_config)
    for v in stats.values():
        assert np.isnan(v)


def test_compute_timing_statistics_near_zero_mean(extraction_config):
    """Verify safeguard against zero or near-zero mean in CV computation."""
    s = pd.Series([0.0000001, -0.0000001] * 20)
    stats = compute_timing_statistics(s, "flight", extraction_config)
    assert np.isnan(stats["flight_cv"])


def test_extract_participant_features_structure(synthetic_events_df, extraction_config):
    """Verify that single participant event aggregation produces exactly the expected predictor set."""
    features = extract_participant_features(synthetic_events_df, extraction_config)
    expected_predictors = set(get_predictor_columns())

    assert set(features.keys()) == expected_predictors
    assert len(features) == len(expected_predictors)

    # Check hand frequency sums to approximately 1.0
    total_hand_freq = sum(features[f"hand_freq_{h}"] for h in ALLOWED_HANDS)
    assert total_hand_freq == pytest.approx(1.0, abs=0.01)


def test_negative_flight_times_preserved(extraction_config):
    """Verify that negative flight times are properly preserved in summary statistics."""
    # Flight time with rollover keypresses (negative flight times)
    s = pd.Series([-50.0, -20.0, 10.0, 30.0, 60.0] * 10)
    df = pd.DataFrame({
        "user_key": ["TEST"] * len(s),
        "event_time": pd.date_range("2017-01-01", periods=len(s), freq="s"),
        "hand": ["L"] * len(s),
        "hold_ms": [100.0] * len(s),
        "direction": ["LL"] * len(s),
        "latency_ms": [200.0] * len(s),
        "flight_ms": s,
        "warning_codes": [""] * len(s),
    })

    feats = extract_participant_features(df, extraction_config)
    assert feats["flight_min"] == -50.0
    assert feats["flight_mean"] == pytest.approx(6.0, abs=0.01)
    assert feats["flight_median_LL"] == 10.0
    assert feats["flight_mean_LL"] == pytest.approx(6.0, abs=0.01)


def test_extract_participant_quality_metadata(synthetic_events_df):
    """Verify participant quality metadata extraction and warning counts."""
    quality = extract_participant_quality(synthetic_events_df, "TESTUSER01", n_files=3)

    assert quality["participant_id"] == "TESTUSER01"
    assert quality["total_records"] == 100
    assert quality["n_files"] == 3
    assert quality["n_event_dates"] == 1
    assert quality["n_warned_records"] == 2
    assert quality["n_w01_backward_timestamp"] == 1
    assert quality["n_w02_duplicate_record"] == 1
    assert quality["n_w04_extreme_hold"] == 0
    assert quality["data_coverage_days"] == 1
