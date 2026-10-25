"""Unit tests for Phase 6 preprocessing components on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.feature_engineering.preprocessing import build_preprocessing_pipeline


@pytest.fixture
def synthetic_feature_matrix() -> pd.DataFrame:
    """Create a synthetic unscaled feature matrix with known missingness."""
    np.random.seed(42)
    n = 50
    cols = ["hold_mean", "hold_std", "latency_mean", "flight_mean", "hand_freq_L"]
    data = {
        "hold_mean": np.random.normal(120.0, 20.0, size=n),
        "hold_std": np.random.normal(40.0, 10.0, size=n),
        "latency_mean": np.random.normal(400.0, 80.0, size=n),
        "flight_mean": np.random.normal(150.0, 30.0, size=n),
        "hand_freq_L": np.random.uniform(0.3, 0.7, size=n),
    }
    df = pd.DataFrame(data)
    # Inject missing values
    df.loc[0:4, "hold_std"] = np.nan
    df.loc[10:14, "flight_mean"] = np.nan
    return df


def test_build_preprocessing_pipeline_unfitted(synthetic_feature_matrix):
    """Verify that build_preprocessing_pipeline returns an unfitted pipeline."""
    pipe = build_preprocessing_pipeline(
        scaler="standard",
        imputer_strategy="median",
        predictor_cols=synthetic_feature_matrix.columns.tolist(),
    )
    assert isinstance(pipe, Pipeline) or hasattr(pipe, "fit_transform")
    # Verify unfitted state before fit is called
    assert not hasattr(pipe, "classes_")


def test_pipeline_standard_scaling_and_imputation(synthetic_feature_matrix):
    """Test median imputation and standard scaling on synthetic training fold."""
    cols = synthetic_feature_matrix.columns.tolist()
    pipe = build_preprocessing_pipeline(
        scaler="standard",
        imputer_strategy="median",
        predictor_cols=cols,
    )

    transformed = pipe.fit_transform(synthetic_feature_matrix)

    # Output has no NaN values
    if isinstance(transformed, pd.DataFrame):
        assert transformed.isna().sum().sum() == 0
        means = transformed.mean()
        for m in means:
            assert m == pytest.approx(0.0, abs=1e-5)
    else:
        assert not np.isnan(transformed).any()
        assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-2)


def test_pipeline_robust_scaling(synthetic_feature_matrix):
    """Test robust scaling on synthetic training fold."""
    cols = synthetic_feature_matrix.columns.tolist()
    pipe = build_preprocessing_pipeline(
        scaler="robust",
        imputer_strategy="median",
        predictor_cols=cols,
    )
    transformed = pipe.fit_transform(synthetic_feature_matrix)

    if isinstance(transformed, pd.DataFrame):
        assert transformed.isna().sum().sum() == 0
        medians = transformed.median()
        for m in medians:
            assert m == pytest.approx(0.0, abs=1e-5)
    else:
        assert not np.isnan(transformed).any()
        medians = np.median(transformed, axis=0)
        assert np.allclose(medians, 0.0, atol=1e-5)


def test_pipeline_passthrough_scaling(synthetic_feature_matrix):
    """Test no scaling option (for tree-based models)."""
    cols = synthetic_feature_matrix.columns.tolist()
    pipe = build_preprocessing_pipeline(
        scaler="none",
        imputer_strategy="median",
        predictor_cols=cols,
    )
    transformed = pipe.fit_transform(synthetic_feature_matrix)

    if isinstance(transformed, pd.DataFrame):
        # Column values should match original imputed values
        assert transformed.isna().sum().sum() == 0
        assert transformed["hold_mean"].mean() == pytest.approx(synthetic_feature_matrix["hold_mean"].mean(), abs=1e-5)
    else:
        assert not np.isnan(transformed).any()


def test_pipeline_missing_indicator(synthetic_feature_matrix):
    """Verify that add_indicator appends binary missingness indicators."""
    cols = synthetic_feature_matrix.columns.tolist()
    pipe = build_preprocessing_pipeline(
        scaler="standard",
        imputer_strategy="median",
        add_indicator=True,
        predictor_cols=cols,
    )
    transformed = pipe.fit_transform(synthetic_feature_matrix)

    # 2 columns had missing values, so output feature count should be 5 + 2 = 7
    assert transformed.shape[1] == 7


def test_unsupported_scaler_raises():
    """Verify that invalid scaler options raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported scaler option"):
        build_preprocessing_pipeline(scaler="minmax")
