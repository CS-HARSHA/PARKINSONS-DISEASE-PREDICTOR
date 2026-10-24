"""Unit tests for feature table quality audit and integrity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.feature_schema import get_predictor_columns
from src.feature_engineering.quality import (
    generate_feature_quality_summary,
    validate_feature_table,
    validate_quality_table,
)


@pytest.fixture
def clean_synthetic_feature_table() -> pd.DataFrame:
    """Create a clean synthetic feature table matching the schema for 10 participants."""
    predictors = get_predictor_columns()
    n = 10
    data = {"participant_id": [f"USER{i:03d}" for i in range(n)], "parkinsons_label": [1, 0] * 5}
    for p in predictors:
        data[p] = np.random.normal(100.0, 15.0, size=n)

    return pd.DataFrame(data)


def test_validate_feature_table_clean(clean_synthetic_feature_table):
    """Verify that a compliant feature table passes all validation checks."""
    res = validate_feature_table(clean_synthetic_feature_table, expected_count=10)
    assert res["passed"]
    assert res["row_count"] == 10
    assert len(res["issues"]) == 0


def test_validate_feature_table_row_count_mismatch(clean_synthetic_feature_table):
    """Verify detection of cohort row count mismatch."""
    res = validate_feature_table(clean_synthetic_feature_table, expected_count=597)
    assert not res["passed"]
    assert any("Row count mismatch" in iss for iss in res["issues"])


def test_validate_feature_table_duplicate_ids(clean_synthetic_feature_table):
    """Verify detection of duplicate participant IDs."""
    df = clean_synthetic_feature_table.copy()
    df.loc[1, "participant_id"] = df.loc[0, "participant_id"]
    res = validate_feature_table(df, expected_count=10)
    assert not res["passed"]
    assert any("Duplicate participant IDs found" in iss for iss in res["issues"])


def test_validate_feature_table_catches_date_corruption(clean_synthetic_feature_table):
    """Verify detection of exact integer YYMMDD date corruption."""
    df = clean_synthetic_feature_table.copy()
    df.loc[0, "hold_max"] = 171226.0  # Exact YYMMDD integer
    res = validate_feature_table(df, expected_count=10)
    assert not res["passed"]
    assert any("Date corruption signature" in iss for iss in res["issues"])


def test_validate_feature_table_catches_infinite_values(clean_synthetic_feature_table):
    """Verify detection of non-finite/infinite values."""
    df = clean_synthetic_feature_table.copy()
    df.loc[0, "hold_mean"] = np.inf
    res = validate_feature_table(df, expected_count=10)
    assert not res["passed"]
    assert any("Infinite values detected" in iss for iss in res["issues"])


def test_generate_feature_quality_summary(clean_synthetic_feature_table):
    """Verify that quality summary generates status and missingness per column."""
    summary_df = generate_feature_quality_summary(clean_synthetic_feature_table)
    assert len(summary_df) == 153
    assert set(summary_df.columns) == {
        "feature_name", "data_type", "missing_count", "missing_pct",
        "n_unique", "is_constant", "n_gt_100k", "status"
    }
