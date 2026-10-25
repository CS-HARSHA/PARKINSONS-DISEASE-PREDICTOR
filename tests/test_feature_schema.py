"""Unit tests for Phase 6 feature schema, allowlists, and taxonomy."""

from __future__ import annotations

import pytest

from src.feature_engineering.feature_schema import (
    ALLOWED_DIRECTIONS,
    ALLOWED_HANDS,
    get_feature_definitions,
    get_predictor_columns,
    get_quality_columns,
    validate_predictor_allowlist,
)


def test_feature_catalog_integrity():
    """Verify that all features in catalog have documented names, units, and families."""
    catalog = get_feature_definitions()
    assert len(catalog) > 100

    names = set()
    for feat in catalog:
        assert feat.name not in names, f"Duplicate feature name: {feat.name}"
        names.add(feat.name)
        assert len(feat.name) > 0
        assert len(feat.family) > 0
        assert len(feat.unit) > 0
        assert len(feat.description) > 0


def test_predictor_columns_count():
    """Verify exact count of predictor columns."""
    predictors = get_predictor_columns()
    assert len(predictors) == 153  # 11*3 overall + 3 hand + 9 dir + 36*3 directional


def test_allowlist_validation_clean():
    """Verify that a valid predictor list passes allowlist check."""
    predictors = get_predictor_columns()
    is_valid, violations = validate_predictor_allowlist(predictors)
    assert is_valid
    assert len(violations) == 0


def test_allowlist_validation_catches_leakage():
    """Verify that allowlist detects prohibited columns (IDs, labels, covariates)."""
    contaminated = [
        "hold_mean",
        "participant_id",
        "parkinsons_label",
        "Age",
        "Gender",
        "UPDRS",
        "Tremor",
    ]
    is_valid, violations = validate_predictor_allowlist(contaminated)
    assert not is_valid
    assert "participant_id" in violations
    assert "parkinsons_label" in violations
    assert "Age" in violations
    assert "Gender" in violations
    assert "UPDRS" in violations


def test_quality_columns_inventory():
    """Verify quality metadata columns."""
    qual_cols = get_quality_columns()
    assert "participant_id" in qual_cols
    assert "total_records" in qual_cols
    assert "n_warned_records" in qual_cols
    assert "data_coverage_days" in qual_cols
