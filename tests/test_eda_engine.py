"""Targeted tests for Phase 5 EDA Engine and Fail-Closed Readiness Gate."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.eda_engine import (
    evaluate_readiness_gate,
    enforce_readiness_gate,
    build_feature_inventory,
    compute_missingness_summary,
    compute_descriptive_statistics,
    screen_outliers_and_anomalies,
    analyze_feature_correlations,
    compute_training_class_summary,
)


@pytest.fixture
def mock_valid_eda_environment(tmp_path: Path):
    """Create a fully compliant synthetic dataset matching the 597 cohort and 153 feature schema."""
    n_total = 597
    n_train = 477
    n_test = 120
    n_features = 153

    # Generate IDs
    p_ids = [f"PART_{i:04d}" for i in range(1, n_total + 1)]
    train_ids = p_ids[:n_train]
    test_ids = p_ids[n_train:]

    # Write split files
    train_csv = tmp_path / "train_ids.csv"
    test_csv = tmp_path / "test_ids.csv"
    pd.DataFrame({"participant_id": train_ids}).to_csv(train_csv, index=False)
    pd.DataFrame({"participant_id": test_ids}).to_csv(test_csv, index=False)

    # Generate 153 predictors across families
    rng = np.random.default_rng(42)
    feature_dict = {
        "participant_id": p_ids,
        "parkinsons_label": [1 if i % 4 != 0 else 0 for i in range(n_total)],
    }

    # Add Hold Time features
    for stat in ["mean", "std", "median", "iqr", "skew", "kurt", "cv"]:
        feature_dict[f"hold_{stat}"] = rng.normal(85.0, 15.0, size=n_total)

    # Add Latency features
    for stat in ["mean", "std", "median", "iqr", "skew", "kurt"]:
        feature_dict[f"latency_{stat}"] = rng.normal(350.0, 50.0, size=n_total)

    # Add Flight Time features using the observed nonnegative interval range.
    feature_dict["flight_min"] = rng.uniform(1.0, 150.0, size=n_total)
    for stat in ["mean", "std", "median", "iqr"]:
        feature_dict[f"flight_{stat}"] = rng.normal(40.0, 20.0, size=n_total)

    # Fill remaining features to reach exactly 153 predictors
    current_count = len(feature_dict) - 2
    for i in range(current_count, n_features):
        feature_dict[f"trans_metric_{i:03d}"] = rng.uniform(0.01, 1.0, size=n_total)

    df_clean = pd.DataFrame(feature_dict)
    parquet_path = tmp_path / "participant_features_raw.parquet"
    df_clean.to_parquet(parquet_path, index=False)

    return {
        "features_path": parquet_path,
        "train_ids_path": train_csv,
        "test_ids_path": test_csv,
        "df_clean": df_clean,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }


def test_evaluate_readiness_gate_passes_on_valid_dataset(mock_valid_eda_environment):
    """Test that all 8 checks pass on a compliant, uncorrupted dataset."""
    env = mock_valid_eda_environment
    res = evaluate_readiness_gate(
        features_path=env["features_path"],
        train_ids_path=env["train_ids_path"],
        test_ids_path=env["test_ids_path"],
        eligible_count=597,
        expected_train_count=477,
        expected_test_count=120,
        expected_feature_count=153,
        allow_legacy_fallback=False,
    )

    assert res["gate_passed"] is True
    assert res["verdict"] == "GATE_PASSED"
    assert len(res["blockers"]) == 0
    assert all(res["checks"].values())

    # Should not raise
    enforce_readiness_gate(res)


def test_evaluate_readiness_gate_fails_closed_when_features_missing(tmp_path: Path):
    """Test that missing canonical feature table fails gate and stops execution."""
    missing_path = tmp_path / "non_existent.parquet"
    tr_path = tmp_path / "train.csv"
    te_path = tmp_path / "test.csv"
    pd.DataFrame({"participant_id": ["P1"]}).to_csv(tr_path, index=False)
    pd.DataFrame({"participant_id": ["P2"]}).to_csv(te_path, index=False)

    res = evaluate_readiness_gate(
        features_path=missing_path,
        train_ids_path=tr_path,
        test_ids_path=te_path,
    )

    assert res["gate_passed"] is False
    assert res["verdict"] == "GATE_FAILED_BLOCKED"
    assert any("does not exist" in b for b in res["blockers"])

    with pytest.raises(RuntimeError) as exc_info:
        enforce_readiness_gate(res)
    assert "[FAIL-CLOSED READINESS GATE]" in str(exc_info.value)


def test_evaluate_readiness_gate_rejects_legacy_features_clean(tmp_path: Path):
    """Test that legacy features_clean.csv is explicitly rejected without silent fallback."""
    legacy_csv = tmp_path / "features_clean.csv"
    legacy_csv.write_text("participant_id,parkinsons_label\nP1,1\n", encoding="utf-8")
    tr_path = tmp_path / "train.csv"
    te_path = tmp_path / "test.csv"
    pd.DataFrame({"participant_id": ["P1"]}).to_csv(tr_path, index=False)
    pd.DataFrame({"participant_id": ["P2"]}).to_csv(te_path, index=False)

    res = evaluate_readiness_gate(
        features_path=legacy_csv,
        train_ids_path=tr_path,
        test_ids_path=te_path,
        allow_legacy_fallback=False,
    )

    assert res["gate_passed"] is False
    assert any("rejected legacy candidate" in b for b in res["blockers"])


def test_evaluate_readiness_gate_detects_prohibited_covariates(mock_valid_eda_environment, tmp_path: Path):
    """Test that inclusion of clinical/demographic covariates causes gate failure."""
    env = mock_valid_eda_environment
    df_corrupted = env["df_clean"].copy()
    # Add prohibited covariates
    df_corrupted["BirthYear"] = 1955
    df_corrupted["UPDRS_score"] = 24.5

    corrupted_path = tmp_path / "corrupted_covariates.parquet"
    df_corrupted.to_parquet(corrupted_path, index=False)

    res = evaluate_readiness_gate(
        features_path=corrupted_path,
        train_ids_path=env["train_ids_path"],
        test_ids_path=env["test_ids_path"],
        expected_feature_count=None,
    )

    assert res["gate_passed"] is False
    assert res["checks"]["check7_schema_valid"] is False
    assert any("Prohibited demographic/clinical covariates" in b for b in res["blockers"])


def test_evaluate_readiness_gate_detects_date_corruption(mock_valid_eda_environment, tmp_path: Path):
    """Test that shifted date-code values (>100,000) are flagged as fatal integrity blockers."""
    env = mock_valid_eda_environment
    df_corrupted = env["df_clean"].copy()
    # Inject date corruption
    df_corrupted.loc[0, "latency_mean"] = 170522.0

    corrupted_path = tmp_path / "date_corrupted.parquet"
    df_corrupted.to_parquet(corrupted_path, index=False)

    res = evaluate_readiness_gate(
        features_path=corrupted_path,
        train_ids_path=env["train_ids_path"],
        test_ids_path=env["test_ids_path"],
        expected_feature_count=153,
    )

    assert res["gate_passed"] is False
    assert res["checks"]["check8_data_unvitiated"] is False
    assert any("100,000" in b for b in res["blockers"])


def test_evaluate_readiness_gate_detects_cohort_truncation(mock_valid_eda_environment, tmp_path: Path):
    """Test that truncated cohort (e.g. 495 vs 597) fails gate."""
    env = mock_valid_eda_environment
    df_truncated = env["df_clean"].iloc[:495].copy()

    trunc_path = tmp_path / "truncated.parquet"
    df_truncated.to_parquet(trunc_path, index=False)

    res = evaluate_readiness_gate(
        features_path=trunc_path,
        train_ids_path=env["train_ids_path"],
        test_ids_path=env["test_ids_path"],
        eligible_count=597,
    )

    assert res["gate_passed"] is False
    assert res["checks"]["check5_cohort_coverage"] is False
    assert any("row count (495) does not match" in b for b in res["blockers"])


def test_training_only_isolation(mock_valid_eda_environment):
    """Test that filtering strictly isolates training participants from test participants."""
    env = mock_valid_eda_environment
    df = env["df_clean"]
    train_set = set(env["train_ids"])
    test_set = set(env["test_ids"])

    df_train = df[df["participant_id"].isin(train_set)].copy().reset_index(drop=True)
    df_test = df[df["participant_id"].isin(test_set)].copy().reset_index(drop=True)

    assert len(df_train) == 477
    assert len(df_test) == 120
    assert len(set(df_train["participant_id"]).intersection(set(df_test["participant_id"]))) == 0


def test_dynamic_feature_computation(mock_valid_eda_environment):
    """Test dynamic computation of inventory, missingness, descriptive stats, and correlations."""
    env = mock_valid_eda_environment
    df_train = env["df_clean"].iloc[:477].copy()

    # Inventory
    inv = build_feature_inventory(df_train)
    assert len(inv) == 153
    flight_min_row = inv.loc[inv["feature_name"] == "flight_min"]
    assert len(flight_min_row) == 1
    flight_min_text = " ".join(flight_min_row.iloc[0].astype(str).tolist())
    assert "Flight Time (FT)" in flight_min_text
    assert "DESTROYED_ROLLOVER" not in flight_min_text
    assert set(inv["family"].unique()).issubset({
        "Hold Time (HT)", "Latency (L)", "Flight Time (FT)",
        "Hand Dynamics", "Transition Dynamics", "Typing Predictor",
    })

    # Missingness
    miss = compute_missingness_summary(df_train)
    assert len(miss) == 153
    assert (miss["missing_count"] == 0).all()

    # Descriptive Stats
    stats = compute_descriptive_statistics(df_train)
    assert len(stats) == 153
    assert "mean" in stats.columns
    assert "median" in stats.columns
    assert "skewness" in stats.columns

    # Anomalies
    anom = screen_outliers_and_anomalies(df_train)
    assert isinstance(anom, pd.DataFrame)

    # Class comparison
    cls_sum = compute_training_class_summary(df_train)
    assert not cls_sum.empty
    assert "median_diff_pd_minus_ctrl" in cls_sum.columns
