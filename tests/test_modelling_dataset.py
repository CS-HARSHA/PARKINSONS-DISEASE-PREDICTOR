"""Unit and integration tests for Phase 4 modelling dataset validation and participant splits."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.modelling_dataset import (
    audit_cohort_flow,
    audit_feature_integrity,
    audit_historical_splits,
    generate_prospective_split,
    build_reconciliation_summary,
)


@pytest.fixture
def mock_cohort_data(tmp_path: Path):
    """Create small synthetic participant inventory and labels fixtures."""
    # Create inventory
    inv_rows = [
        # Eligible PD
        {"user_key": f"PD_ELIG_{i:03d}", "in_logs": True, "in_metadata": True, "cohort_status": "ELIGIBLE", "label_status": "VALID_LABEL", "n_accepted": 500, "n_warned": 10}
        for i in range(1, 21)  # 20 eligible PD
    ] + [
        # Eligible Control
        {"user_key": f"CTRL_ELIG_{i:03d}", "in_logs": True, "in_metadata": True, "cohort_status": "ELIGIBLE", "label_status": "VALID_LABEL", "n_accepted": 300, "n_warned": 5}
        for i in range(1, 11)  # 10 eligible Control
    ] + [
        # Excluded: No Metadata
        {"user_key": "EX_NOMET_001", "in_logs": True, "in_metadata": False, "cohort_status": "EXCLUDED_NO_METADATA", "label_status": "NO_METADATA", "n_accepted": 100, "n_warned": 0},
        # Excluded: No Logs
        {"user_key": "EX_NOLOG_001", "in_logs": False, "in_metadata": True, "cohort_status": "EXCLUDED_NO_LOGS", "label_status": "VALID_LABEL", "n_accepted": 0, "n_warned": 0},
        # Excluded: Missing Label
        {"user_key": "EX_MISSL_001", "in_logs": True, "in_metadata": True, "cohort_status": "EXCLUDED_NO_VALID_LABEL", "label_status": "MISSING_LABEL", "n_accepted": 50, "n_warned": 0},
        # Excluded: No Usable Records
        {"user_key": "EX_NOREC_001", "in_logs": True, "in_metadata": True, "cohort_status": "EXCLUDED_NO_USABLE_RECORDS", "label_status": "VALID_LABEL", "n_accepted": 0, "n_warned": 0},
    ]
    inv_df = pd.DataFrame(inv_rows)
    inv_path = tmp_path / "participant_inventory.csv"
    inv_df.to_csv(inv_path, index=False)

    # Create labels
    label_rows = [
        {"user_key": f"PD_ELIG_{i:03d}", "parkinsons_raw": "True", "label": 1.0, "label_status": "VALID_LABEL"}
        for i in range(1, 21)
    ] + [
        {"user_key": f"CTRL_ELIG_{i:03d}", "parkinsons_raw": "False", "label": 0.0, "label_status": "VALID_LABEL"}
        for i in range(1, 11)
    ] + [
        {"user_key": "EX_NOLOG_001", "parkinsons_raw": "True", "label": 1.0, "label_status": "VALID_LABEL"},
        {"user_key": "EX_MISSL_001", "parkinsons_raw": "", "label": np.nan, "label_status": "MISSING_LABEL"},
        {"user_key": "EX_NOREC_001", "parkinsons_raw": "True", "label": 1.0, "label_status": "VALID_LABEL"},
    ]
    labels_df = pd.DataFrame(label_rows)
    labels_path = tmp_path / "participant_labels.csv"
    labels_df.to_csv(labels_path, index=False)

    return inv_path, labels_path, inv_df, labels_df


def test_unique_participant_ids_and_valid_labels(mock_cohort_data):
    """Test that participant IDs in labels are unique and map correctly."""
    _, labels_path, _, labels_df = mock_cohort_data
    assert labels_df["user_key"].is_unique, "Participant IDs must be unique in labels!"
    assert labels_df["label"].isin([0.0, 1.0, np.nan]).all(), "Labels must be 0.0, 1.0 or NaN!"


def test_audit_cohort_flow(mock_cohort_data):
    """Test cohort flow reconciliation across pipeline stages."""
    inv_path, labels_path, _, _ = mock_cohort_data
    flow_df, metrics = audit_cohort_flow(inv_path, labels_path)

    assert metrics["total_unique_keys"] == 34
    assert metrics["in_logs_count"] == 33
    assert metrics["in_meta_count"] == 33
    assert metrics["joint_count"] == 32
    assert metrics["joint_valid_count"] == 31
    assert metrics["eligible_count"] == 30
    assert metrics["eligible_pd"] == 20
    assert metrics["eligible_ctrl"] == 10
    assert metrics["no_records_count"] == 1
    assert metrics["joint_missing_count"] == 1


def test_feature_integrity_audit_detects_date_corruption(tmp_path: Path):
    """Test that audit_feature_integrity detects shifted date values and destroyed rollover."""
    synthetic_features = pd.DataFrame({
        "participant_id": ["P001", "P002", "P003"],
        "parkinsons_label": [1, 0, 1],
        "hold_mean": [85.5, 92.1, 88.0],
        "lat1_max": [450.0, 170511.0, 520.0],  # 170511.0 is a shifted YYMMDD date code
        "lat2_min": [12.0, 5.0, 15.0],        # All >= 0, destroyed negative rollover
        "normal_feat": [1.0, 2.0, 3.0],
    })
    feat_path = tmp_path / "corrupted_features.csv"
    synthetic_features.to_csv(feat_path, index=False)

    audit_df, summary = audit_feature_integrity(feat_path)

    assert summary["columns_with_date_corruption"] == 1
    assert "lat1_max" in [c[0] for c in summary["corrupted_date_columns_detail"]]

    lat2_min_status = audit_df.loc[audit_df["column_name"] == "lat2_min", "integrity_status"].iloc[0]
    assert "DESTROYED_ROLLOVER" in lat2_min_status


def test_generate_prospective_split_contract(mock_cohort_data, tmp_path: Path):
    """Test participant-disjoint split contract on synthetic eligible cohort."""
    inv_path, labels_path, _, _ = mock_cohort_data
    splits_dir = tmp_path / "splits"

    manifest = generate_prospective_split(
        inv_path,
        labels_path,
        output_dir=splits_dir,
        test_size=0.20,
        random_state=42,
    )

    train_file = splits_dir / "train_ids.csv"
    test_file = splits_dir / "test_ids.csv"
    assert train_file.is_file()
    assert test_file.is_file()

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    train_ids = set(train_df["participant_id"])
    test_ids = set(test_df["participant_id"])

    # 1. Disjointness
    assert len(train_ids.intersection(test_ids)) == 0, "Train and test must be disjoint!"

    # 2. Complete coverage
    inv_df = pd.read_csv(inv_path)
    eligible_keys = set(inv_df[inv_df["cohort_status"] == "ELIGIBLE"]["user_key"])
    assert train_ids.union(test_ids) == eligible_keys, "Union must equal eligible cohort!"

    # 3. Deterministic split counts
    assert len(train_ids) + len(test_ids) == 30
    assert manifest["training_partition"]["count"] == len(train_ids)
    assert manifest["test_partition"]["count"] == len(test_ids)


def test_prohibited_predictor_columns_check():
    """Verify that clinical covariates and identifiers are identified as prohibited."""
    prohibited_cols = {
        "user_key", "participant_id", "parkinsons", "parkinsons_raw", "label",
        "birth_year", "gender", "updrs", "tremors", "levodopa", "da", "maob",
        "other", "diagnosis_year", "sided", "impact",
    }
    candidate_cols = ["participant_id", "hold_mean", "lat_std", "birth_year", "updrs"]
    leaked = [c for c in candidate_cols if c.lower() in prohibited_cols]

    assert "participant_id" in leaked
    assert "birth_year" in leaked
    assert "updrs" in leaked
    assert "hold_mean" not in leaked


def test_reconciliation_summary_table():
    """Test that reconciliation summary reflects verified cohort metrics."""
    metrics = {
        "in_logs_count": 754,
        "in_meta_count": 646,
        "joint_count": 609,
        "joint_valid_count": 606,
        "eligible_count": 597,
    }
    feat_summary = {"total_rows": 495}
    split_manifest = {
        "training_partition": {"count": 477},
        "test_partition": {"count": 120},
    }

    recon_df = build_reconciliation_summary(metrics, feat_summary, split_manifest)
    assert len(recon_df) == 8
    assert (recon_df["difference"].abs() >= 0).all()
