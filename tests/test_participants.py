"""Unit tests for participant inventory, labels, and cohort reconciliation (Task T7)."""
import pytest
from pathlib import Path
import pandas as pd

from src.data_pipeline.config import load_config
from src.data_pipeline.participants import build_participant_datasets, ParticipantRecord

CONFIG = load_config("configs/pipeline.toml")


def test_build_participant_datasets(tmp_path: Path):
    meta_dir = tmp_path / "metadata"
    meta_dir.mkdir()

    # User 1: Eligible control (has logs and valid False label)
    (meta_dir / "User_USER000001.txt").write_text(
        "BirthYear: 1960\nGender: Female\nParkinsons: False\n", encoding="utf-8"
    )

    # User 2: Eligible PD (has logs and valid True label)
    (meta_dir / "User_USER000002.txt").write_text(
        "BirthYear: 1950\nGender: Male\nParkinsons: True\n", encoding="utf-8"
    )

    # User 3: Missing label (in logs and metadata, but blank label)
    (meta_dir / "User_USER000003.txt").write_text(
        "BirthYear: 1970\nGender: \nParkinsons: \n", encoding="utf-8"
    )

    # User 4: Metadata only (no logs)
    (meta_dir / "User_USER000004.txt").write_text(
        "BirthYear: 1965\nGender: Female\nParkinsons: True\n", encoding="utf-8"
    )

    # Mock log summaries
    # User 1 has 1 file, 100 accepted lines
    # User 2 has 2 files, 50 accepted, 10 warned lines
    # User 3 has 1 file, 20 accepted lines
    # User 5 has 1 file (logs only, no metadata)
    log_stats = {
        "USER000001": {
            "n_files": 1,
            "first_yymm": "1607",
            "last_yymm": "1607",
            "n_lines": 100,
            "n_accepted": 100,
            "n_warned": 0,
            "n_quarantined": 0,
            "n_blank": 0,
        },
        "USER000002": {
            "n_files": 2,
            "first_yymm": "1608",
            "last_yymm": "1609",
            "n_lines": 60,
            "n_accepted": 50,
            "n_warned": 10,
            "n_quarantined": 0,
            "n_blank": 0,
        },
        "USER000003": {
            "n_files": 1,
            "first_yymm": "1701",
            "last_yymm": "1701",
            "n_lines": 20,
            "n_accepted": 20,
            "n_warned": 0,
            "n_quarantined": 0,
            "n_blank": 0,
        },
        "USER000005": {
            "n_files": 1,
            "first_yymm": "1702",
            "last_yymm": "1702",
            "n_lines": 50,
            "n_accepted": 50,
            "n_warned": 0,
            "n_quarantined": 0,
            "n_blank": 0,
        },
    }

    labels_df, covar_df, inv_df = build_participant_datasets(meta_dir, log_stats)

    # Verify labels dataframe
    assert len(labels_df) == 4
    u1_lbl = labels_df.loc[labels_df["user_key"] == "USER000001"].iloc[0]
    assert u1_lbl["label"] == 0
    assert u1_lbl["label_status"] == "VALID_LABEL"

    u2_lbl = labels_df.loc[labels_df["user_key"] == "USER000002"].iloc[0]
    assert u2_lbl["label"] == 1

    u3_lbl = labels_df.loc[labels_df["user_key"] == "USER000003"].iloc[0]
    assert pd.isna(u3_lbl["label"])
    assert u3_lbl["label_status"] == "MISSING_LABEL"

    # Verify inventory
    # Total unique users: USER000001, 2, 3, 4, 5 -> 5 users
    assert len(inv_df) == 5

    # Check cohort statuses
    u1_inv = inv_df.loc[inv_df["user_key"] == "USER000001"].iloc[0]
    assert u1_inv["cohort_status"] == "ELIGIBLE"
    assert u1_inv["exclusion_reasons"] == ""

    u3_inv = inv_df.loc[inv_df["user_key"] == "USER000003"].iloc[0]
    assert u3_inv["cohort_status"] == "EXCLUDED_NO_VALID_LABEL"
    assert "MISSING_LABEL" in u3_inv["exclusion_reasons"]

    u4_inv = inv_df.loc[inv_df["user_key"] == "USER000004"].iloc[0]
    assert u4_inv["cohort_status"] == "EXCLUDED_NO_LOGS"
    assert "NO_LOGS" in u4_inv["exclusion_reasons"]

    u5_inv = inv_df.loc[inv_df["user_key"] == "USER000005"].iloc[0]
    assert u5_inv["cohort_status"] == "EXCLUDED_NO_METADATA"
    assert "NO_METADATA" in u5_inv["exclusion_reasons"]
