"""Unit tests for reporting module (Task T8)."""
import pytest
import json
from pathlib import Path
import pandas as pd

from src.data_pipeline.cleaning import FileCleaningSummary
from src.data_pipeline.reporting import generate_reports


def test_generate_reports(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    # Create mock FileCleaningSummary
    s1 = FileCleaningSummary(
        source_file="USER000001_1607.txt",
        user_key="USER000001",
        yymm="1607",
        n_lines=10,
        n_accepted=8,
        n_warned=1,
        n_quarantined=1,
        n_blank=0,
        backward_deltas_ms=[500.0, 1500.0],
    )
    s1.error_counts["E07_BAD_HAND"] = 1
    s1.affected_lines_per_code["E07_BAD_HAND"] = 1
    s1.warning_counts_accepted["W04_EXTREME_HOLD"] = 1
    s1.affected_lines_per_code["W04_EXTREME_HOLD"] = 1

    file_summaries = [s1]

    # Mock inventory
    inv_df = pd.DataFrame([{
        "user_key": "USER000001",
        "in_logs": True,
        "n_log_files": 1,
        "first_yymm": "1607",
        "last_yymm": "1607",
        "in_metadata": True,
        "label_status": "VALID_LABEL",
        "n_lines": 10,
        "n_accepted": 8,
        "n_warned": 1,
        "n_quarantined": 1,
        "n_blank": 0,
        "cohort_status": "ELIGIBLE",
        "exclusion_reasons": "",
    }])

    # Mock raw hashes
    raw_hashes_df = pd.DataFrame([{
        "relative_path": "data/raw/keystroke_logs/USER000001_1607.txt",
        "sha256_before": "abc123hash",
        "sha256_after": "abc123hash",
        "is_identical": True,
    }])

    manifest_data = {
        "git_commit": "testcommit",
        "git_dirty": False,
        "python_version": "3.13.7",
        "raw_hashes_verified": True,
    }

    result = generate_reports(
        file_summaries=file_summaries,
        inventory_df=inv_df,
        raw_hashes_df=raw_hashes_df,
        reports_dir=reports_dir,
        manifest_data=manifest_data,
    )

    assert (reports_dir / "data_quality_report.md").exists()
    assert (reports_dir / "summary.json").exists()
    assert (reports_dir / "per_file_counts.csv").exists()
    assert (reports_dir / "code_counts.csv").exists()
    assert (reports_dir / "backward_timestamp_summary.csv").exists()
    assert (reports_dir / "audit_reconciliation.csv").exists()
    assert (reports_dir / "cohort_reconciliation.csv").exists()
    assert (reports_dir / "raw_files_sha256.csv").exists()

    # Verify code_counts distinguishes occurrences and affected lines
    code_df = pd.read_csv(reports_dir / "code_counts.csv")
    assert "total_occurrences" in code_df.columns
    assert "unique_lines_affected" in code_df.columns
    assert "accepted_lines_count" in code_df.columns
    assert "quarantined_lines_count" in code_df.columns
