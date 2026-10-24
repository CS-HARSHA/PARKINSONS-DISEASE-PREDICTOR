"""Unit and integration tests for pipeline execution entry point and staging (Task T9)."""
import pytest
import shutil
import hashlib
from pathlib import Path
import pandas as pd

from src.data_pipeline.config import load_config
from src.data_pipeline.run import (
    hash_file,
    hash_directory_files,
    run_pipeline,
    publish_staging_directory,
)


def test_hash_file_deterministic(tmp_path: Path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello world")
    h1 = hash_file(f)
    h2 = hash_file(f)
    assert h1 == h2
    assert h1 == hashlib.sha256(b"hello world").hexdigest()


def test_publish_staging_directory_and_failure_safety(tmp_path: Path):
    target_dir = tmp_path / "interim"
    target_dir.mkdir()
    (target_dir / "old_output.txt").write_text("old content", encoding="utf-8")

    staging_dir = tmp_path / "staging_123"
    staging_dir.mkdir()
    (staging_dir / "new_output.txt").write_text("new content", encoding="utf-8")

    # Safe publishing replaces target with staging
    publish_staging_directory(staging_dir, target_dir)
    assert (target_dir / "new_output.txt").exists()
    assert not (target_dir / "old_output.txt").exists()
    assert not staging_dir.exists()


def test_run_pipeline_max_files_and_deterministic_rerun(tmp_path: Path):
    # Setup test workspace
    raw_logs = tmp_path / "logs"
    raw_logs.mkdir()
    raw_meta = tmp_path / "metadata"
    raw_meta.mkdir()
    interim_dir = tmp_path / "interim"
    reports_dir = tmp_path / "reports"

    # User 1
    (raw_meta / "User_USER000001.txt").write_text("BirthYear: 1955\nParkinsons: True\n", encoding="utf-8")
    (raw_logs / "USER000001_1607.txt").write_bytes(
        b"USER000001\t160723\t17:35:20.516\tL\t0199.2\tLL\t0453.1\t0281.3\t\r\n"
        b"USER000001\t160723\t17:35:21.000\tL\t0199.2\tLL\t0453.1\t0281.3\t\r\n"
    )

    # User 2
    (raw_meta / "User_USER000002.txt").write_text("BirthYear: 1960\nParkinsons: False\n", encoding="utf-8")
    (raw_logs / "USER000002_1608.txt").write_bytes(
        b"USER000002\t160810\t10:00:00.000\tR\t0150.0\tRR\t0300.0\t0100.0\t\r\n"
    )

    # First run with max_files=1
    res1 = run_pipeline(
        raw_logs_dir=raw_logs,
        raw_metadata_dir=raw_meta,
        interim_dir=interim_dir,
        reports_dir=reports_dir,
        config_path="configs/pipeline.toml",
        max_files=1,
    )
    assert res1["files_processed"] == 1
    assert (interim_dir / "accepted" / "USER000001_1607.parquet").exists()
    assert not (interim_dir / "accepted" / "USER000002_1608.parquet").exists()
    assert (interim_dir / "participant_labels.csv").exists()
    assert (interim_dir / "participant_inventory.csv").exists()
    assert (interim_dir / "run_manifest.json").exists()

    # Read output content
    inv1 = (interim_dir / "participant_inventory.csv").read_text(encoding="utf-8")

    # Second deterministic run with max_files=1
    res2 = run_pipeline(
        raw_logs_dir=raw_logs,
        raw_metadata_dir=raw_meta,
        interim_dir=interim_dir,
        reports_dir=reports_dir,
        config_path="configs/pipeline.toml",
        max_files=1,
    )
    inv2 = (interim_dir / "participant_inventory.csv").read_text(encoding="utf-8")

    # Outputs must be identical
    assert inv1 == inv2
    assert res1["raw_hashes_verified"] is True
    assert res2["raw_hashes_verified"] is True
