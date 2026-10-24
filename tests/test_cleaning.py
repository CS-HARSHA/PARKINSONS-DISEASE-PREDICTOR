"""Unit tests for cleaning, classification, and parquet output writing (Task T6)."""
import pytest
import pyarrow.parquet as pq
from pathlib import Path

from src.data_pipeline.config import load_config
from src.data_pipeline.cleaning import clean_log_file, FileCleaningSummary

CONFIG = load_config("configs/pipeline.toml")


def test_clean_log_file_with_mixed_outcomes(tmp_path: Path):
    out_dir = tmp_path / "interim"
    log_file = tmp_path / "0EA27ICBLF_1607.txt"

    # Construct test data:
    # Line 1: Valid clean accepted record
    # Line 2: Blank line
    # Line 3: Accepted with warning (extreme hold 2500ms > 2000ms)
    # Line 4: Accepted with warning (duplicate of line 1)
    # Line 5: Quarantined (bad hand 'X')
    # Line 6: Quarantined (bad encoding 0xff)
    content = (
        b"0EA27ICBLF\t160723\t17:35:20.516\tL\t0199.2\tLL\t0453.1\t0281.3\t\r\n"
        b"\r\n"
        b"0EA27ICBLF\t160723\t17:35:21.000\tL\t2500.0\tLL\t0453.1\t0281.3\t\r\n"
        b"0EA27ICBLF\t160723\t17:35:20.516\tL\t0199.2\tLL\t0453.1\t0281.3\t\r\n"
        b"0EA27ICBLF\t160723\t17:35:22.000\tX\t0199.2\tLL\t0453.1\t0281.3\t\r\n"
        b"0EA27ICBLF\t160723\t\xff\xfe\tL\t0199.2\tLL\t0453.1\t0281.3\t\r\n"
    )
    log_file.write_bytes(content)

    summary = clean_log_file(log_file, out_dir, CONFIG)

    # Check invariant: physical_lines == accepted + accepted_with_warning + quarantined + blank
    assert summary.n_lines == 6
    assert summary.n_accepted == 1
    assert summary.n_blank == 1
    assert summary.n_warned == 2
    assert summary.n_quarantined == 2
    assert summary.n_lines == (
        summary.n_accepted + summary.n_warned + summary.n_quarantined + summary.n_blank
    )

    # Check parquet output existence
    accepted_pq = out_dir / "accepted" / "0EA27ICBLF_1607.parquet"
    quarantine_pq = out_dir / "quarantine" / "0EA27ICBLF_1607.parquet"

    assert accepted_pq.exists()
    assert quarantine_pq.exists()

    # Read accepted parquet table
    t_acc = pq.read_table(accepted_pq)
    assert t_acc.num_rows == 3  # 1 accepted + 2 accepted_with_warning
    assert "event_time" in t_acc.column_names
    assert "warning_codes" in t_acc.column_names

    # Read quarantine parquet table and verify lossless byte retention
    t_quar = pq.read_table(quarantine_pq)
    assert t_quar.num_rows == 2
    raw_bytes_col = t_quar.column("raw_line_bytes").to_pylist()
    assert any(b"\xff\xfe" in b for b in raw_bytes_col)


def test_clean_log_file_without_quarantine_does_not_create_quarantine_file(tmp_path: Path):
    out_dir = tmp_path / "interim"
    log_file = tmp_path / "CLEANFILE1_1701.txt"

    content = b"CLEANFILE1\t170110\t10:00:00.000\tL\t0100.0\tLL\t0200.0\t0100.0\t\r\n"
    log_file.write_bytes(content)

    summary = clean_log_file(log_file, out_dir, CONFIG)
    assert summary.n_accepted == 1
    assert summary.n_quarantined == 0

    accepted_pq = out_dir / "accepted" / "CLEANFILE1_1701.parquet"
    quarantine_pq = out_dir / "quarantine" / "CLEANFILE1_1701.parquet"

    assert accepted_pq.exists()
    assert not quarantine_pq.exists()
