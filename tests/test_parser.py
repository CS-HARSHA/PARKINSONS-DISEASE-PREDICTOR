"""Unit tests for byte-level streaming parser and metadata parser (Task T4)."""
import pytest
from pathlib import Path

from src.data_pipeline.config import load_config
from src.data_pipeline.parser import (
    iter_physical_lines,
    parse_log_file,
    parse_metadata_file,
    RawRecord,
    MetadataRecord,
)

DEFAULT_CONFIG = load_config("configs/pipeline.toml")


def test_iter_physical_lines_crlf_and_lf():
    data = b"line1\r\nline2\nline3\r\n"
    lines = list(iter_physical_lines(data))
    assert len(lines) == 3
    assert lines[0] == (1, b"line1", b"\r\n")
    assert lines[1] == (2, b"line2", b"\n")
    assert lines[2] == (3, b"line3", b"\r\n")


def test_iter_physical_lines_unterminated_last_line():
    data = b"line1\r\nline2"
    lines = list(iter_physical_lines(data))
    assert len(lines) == 2
    assert lines[0] == (1, b"line1", b"\r\n")
    assert lines[1] == (2, b"line2", b"")


def test_iter_physical_lines_lone_cr():
    data = b"line1\rline2\rline3"
    lines = list(iter_physical_lines(data))
    assert len(lines) == 3
    assert lines[0] == (1, b"line1", b"\r")
    assert lines[1] == (2, b"line2", b"\r")
    assert lines[2] == (3, b"line3", b"")


def test_iter_physical_lines_empty_data():
    data = b""
    lines = list(iter_physical_lines(data))
    assert len(lines) == 0


def test_parse_log_file_standard_record(tmp_path: Path):
    fpath = tmp_path / "0EA27ICBLF_1607.txt"
    fpath.write_bytes(b"0EA27ICBLF\t160723\t17:35:20.516\tL\t0199.2\tLL\t0453.1\t0281.3\t\r\n")

    records, is_empty = parse_log_file(fpath, DEFAULT_CONFIG)
    assert not is_empty
    assert len(records) == 1
    rec = records[0]
    assert rec.source_file == "0EA27ICBLF_1607.txt"
    assert rec.line_no == 1
    assert rec.is_blank is False
    assert rec.is_utf8 is True
    assert rec.field_count == 8
    assert rec.fields == [
        "0EA27ICBLF", "160723", "17:35:20.516", "L",
        "0199.2", "LL", "0453.1", "0281.3"
    ]


def test_parse_log_file_bom_handling(tmp_path: Path):
    fpath = tmp_path / "0EA27ICBLF_1607.txt"
    fpath.write_bytes(b"\xef\xbb\xbf0EA27ICBLF\t160723\t17:35:20.516\tL\t0199.2\tLL\t0453.1\t0281.3\t\r\n")

    records, is_empty = parse_log_file(fpath, DEFAULT_CONFIG)
    assert len(records) == 1
    assert records[0].has_bom is True
    assert records[0].fields[0] == "0EA27ICBLF"
    assert records[0].raw_line_bytes.startswith(b"\xef\xbb\xbf")


def test_parse_log_file_blank_and_whitespace_lines(tmp_path: Path):
    fpath = tmp_path / "TEST_1607.txt"
    # Line 1: normal, Line 2: completely blank, Line 3: spaces only (not blank)
    content = (
        b"0EA27ICBLF\t160723\t17:35:20.516\tL\t0199.2\tLL\t0453.1\t0281.3\t\r\n"
        b"\r\n"
        b"   \r\n"
    )
    fpath.write_bytes(content)

    records, _ = parse_log_file(fpath, DEFAULT_CONFIG)
    assert len(records) == 3
    assert records[0].is_blank is False
    assert records[1].is_blank is True
    assert records[2].is_blank is False
    # Line 3 is not blank, and won't match 8 tab fields
    assert records[2].field_count != 8


def test_parse_log_file_undecodable_utf8_bytes(tmp_path: Path):
    fpath = tmp_path / "TEST_1607.txt"
    # Insert invalid UTF-8 byte 0xff
    fpath.write_bytes(b"0EA27ICBLF\t160723\t\xff\xfe\tL\t0199.2\tLL\t0453.1\t0281.3\t\r\n")

    records, _ = parse_log_file(fpath, DEFAULT_CONFIG)
    assert len(records) == 1
    rec = records[0]
    assert rec.is_utf8 is False
    assert b"\xff\xfe" in rec.raw_line_bytes
    assert "\ufffd" in rec.raw_line_display


def test_parse_log_file_empty(tmp_path: Path):
    fpath = tmp_path / "EMPTY_1703.txt"
    fpath.write_bytes(b"")

    records, is_empty = parse_log_file(fpath, DEFAULT_CONFIG)
    assert is_empty is True
    assert len(records) == 0


def test_parse_metadata_file_valid_true(tmp_path: Path):
    fpath = tmp_path / "User_0EA27ICBLF.txt"
    fpath.write_text(
        "BirthYear: 1955\n"
        "Gender: Male\n"
        "Parkinsons: True\n"
        "Tremors: False\n",
        encoding="utf-8"
    )
    meta = parse_metadata_file(fpath)
    assert meta.user_key == "0EA27ICBLF"
    assert meta.parkinsons_raw == "True"
    assert meta.label == 1
    assert meta.label_status == "VALID_LABEL"
    assert meta.covariates["BirthYear"] == "1955"


def test_parse_metadata_file_valid_false(tmp_path: Path):
    fpath = tmp_path / "User_0JIWJUQ4ZN.txt"
    fpath.write_text("Parkinsons: false\nBirthYear: 1960\n", encoding="utf-8")
    meta = parse_metadata_file(fpath)
    assert meta.label == 0
    assert meta.label_status == "VALID_LABEL"


def test_parse_metadata_file_missing_label(tmp_path: Path):
    fpath = tmp_path / "User_FXD2HCNIPJ.txt"
    fpath.write_text("BirthYear: \nParkinsons: \n", encoding="utf-8")
    meta = parse_metadata_file(fpath)
    assert meta.label is None
    assert meta.label_status == "MISSING_LABEL"


def test_parse_metadata_file_invalid_label(tmp_path: Path):
    fpath = tmp_path / "User_INVALID01.txt"
    fpath.write_text("Parkinsons: Maybe\n", encoding="utf-8")
    meta = parse_metadata_file(fpath)
    assert meta.label is None
    assert meta.label_status == "INVALID_LABEL"


def test_parse_metadata_duplicate_unique_field(tmp_path: Path):
    fpath = tmp_path / "User_DUPKEY01.txt"
    fpath.write_text("Parkinsons: True\nParkinsons: False\n", encoding="utf-8")
    meta = parse_metadata_file(fpath)
    assert meta.label_status == "INVALID_DUPLICATE_FIELD"
    assert meta.label is None
