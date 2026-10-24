"""Unit tests for pipeline configuration loading and validation."""
import pytest
from pathlib import Path

from src.data_pipeline.config import PipelineConfig, ConfigError, load_config


def test_load_valid_config(tmp_path: Path):
    config_file = tmp_path / "valid.toml"
    config_file.write_text(
        """
[paths]
raw_logs_dir = "data/raw/keystroke_logs"
raw_metadata_dir = "data/raw/participant_metadata"
interim_dir = "data/interim"
reports_dir = "reports/data_quality"

[format]
delimiter = "\t"
has_trailing_delimiter = true
canonical_field_count = 8
userkey_length = 10
century_base = 2000
allowed_hands = ["L", "R", "S"]
allowed_directions = ["LL", "LR", "RR", "RL", "LS", "RS", "SR", "SL", "SS"]

[thresholds]
extreme_hold_ms = 2000.0

[execution]
max_files = 0
log_filename_pattern = "^[A-Z0-9]{10}_\\\\d{4}\\\\.txt$"
metadata_filename_pattern = "^User_[A-Z0-9]{10}\\\\.txt$"
        """,
        encoding="utf-8",
    )

    cfg = load_config(config_file)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.delimiter == "\t"
    assert cfg.has_trailing_delimiter is True
    assert cfg.canonical_field_count == 8
    assert cfg.allowed_hands == {"L", "R", "S"}
    assert cfg.extreme_hold_ms == 2000.0


def test_load_default_pipeline_toml():
    """Verify the repository's configs/pipeline.toml loads cleanly."""
    cfg = load_config("configs/pipeline.toml")
    assert cfg.canonical_field_count == 8
    assert "LL" in cfg.allowed_directions
    assert "L" in cfg.allowed_hands


def test_missing_section_raises_error(tmp_path: Path):
    config_file = tmp_path / "missing_sec.toml"
    config_file.write_text("[paths]\nraw_logs_dir = 'logs'", encoding="utf-8")
    with pytest.raises(ConfigError, match="Missing required configuration section"):
        load_config(config_file)


def test_invalid_threshold_raises_error(tmp_path: Path):
    config_file = tmp_path / "bad_thresh.toml"
    config_file.write_text(
        """
[paths]
raw_logs_dir = "a"
raw_metadata_dir = "b"
interim_dir = "c"
reports_dir = "d"

[format]
delimiter = "\t"
canonical_field_count = 8
userkey_length = 10
allowed_hands = ["L"]
allowed_directions = ["LL"]

[thresholds]
extreme_hold_ms = -50.0

[execution]
max_files = 0
        """,
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Invalid extreme_hold_ms"):
        load_config(config_file)


def test_nonexistent_file_raises_not_found():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent_pipeline_file.toml")
