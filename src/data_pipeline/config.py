"""Configuration loader and validator for Phase 2 data pipeline."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set


class ConfigError(ValueError):
    """Raised when pipeline configuration is invalid or missing required keys."""
    pass


@dataclass(frozen=True)
class PipelineConfig:
    """Strongly-typed pipeline configuration container."""
    # Paths
    raw_logs_dir: Path
    raw_metadata_dir: Path
    interim_dir: Path
    reports_dir: Path

    # Format contracts
    delimiter: str
    has_trailing_delimiter: bool
    canonical_field_count: int
    userkey_length: int
    century_base: int
    allowed_hands: Set[str]
    allowed_directions: Set[str]

    # Thresholds
    extreme_hold_ms: float

    # Execution limits & filters
    max_files: int
    log_filename_pattern: str
    metadata_filename_pattern: str


def load_config(config_path: str | Path) -> PipelineConfig:
    """Load and validate pipeline configuration from a TOML file.

    Parameters
    ----------
    config_path : str | Path
        Path to the TOML configuration file.

    Returns
    -------
    PipelineConfig
        Validated configuration instance.

    Raises
    ------
    ConfigError
        If required sections or values are missing or invalid.
    FileNotFoundError
        If config_path does not exist.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("rb") as f:
        try:
            data = tomllib.load(f)
        except Exception as err:
            raise ConfigError(f"Error parsing TOML configuration {path}: {err}") from err

    # Validate sections
    for section in ("paths", "format", "thresholds", "execution"):
        if section not in data:
            raise ConfigError(f"Missing required configuration section: '{section}'")

    paths = data["paths"]
    fmt = data["format"]
    thresh = data["thresholds"]
    exec_sec = data["execution"]

    # Validate paths
    for p_key in ("raw_logs_dir", "raw_metadata_dir", "interim_dir", "reports_dir"):
        if p_key not in paths or not str(paths[p_key]).strip():
            raise ConfigError(f"Missing or empty path configuration: '{p_key}'")

    # Validate format contracts
    delimiter = fmt.get("delimiter")
    if not isinstance(delimiter, str) or len(delimiter) == 0:
        raise ConfigError("Invalid delimiter: must be a non-empty string.")

    has_trailing_delimiter = bool(fmt.get("has_trailing_delimiter", False))

    field_count = fmt.get("canonical_field_count")
    if not isinstance(field_count, int) or field_count <= 0:
        raise ConfigError(f"Invalid canonical_field_count: {field_count}. Must be positive integer.")

    userkey_len = fmt.get("userkey_length")
    if not isinstance(userkey_len, int) or userkey_len <= 0:
        raise ConfigError(f"Invalid userkey_length: {userkey_len}. Must be positive integer.")

    century_base = fmt.get("century_base", 2000)
    if not isinstance(century_base, int) or century_base < 1900 or century_base > 2100:
        raise ConfigError(f"Invalid century_base: {century_base}.")

    hands = fmt.get("allowed_hands")
    if not isinstance(hands, list) or len(hands) == 0 or not all(isinstance(h, str) for h in hands):
        raise ConfigError("Invalid allowed_hands: must be a non-empty list of strings.")

    directions = fmt.get("allowed_directions")
    if not isinstance(directions, list) or len(directions) == 0 or not all(isinstance(d, str) for d in directions):
        raise ConfigError("Invalid allowed_directions: must be a non-empty list of strings.")

    # Validate thresholds
    extreme_hold_ms = thresh.get("extreme_hold_ms")
    if not isinstance(extreme_hold_ms, (int, float)) or extreme_hold_ms <= 0:
        raise ConfigError(f"Invalid extreme_hold_ms: {extreme_hold_ms}. Must be positive number.")

    # Validate execution
    max_files = exec_sec.get("max_files", 0)
    if not isinstance(max_files, int) or max_files < 0:
        raise ConfigError(f"Invalid max_files: {max_files}. Must be >= 0.")

    log_pattern = exec_sec.get("log_filename_pattern", r"^[A-Z0-9]{10}_\d{4}\.txt$")
    meta_pattern = exec_sec.get("metadata_filename_pattern", r"^User_[A-Z0-9]{10}\.txt$")

    return PipelineConfig(
        raw_logs_dir=Path(paths["raw_logs_dir"]),
        raw_metadata_dir=Path(paths["raw_metadata_dir"]),
        interim_dir=Path(paths["interim_dir"]),
        reports_dir=Path(paths["reports_dir"]),
        delimiter=delimiter,
        has_trailing_delimiter=has_trailing_delimiter,
        canonical_field_count=field_count,
        userkey_length=userkey_len,
        century_base=century_base,
        allowed_hands=set(hands),
        allowed_directions=set(directions),
        extreme_hold_ms=float(extreme_hold_ms),
        max_files=max_files,
        log_filename_pattern=log_pattern,
        metadata_filename_pattern=meta_pattern,
    )
