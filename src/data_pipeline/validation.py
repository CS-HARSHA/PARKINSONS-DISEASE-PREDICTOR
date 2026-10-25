"""Pure validation functions and per-file validation state (Task T5)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date as dt_date, datetime, time as dt_time
from typing import List, Optional, Set, Tuple

from src.data_pipeline.config import PipelineConfig

USERKEY_PATTERN = re.compile(r"^[A-Z0-9]{10}$")
DATE_PATTERN = re.compile(r"^\d{6}$")
TIME_PATTERN = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$")
PLAIN_DECIMAL_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")


@dataclass
class FileValidationState:
    """Per-file state tracker for chronological and uniqueness checks."""
    filename: str
    expected_user_key: str
    expected_yymm: str
    last_valid_dt: Optional[datetime] = None
    seen_records: Set[Tuple[str, ...]] = field(default_factory=set)
    backward_deltas_ms: List[float] = field(default_factory=list)

    def record_backward_delta(self, delta_ms: float) -> None:
        self.backward_deltas_ms.append(delta_ms)


def validate_record(
    fields: List[str],
    state: FileValidationState,
    config: PipelineConfig,
) -> Tuple[List[str], List[str], Optional[datetime]]:
    """Validate a parsed keystroke record against rules E01-E11 and W01-W05.

    Parameters
    ----------
    fields : list[str]
        Extracted fields from the line.
    state : FileValidationState
        Per-file tracking state for sequence checks.
    config : PipelineConfig
        Pipeline configuration containing thresholds and allowed sets.

    Returns
    -------
    tuple
        (error_codes: list[str], warning_codes: list[str], event_dt: Optional[datetime])
    """
    errors: List[str] = []
    warnings: List[str] = []

    if len(fields) != config.canonical_field_count:
        errors.append("E01_FIELD_COUNT")
        return (errors, warnings, None)

    user_key, date_str, time_str, hand, hold_str, direction, latency_str, flight_str = fields

    # E03: BAD_USERKEY
    if not USERKEY_PATTERN.match(user_key):
        errors.append("E03_BAD_USERKEY")

    # E04: USERKEY_MISMATCH
    if user_key != state.expected_user_key:
        errors.append("E04_USERKEY_MISMATCH")

    # E05: BAD_DATE
    valid_date: Optional[dt_date] = None
    if DATE_PATTERN.match(date_str):
        yy = int(date_str[:2])
        mm = int(date_str[2:4])
        dd = int(date_str[4:6])
        full_year = config.century_base + yy
        try:
            valid_date = dt_date(full_year, mm, dd)
        except ValueError:
            errors.append("E05_BAD_DATE")
    else:
        errors.append("E05_BAD_DATE")

    # E06: BAD_TIME
    valid_time: Optional[dt_time] = None
    time_match = TIME_PATTERN.match(time_str)
    if time_match:
        hh = int(time_match.group(1))
        mi = int(time_match.group(2))
        ss = int(time_match.group(3))
        ms = int(time_match.group(4))
        if 0 <= hh <= 23 and 0 <= mi <= 59 and 0 <= ss <= 59 and 0 <= ms <= 999:
            valid_time = dt_time(hh, mi, ss, ms * 1000)
        else:
            errors.append("E06_BAD_TIME")
    else:
        errors.append("E06_BAD_TIME")

    # Construct event datetime if both date and time are valid
    event_dt: Optional[datetime] = None
    if valid_date is not None and valid_time is not None:
        event_dt = datetime.combine(valid_date, valid_time)

    # E07: BAD_HAND
    if hand not in config.allowed_hands:
        errors.append("E07_BAD_HAND")

    # E08: BAD_HOLD
    hold_val: Optional[float] = None
    if PLAIN_DECIMAL_PATTERN.match(hold_str):
        hold_val = float(hold_str)
        if hold_val < 0.0:
            errors.append("E08_BAD_HOLD")
    else:
        errors.append("E08_BAD_HOLD")

    # E09: BAD_DIRECTION
    if direction not in config.allowed_directions:
        errors.append("E09_BAD_DIRECTION")

    # E10: BAD_LATENCY
    latency_val: Optional[float] = None
    if PLAIN_DECIMAL_PATTERN.match(latency_str):
        latency_val = float(latency_str)
        if latency_val < 0.0:
            errors.append("E10_BAD_LATENCY")
    else:
        errors.append("E10_BAD_LATENCY")

    # E11: BAD_FLIGHT (negative is permitted)
    flight_val: Optional[float] = None
    if PLAIN_DECIMAL_PATTERN.match(flight_str):
        flight_val = float(flight_str)
    else:
        errors.append("E11_BAD_FLIGHT")

    # --- Warnings (W01 - W05) ---

    # W01: BACKWARD_TIMESTAMP
    if event_dt is not None:
        if state.last_valid_dt is not None:
            if event_dt < state.last_valid_dt:
                delta_ms = (state.last_valid_dt - event_dt).total_seconds() * 1000.0
                warnings.append("W01_BACKWARD_TIMESTAMP")
                state.record_backward_delta(delta_ms)
        state.last_valid_dt = event_dt

    # W02: DUPLICATE_RECORD
    rec_tuple = tuple(fields)
    if rec_tuple in state.seen_records:
        warnings.append("W02_DUPLICATE_RECORD")
    else:
        state.seen_records.add(rec_tuple)

    # W03: DATE_FILENAME_MISMATCH
    if len(date_str) >= 4 and date_str[:4] != state.expected_yymm:
        warnings.append("W03_DATE_FILENAME_MISMATCH")

    # W04: EXTREME_HOLD
    if hold_val is not None and hold_val > config.extreme_hold_ms:
        warnings.append("W04_EXTREME_HOLD")

    # W05: SUSPECT_DATE_CODE_VALUE
    if DATE_PATTERN.match(date_str):
        date_num = float(date_str)
        for num_val in (hold_val, latency_val, flight_val):
            if num_val is not None and num_val == date_num:
                warnings.append("W05_SUSPECT_DATE_CODE_VALUE")
                break

    errors.sort()
    warnings.sort()
    return (errors, warnings, event_dt)
