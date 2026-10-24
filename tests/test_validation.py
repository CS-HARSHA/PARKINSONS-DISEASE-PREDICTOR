"""Unit tests for record validation rules (Task T5)."""
import pytest
from datetime import datetime
from src.data_pipeline.config import load_config
from src.data_pipeline.validation import (
    validate_record,
    FileValidationState,
    PLAIN_DECIMAL_PATTERN,
)

CONFIG = load_config("configs/pipeline.toml")


def make_valid_fields(
    user_key="0EA27ICBLF",
    date="160723",
    time="17:35:20.516",
    hand="L",
    hold="0199.2",
    direction="LL",
    latency="0453.1",
    flight="0281.3",
):
    return [user_key, date, time, hand, hold, direction, latency, flight]


def test_plain_decimal_pattern():
    # Valid plain decimals
    assert PLAIN_DECIMAL_PATTERN.match("0")
    assert PLAIN_DECIMAL_PATTERN.match("123")
    assert PLAIN_DECIMAL_PATTERN.match("123.45")
    assert PLAIN_DECIMAL_PATTERN.match("-123.45")
    assert PLAIN_DECIMAL_PATTERN.match("0078.1")
    assert PLAIN_DECIMAL_PATTERN.match("-0050.0")

    # Invalid plain decimals
    assert not PLAIN_DECIMAL_PATTERN.match("+100")
    assert not PLAIN_DECIMAL_PATTERN.match("1_000")
    assert not PLAIN_DECIMAL_PATTERN.match("1e5")
    assert not PLAIN_DECIMAL_PATTERN.match("nan")
    assert not PLAIN_DECIMAL_PATTERN.match("NaN")
    assert not PLAIN_DECIMAL_PATTERN.match("inf")
    assert not PLAIN_DECIMAL_PATTERN.match("Infinity")
    assert not PLAIN_DECIMAL_PATTERN.match(" 100")
    assert not PLAIN_DECIMAL_PATTERN.match("100 ")
    assert not PLAIN_DECIMAL_PATTERN.match("0066,4")  # comma decimal
    assert not PLAIN_DECIMAL_PATTERN.match("")


def test_valid_record_passes_cleanly():
    fields = make_valid_fields()
    state = FileValidationState(filename="0EA27ICBLF_1607.txt", expected_user_key="0EA27ICBLF", expected_yymm="1607")
    errors, warnings, dt = validate_record(fields, state, CONFIG)
    assert errors == []
    assert warnings == []
    assert dt == datetime(2016, 7, 23, 17, 35, 20, 516000)


def test_e01_bad_field_count():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")
    errors, warnings, _ = validate_record(["A", "B", "C"], state, CONFIG)
    assert "E01_FIELD_COUNT" in errors


def test_e03_and_e04_userkey_errors():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")

    # Bad user key format (lowercase or wrong length)
    fields = make_valid_fields(user_key="short")
    errors, _, _ = validate_record(fields, state, CONFIG)
    assert "E03_BAD_USERKEY" in errors
    assert "E04_USERKEY_MISMATCH" in errors

    # Valid key format but mismatch with filename
    fields2 = make_valid_fields(user_key="DIFFKEY001")
    errors2, _, _ = validate_record(fields2, state, CONFIG)
    assert "E03_BAD_USERKEY" not in errors2
    assert "E04_USERKEY_MISMATCH" in errors2


def test_e05_bad_date():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")

    # 5-digit date
    assert "E05_BAD_DATE" in validate_record(make_valid_fields(date="16072"), state, CONFIG)[0]
    # Month 13
    assert "E05_BAD_DATE" in validate_record(make_valid_fields(date="161301"), state, CONFIG)[0]
    # Day 32
    assert "E05_BAD_DATE" in validate_record(make_valid_fields(date="160732"), state, CONFIG)[0]
    # 30 February
    assert "E05_BAD_DATE" in validate_record(make_valid_fields(date="160230"), state, CONFIG)[0]
    # 29 Feb on leap year (2016 is leap) -> VALID
    assert "E05_BAD_DATE" not in validate_record(make_valid_fields(date="160229"), state, CONFIG)[0]
    # 29 Feb on non-leap year (2017 is not leap) -> INVALID
    assert "E05_BAD_DATE" in validate_record(make_valid_fields(date="170229"), state, CONFIG)[0]


def test_e06_bad_time():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")

    # Hour 24
    assert "E06_BAD_TIME" in validate_record(make_valid_fields(time="24:00:00.000"), state, CONFIG)[0]
    # Minute 60
    assert "E06_BAD_TIME" in validate_record(make_valid_fields(time="12:60:00.000"), state, CONFIG)[0]
    # Second 60
    assert "E06_BAD_TIME" in validate_record(make_valid_fields(time="12:00:60.000"), state, CONFIG)[0]
    # Missing milliseconds
    assert "E06_BAD_TIME" in validate_record(make_valid_fields(time="12:00:00"), state, CONFIG)[0]


def test_e07_bad_hand():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")
    assert "E07_BAD_HAND" in validate_record(make_valid_fields(hand="X"), state, CONFIG)[0]
    assert "E07_BAD_HAND" in validate_record(make_valid_fields(hand="l"), state, CONFIG)[0]  # lowercase


def test_e08_bad_hold():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")
    # Negative hold rejected
    assert "E08_BAD_HOLD" in validate_record(make_valid_fields(hold="-10.0"), state, CONFIG)[0]
    # Non-plain decimal
    assert "E08_BAD_HOLD" in validate_record(make_valid_fields(hold="0078,1"), state, CONFIG)[0]
    assert "E08_BAD_HOLD" in validate_record(make_valid_fields(hold="nan"), state, CONFIG)[0]
    # Zero hold is valid
    assert "E08_BAD_HOLD" not in validate_record(make_valid_fields(hold="0000.0"), state, CONFIG)[0]


def test_e09_bad_direction():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")
    assert "E09_BAD_DIRECTION" in validate_record(make_valid_fields(direction="XX"), state, CONFIG)[0]
    assert "E09_BAD_DIRECTION" in validate_record(make_valid_fields(direction="L"), state, CONFIG)[0]


def test_e10_bad_latency():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")
    assert "E10_BAD_LATENCY" in validate_record(make_valid_fields(latency="-5.0"), state, CONFIG)[0]
    assert "E10_BAD_LATENCY" in validate_record(make_valid_fields(latency="inf"), state, CONFIG)[0]


def test_e11_bad_flight_and_negative_flight_permitted():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")
    # Negative flight is valid and must NOT trigger E11
    assert "E11_BAD_FLIGHT" not in validate_record(make_valid_fields(flight="-0050.2"), state, CONFIG)[0]
    # Non-decimal flight triggers E11
    assert "E11_BAD_FLIGHT" in validate_record(make_valid_fields(flight="0281,3"), state, CONFIG)[0]
    assert "E11_BAD_FLIGHT" in validate_record(make_valid_fields(flight="abc"), state, CONFIG)[0]


def test_warnings_w01_backward_timestamp():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")
    fields1 = make_valid_fields(time="10:00:05.000")
    _, w1, _ = validate_record(fields1, state, CONFIG)
    assert "W01_BACKWARD_TIMESTAMP" not in w1

    # Record 2 is earlier than Record 1
    fields2 = make_valid_fields(time="10:00:02.000")
    _, w2, _ = validate_record(fields2, state, CONFIG)
    assert "W01_BACKWARD_TIMESTAMP" in w2


def test_warnings_w02_duplicate_record():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")
    fields = make_valid_fields()
    _, w1, _ = validate_record(fields, state, CONFIG)
    assert "W02_DUPLICATE_RECORD" not in w1

    # Identical record again
    _, w2, _ = validate_record(fields, state, CONFIG)
    assert "W02_DUPLICATE_RECORD" in w2


def test_warnings_w03_date_filename_mismatch():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")
    fields = make_valid_fields(date="160815")  # 1608 != 1607
    _, warnings, _ = validate_record(fields, state, CONFIG)
    assert "W03_DATE_FILENAME_MISMATCH" in warnings


def test_warnings_w04_extreme_hold():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")
    fields = make_valid_fields(hold="2500.0")  # > 2000.0
    _, warnings, _ = validate_record(fields, state, CONFIG)
    assert "W04_EXTREME_HOLD" in warnings


def test_warnings_w05_suspect_date_code_value():
    state = FileValidationState("0EA27ICBLF_1607.txt", "0EA27ICBLF", "1607")
    # Hold matches date 160723 numerically
    fields = make_valid_fields(date="160723", hold="160723.0")
    _, warnings, _ = validate_record(fields, state, CONFIG)
    assert "W05_SUSPECT_DATE_CODE_VALUE" in warnings
