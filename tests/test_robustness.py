"""Tests for held-out timing perturbation and truncation helpers."""
import numpy as np

from src.data_loading import Session
from src.robustness import transform_sessions, truncate_sessions


def session():
    return Session(
        session_id="s_1",
        subject_id="s",
        cohort="MIT-CS1PD",
        label=0,
        hold_times_s=[0.10, 0.12, 0.14, 0.16],
        press_times_s=[0.0, 1.0, 2.0, 3.0],
    )


def test_offset_preserves_length_and_changes_values():
    transformed = transform_sessions([session()], "offset", 0.01)
    assert len(transformed) == 1
    assert len(transformed[0].hold_times_s) == 4
    assert np.allclose(transformed[0].hold_times_s, [0.11, 0.13, 0.15, 0.17])


def test_truncate_sessions_updates_metadata_and_counts_short_sessions():
    truncated, short = truncate_sessions([session()], 3)
    assert short == 0
    assert truncated[0].n_keystrokes == 3
    assert len(truncated[0].hold_times_s) == 3
    assert len(truncated[0].press_times_s) == 3
