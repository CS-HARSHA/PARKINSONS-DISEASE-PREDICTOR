"""Tests for the mean-hold, standard-deviation, and nQi-style baselines."""
import numpy as np

from src.baselines_nqi import nqi_window_features
from src.baselines_simple import MeanHoldModel, StdHoldModel
from src.data_loading import Session


def make_session(subject_id, label, values, cohort="MIT-CS1PD"):
    values = list(values)
    return Session(
        session_id=f"{subject_id}_1",
        subject_id=subject_id,
        cohort=cohort,
        label=label,
        hold_times_s=values,
        press_times_s=list(np.arange(len(values), dtype=float)),
    )


def test_mean_hold_baseline_returns_one_score_per_session():
    train = [
        make_session("a", 0, [0.10, 0.11, 0.12]),
        make_session("b", 1, [0.20, 0.21, 0.22]),
    ]
    test = [
        make_session("c", 0, [0.10, 0.11, 0.12]),
        make_session("d", 1, [0.20, 0.21, 0.22]),
    ]

    model = MeanHoldModel()
    model.fit(train)
    scores = model.scores(test)

    assert len(scores) == len(test)
    assert np.all(np.isfinite(scores))


def test_standard_deviation_baseline_returns_one_score_per_session():
    train = [
        make_session("a", 0, [0.10, 0.10, 0.10]),
        make_session("b", 1, [0.10, 0.20, 0.30]),
    ]
    test = [
        make_session("c", 0, [0.10, 0.10, 0.10]),
        make_session("d", 1, [0.10, 0.20, 0.30]),
    ]

    model = StdHoldModel()
    model.fit(train)
    scores = model.scores(test)

    assert len(scores) == len(test)
    assert np.all(np.isfinite(scores))


def test_nqi_window_features_have_expected_shape():
    hold_times = [0.10] * 40
    press_times = list(np.arange(40, dtype=float))

    features = nqi_window_features(
        hold_times,
        press_times,
        window_s=90.0,
        min_keystrokes=30,
    )

    assert features.ndim == 2
    assert features.shape[1] == 7
    assert features.shape[0] >= 1


def test_nqi_window_features_skip_short_windows():
    hold_times = [0.10] * 20
    press_times = list(np.arange(20, dtype=float))

    features = nqi_window_features(
        hold_times,
        press_times,
        window_s=90.0,
        min_keystrokes=30,
    )

    assert features.shape == (0, 7)
