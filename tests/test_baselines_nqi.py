"""Tests for the nQi-style window baseline."""
import numpy as np

from src.baselines_nqi import NqiModel, nqi_window_features
from src.data_loading import Session


def make_session(subject_id, label, cohort):
    n = 40
    return Session(
        session_id=f"{subject_id}_1",
        subject_id=subject_id,
        cohort=cohort,
        label=label,
        hold_times_s=(np.linspace(0.08, 0.20, n)).tolist(),
        press_times_s=np.arange(n, dtype=float).tolist(),
    )


def test_nqi_features_use_seven_window_features():
    features = nqi_window_features(
        [0.1] * 40,
        list(np.arange(40, dtype=float)),
        window_s=90.0,
        min_keystrokes=30,
    )
    assert features.shape == (1, 7)


def test_nqi_model_fits_and_scores_sessions():
    train = [
        make_session("a", 0, "MIT-CS1PD"),
        make_session("b", 1, "MIT-CS1PD"),
    ]
    test = [make_session("c", 0, "MIT-CS2PD")]

    model = NqiModel(n_members=3, seed=42)
    model.fit(train)
    scores = model.scores(test)

    assert len(model.models) == 3
    assert scores.shape == (1,)
    assert np.isfinite(scores[0])
