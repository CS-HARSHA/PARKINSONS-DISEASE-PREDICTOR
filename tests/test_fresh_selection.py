"""Tests for FRESH extraction and training-only feature selection helpers."""
import numpy as np
import pandas as pd

from src.fresh_features import rank_features, select_top_m


def test_fresh_ranking_excludes_session_id():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({
        "session_id": [f"s{i}" for i in range(20)],
        "strong": np.r_[np.zeros(10), np.ones(10)] + rng.normal(0, 0.01, 20),
        "weak": rng.normal(0, 1, 20),
    })
    labels = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]

    ranked = rank_features(frame, labels)
    selected = select_top_m(frame, labels, 1)

    assert "session_id" not in ranked
    assert selected[0] in {"strong", "weak"}
    assert len(selected) == 1


def test_fresh_selection_returns_requested_number_of_features():
    frame = pd.DataFrame({
        "session_id": [f"s{i}" for i in range(12)],
        "f1": np.arange(12, dtype=float),
        "f2": np.arange(12, dtype=float)[::-1],
        "f3": np.ones(12),
    })
    labels = np.array([0, 1] * 6)

    selected = select_top_m(frame, labels, 2)

    assert len(selected) == 2
    assert all(name != "session_id" for name in selected)
