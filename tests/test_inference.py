import json
from pathlib import Path

import numpy as np
import pytest

from src.inference import InferenceError, analyse_keystrokes, load_macd_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "macd_model.json"


def increasing_holds(count=220):
    """Create a deterministic sample with adjacent observations in the upper corridor."""
    return np.linspace(0.01, 0.10, count).tolist()


def test_load_existing_exported_model():
    model = load_macd_model(MODEL_PATH)
    assert model["model_type"] == "logistic_regression"
    assert model["feature"] == "macd"
    assert model["ql"] == pytest.approx(0.60)
    assert model["qh"] == pytest.approx(1.00)
    assert model["training_stats"]["min"] < model["training_stats"]["median"]


def test_analysis_returns_expected_numeric_measurements():
    result = analyse_keystrokes(increasing_holds(), model_path=MODEL_PATH, minimum_keystrokes=200)

    assert result["valid_keystrokes"] == 220
    assert result["macd_pairs"] > 0
    assert np.isfinite(result["macd_seconds"])
    assert np.isfinite(result["model_output_score"])
    assert 0.0 < result["model_output_score"] < 1.0
    assert result["mean_hold_seconds"] == pytest.approx(np.mean(increasing_holds()))
    assert result["median_hold_seconds"] == pytest.approx(np.median(increasing_holds()))
    assert "training-set median" in result["relative_training_band"]


def test_invalid_hold_durations_are_filtered_before_count_check():
    values = increasing_holds(205) + [float("nan"), float("inf"), -0.1, 5.0]
    result = analyse_keystrokes(values, model_path=MODEL_PATH, minimum_keystrokes=200)
    assert result["valid_keystrokes"] == 205


def test_rejects_insufficient_valid_keystrokes():
    with pytest.raises(InferenceError, match="at least 200 are required"):
        analyse_keystrokes(increasing_holds(199), model_path=MODEL_PATH, minimum_keystrokes=200)


def test_rejects_more_than_maximum_keystrokes():
    values = increasing_holds(1001)
    with pytest.raises(InferenceError, match="maximum allowed is 1000"):
        analyse_keystrokes(
            values,
            model_path=MODEL_PATH,
            minimum_keystrokes=200,
            maximum_keystrokes=1000,
        )


def test_accepts_maximum_keystroke_count():
    values = increasing_holds(1000)
    result = analyse_keystrokes(
        values,
        model_path=MODEL_PATH,
        minimum_keystrokes=200,
        maximum_keystrokes=1000,
    )
    assert result["valid_keystrokes"] == 1000


def test_rejects_sample_without_qualifying_consecutive_corridor_pairs():
    values = [0.01, 0.10] * 110
    with pytest.raises(InferenceError, match="no consecutive observations"):
        analyse_keystrokes(values, model_path=MODEL_PATH, minimum_keystrokes=200)


def test_missing_model_file_is_reported():
    with pytest.raises(FileNotFoundError, match="Model file was not found"):
        load_macd_model(ROOT / "models" / "missing_model.json")


def test_rejects_non_macd_model(tmp_path):
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    model["feature"] = "mean_hold"
    path = tmp_path / "wrong_model.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(InferenceError, match="does not use the MACD"):
        load_macd_model(path)
