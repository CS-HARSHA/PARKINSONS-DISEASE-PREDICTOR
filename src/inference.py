"""Runtime inference helpers for the Streamlit keystroke research demo.

This module deliberately returns an experimental model score and descriptive
measurements only.  The exported model does not include a validated decision
threshold, and browser-collected timing has not been established as equivalent
to the PhysioNet acquisition process.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .features import macd_with_count

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "macd_model.json"
MAX_HOLD_SECONDS = 5.0


class InferenceError(ValueError):
    """Raised when captured timing data cannot be analysed safely."""


def load_macd_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load and validate the JSON-exported MACD logistic-regression model."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file was not found: {path}")

    try:
        model = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InferenceError(f"Model JSON is invalid: {path}") from exc

    if not isinstance(model, dict):
        raise InferenceError("The model JSON root must be an object.")
    if model.get("model_type") != "logistic_regression":
        raise InferenceError("The exported model is not logistic regression.")
    if model.get("feature") != "macd":
        raise InferenceError("The exported model does not use the MACD feature.")

    coefficient = model.get("coef", model.get("coefficient"))
    intercept = model.get("intercept")
    ql = model.get("ql", model.get("corridor_low"))
    qh = model.get("qh", model.get("corridor_high"))
    required = {"coefficient": coefficient, "intercept": intercept, "ql": ql, "qh": qh}
    numeric: dict[str, float] = {}
    for name, value in required.items():
        try:
            numeric[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise InferenceError(f"Model is missing a valid {name} value.") from exc
        if not np.isfinite(numeric[name]):
            raise InferenceError(f"Model is missing a valid {name} value.")

    if not 0.0 <= numeric["ql"] < numeric["qh"] <= 1.0:
        raise InferenceError("Model percentile corridor must satisfy 0 <= ql < qh <= 1.")

    stats = model.get("training_stats") or {}
    if not isinstance(stats, dict):
        raise InferenceError("Model training statistics must be a JSON object.")
    min_value = stats.get("min", model.get("training_feature_min"))
    max_value = stats.get("max", model.get("training_feature_max"))
    median_value = stats.get("median", model.get("training_feature_median"))
    try:
        min_value, max_value, median_value = map(float, (min_value, max_value, median_value))
    except (TypeError, ValueError) as exc:
        raise InferenceError("Model is missing valid training feature-range statistics.") from exc
    if not all(np.isfinite(v) for v in (min_value, max_value, median_value)):
        raise InferenceError("Model training feature statistics must be finite.")
    if not min_value <= median_value <= max_value:
        raise InferenceError("Model training feature range or median is inconsistent.")
    try:
        min_pairs = int(model.get("min_corridor_pairs", 1))
        min_keystrokes = int(model.get("min_keystrokes", 20))
    except (TypeError, ValueError) as exc:
        raise InferenceError("Model minimum sample counts must be integers.") from exc
    if min_pairs < 1:
        raise InferenceError("Model minimum corridor-pair count must be at least one.")

    # Normalize keys so callers do not need to know legacy JSON aliases.
    if min_keystrokes < 2:
        raise InferenceError("Model minimum keystroke count must be at least two.")
    model["coef"] = numeric["coefficient"]
    model["intercept"] = numeric["intercept"]
    model["ql"] = numeric["ql"]
    model["qh"] = numeric["qh"]
    model["min_corridor_pairs"] = min_pairs
    model["min_keystrokes"] = min_keystrokes
    model["training_stats"] = {
        "min": min_value,
        "max": max_value,
        "median": median_value,
    }
    return model


def _valid_hold_times(hold_times_s: Iterable[float]) -> np.ndarray:
    """Keep finite browser hold durations in the project's accepted range."""
    try:
        values = np.asarray(list(hold_times_s), dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise InferenceError("Keystroke timings must be a one-dimensional numeric sequence.") from exc

    return values[
        np.isfinite(values)
        & (values >= 0.0)
        & (values < MAX_HOLD_SECONDS)
    ]


def _sigmoid(value: float) -> float:
    """Numerically stable logistic sigmoid."""
    if value >= 0:
        exp_value = np.exp(-value)
        return float(1.0 / (1.0 + exp_value))
    exp_value = np.exp(value)
    return float(exp_value / (1.0 + exp_value))


def analyse_keystrokes(
    hold_times_s: Iterable[float],
    *,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    minimum_keystrokes: int | None = None,
    maximum_keystrokes: int | None = None,
) -> dict[str, Any]:
    """Extract MACD and descriptive statistics, then run the exported model.

    `minimum_keystrokes` and `maximum_keystrokes` can enforce an application-
    specific sample-size range.
    The returned logistic output is an uncalibrated research-model score, not a
    clinical probability and not a diagnosis.
    """
    model = load_macd_model(model_path)
    holds = _valid_hold_times(hold_times_s)

    required_count = max(
        model["min_keystrokes"],
        int(minimum_keystrokes) if minimum_keystrokes is not None else 0,
    )
    maximum_count = (
        int(maximum_keystrokes)
        if maximum_keystrokes is not None
        else None
    )
    if maximum_count is not None and maximum_count < required_count:
        raise InferenceError(
            "Maximum keystroke count cannot be less than the minimum."
        )
    if maximum_count is not None and holds.size > maximum_count:
        raise InferenceError(
            f"Received {holds.size} valid keystrokes; "
            f"the maximum allowed is {maximum_count}."
        )
    if required_count < 2:
        raise InferenceError("The minimum keystroke count must be at least two.")
    if holds.size < required_count:
        raise InferenceError(
            f"Only {holds.size} valid keystrokes were received; "
            f"at least {required_count} are required. Please repeat the test."
        )

    macd_value, pair_count = macd_with_count(holds, ql=model["ql"], qh=model["qh"])
    if not np.isfinite(macd_value):
        raise InferenceError(
            "MACD could not be calculated because no consecutive observations "
            "fell inside the configured percentile corridor."
        )
    if pair_count < model["min_corridor_pairs"]:
        raise InferenceError(
            f"Only {pair_count} qualifying MACD pairs were found; "
            f"the model requires at least {model['min_corridor_pairs']}."
        )

    logit = model["coef"] * float(macd_value) + model["intercept"]
    model_output = _sigmoid(float(logit))

    training = model["training_stats"]
    below_range = bool(macd_value < training["min"])
    above_range = bool(macd_value > training["max"])
    median = training["median"]
    if macd_value < median * 0.8:
        relative_band = "Lower than the training-set median"
    elif macd_value > median * 1.2:
        relative_band = "Higher than the training-set median"
    else:
        relative_band = "Within 20% of the training-set median"

    return {
        "valid_keystrokes": int(holds.size),
        "macd_pairs": int(pair_count),
        "macd_seconds": float(macd_value),
        "mean_hold_seconds": float(np.mean(holds)),
        "median_hold_seconds": float(np.median(holds)),
        "std_hold_seconds": float(np.std(holds, ddof=0)),
        "model_output_score": model_output,
        "relative_training_band": relative_band,
        "outside_training_range": bool(below_range or above_range),
        "below_training_range": below_range,
        "above_training_range": above_range,
        "training_feature_min": training["min"],
        "training_feature_max": training["max"],
        "training_feature_median": training["median"],
        "percentile_low": model["ql"],
        "percentile_high": model["qh"],
        "model_disclaimer": model.get(
            "disclaimer", "Research demonstration only; not a medical diagnosis."
        ),
    }
