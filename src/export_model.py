from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config


def export_logistic_model(model: LogisticRegression, features: np.ndarray, output: Path, feature_name: str = "macd") -> None:
    x = np.asarray(features, dtype=float)
    payload = {
        "model_type": "logistic_regression",
        "feature": feature_name,
        "coefficient": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
        "corridor_low": config.CORRIDOR_LOW,
        "corridor_high": config.CORRIDOR_HIGH,
        "training_feature_min": float(np.nanmin(x)),
        "training_feature_max": float(np.nanmax(x)),
        "training_feature_median": float(np.nanmedian(x)),
        "disclaimer": "Research screening demo only; not a medical diagnosis.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
