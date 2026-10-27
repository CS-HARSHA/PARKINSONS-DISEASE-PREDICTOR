from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config


def fit_logistic(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(random_state=config.RANDOM_SEED)
    model.fit(np.asarray(features, dtype=float).reshape(-1, 1), np.asarray(labels, dtype=int))
    return model


def predict_probability(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    return model.predict_proba(np.asarray(features, dtype=float).reshape(-1, 1))[:, 1]
