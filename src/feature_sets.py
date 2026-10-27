from __future__ import annotations

import numpy as np

from .features import macd, stdev

FEATURE_NAMES = ("macd", "stdev", "mean_hold", "median_hold")


def feature_vector(hold_times: np.ndarray, names: tuple[str, ...] = FEATURE_NAMES) -> dict[str, float]:
    x = np.asarray(hold_times, dtype=float)
    values = {
        "macd": macd(x),
        "stdev": stdev(x),
        "mean_hold": float(np.mean(x)),
        "median_hold": float(np.median(x)),
    }
    return {name: values[name] for name in names}
