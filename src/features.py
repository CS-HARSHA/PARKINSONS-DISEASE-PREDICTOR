from __future__ import annotations

import numpy as np


def macd(hold_times: np.ndarray, ql: float = 0.60, qh: float = 1.00) -> float:
    """Mean absolute consecutive change when both values are in a quantile corridor."""
    x = np.asarray(hold_times, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    low, high = np.quantile(x, [ql, qh])
    keep = (x >= low) & (x <= high)
    pair_keep = keep[:-1] & keep[1:]
    if not np.any(pair_keep):
        return float("nan")
    return float(np.mean(np.abs(np.diff(x)[pair_keep])))


def stdev(hold_times: np.ndarray) -> float:
    x = np.asarray(hold_times, dtype=float)
    return float(np.std(x, ddof=0))


def macd_log(hold_times: np.ndarray, ql: float = 0.60, qh: float = 1.00) -> float:
    x = np.asarray(hold_times, dtype=float)
    return macd(np.log1p(x), ql, qh)


def macd_mean_normalized(hold_times: np.ndarray, ql: float = 0.60, qh: float = 1.00) -> float:
    x = np.asarray(hold_times, dtype=float)
    mean = np.nanmean(x)
    return macd(x / mean, ql, qh) if mean > 0 else float("nan")
