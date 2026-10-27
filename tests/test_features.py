import numpy as np

from src.features import macd


def test_macd_uses_only_pairs_with_both_values_in_corridor():
    x = np.array([1.0, 2.0, 3.0, 100.0])
    # 60th percentile is 2.8; only pair (3,100) remains in the inclusive corridor.
    assert macd(x, ql=0.60, qh=1.0) == 97.0


def test_macd_is_mean_absolute_consecutive_change():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert macd(x, ql=0.0, qh=1.0) == 1.0
