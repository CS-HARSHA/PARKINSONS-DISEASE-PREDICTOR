# Model comparison

All numbers below are generated from files in `results/metrics/`. The pooled AUC is the primary comparable result; direction-specific values remain important because the two cohorts are not identical.

| Model | Pooled AUC | CI low | CI high | Direction A | Direction B |
|---|---:|---:|---:|---:|---:|
| MACD + logistic regression | 0.8505 | 0.7591 | 0.9302 | 0.8222 | 0.8803 |
| baseline_mean | 0.5565 | 0.4319 | 0.6789 | 0.7056 | 0.7906 |
| baseline_std | 0.5936 | 0.4657 | 0.7126 | 0.7625 | 0.9017 |
| nQi-style SVR ensemble | 0.7447 | 0.6379 | 0.8466 | 0.7319 | 0.7991 |
| macd | 0.8505 | 0.7591 | 0.9302 | 0.8222 | 0.8803 |
| macd_log | 0.8544 | 0.7647 | 0.9314 | 0.8292 | 0.8803 |
| macd_norm | 0.7436 | 0.6323 | 0.8400 | 0.7444 | 0.7650 |
| Classifier/feature ablation | 0.8505 | 0.7591 | 0.9302 | 0.8222 | 0.8803 |
| Classifier/feature ablation | 0.8505 | 0.7591 | 0.9302 | 0.8222 | 0.8803 |
| Classifier/feature ablation | 0.8505 | 0.7591 | 0.9302 | 0.8222 | 0.8803 |
| Classifier/feature ablation | 0.8505 | 0.7591 | 0.9302 | 0.8222 | 0.8803 |
| Classifier/feature ablation | 0.6107 | 0.4806 | 0.7270 | 0.8028 | 0.9103 |
| Classifier/feature ablation | 0.5825 | 0.4551 | 0.7038 | 0.7542 | 0.8419 |

## FRESH curve

| Top m features | Pooled AUC | CI low | CI high |
|---:|---:|---:|---:|
| 1 | 0.7492 | 0.6406 | 0.8516 |
| 2 | 0.7458 | 0.6362 | 0.8406 |
| 3 | 0.7547 | 0.6462 | 0.8466 |
| 4 | 0.7353 | 0.6262 | 0.8333 |
| 5 | 0.7364 | 0.6262 | 0.8356 |
| 6 | 0.7458 | 0.6368 | 0.8433 |
| 7 | 0.7481 | 0.6401 | 0.8439 |
| 8 | 0.7436 | 0.6340 | 0.8400 |
| 9 | 0.7453 | 0.6345 | 0.8406 |
| 10 | 0.7569 | 0.6500 | 0.8505 |

The random forest and other classifiers are comparison analyses. A one-feature monotone model shares the common feature ranking for pooled AUC, while the fitted classifier remains part of the ablation. No comparison is promoted to clinical use.
