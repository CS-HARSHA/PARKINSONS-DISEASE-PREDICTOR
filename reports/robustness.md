# Robustness report

## Corridor sensitivity

| ql | AUC | CI low | CI high |
|---:|---:|---:|---:|
| 0.0 | 0.8228 | 0.7326 | 0.9020 |
| 0.4 | 0.8427 | 0.7530 | 0.9203 |
| 0.5 | 0.8571 | 0.7691 | 0.9330 |
| 0.6 | 0.8505 | 0.7591 | 0.9302 |
| 0.7 | 0.8400 | 0.7458 | 0.9264 |
| 0.8 | 0.8079 | 0.7021 | 0.8992 |

## Keystroke curve

| N | AUC | CI low | CI high | Sessions shorter than N |
|---:|---:|---:|---:|---:|
| 50 | 0.6999 | 0.5803 | 0.8068 | 0 |
| 100 | 0.7641 | 0.6550 | 0.8544 | 0 |
| 150 | 0.8029 | 0.7004 | 0.8887 | 0 |
| 200 | 0.8272 | 0.7320 | 0.9086 | 0 |
| 300 | 0.8405 | 0.7525 | 0.9225 | 1 |
| 500 | 0.8422 | 0.7470 | 0.9236 | 5 |
| 1000 | 0.8289 | 0.7298 | 0.9159 | 23 |
| full | 0.8505 | 0.7591 | 0.9302 | 116 |

## Timing perturbations

Noise and transformations are applied only to held-out sessions. Offsets and positive scaling preserve the MACD ranking, so AUC is unchanged; browser deployment still needs feature-range and quality warnings.

- jitter=0.002: AUC 0.8549 (CI 0.7636–0.9341)
- jitter=0.005: AUC 0.8494 (CI 0.7547–0.9308)
- jitter=0.01: AUC 0.8538 (CI 0.7597–0.9341)
- jitter=0.02: AUC 0.8499 (CI 0.7586–0.9291)
- quantize=0.001: AUC 0.8527 (CI 0.7613–0.9308)
- quantize=0.005: AUC 0.8533 (CI 0.7630–0.9330)
- quantize=0.016: AUC 0.8544 (CI 0.7641–0.9319)
- offset=-0.02: AUC 0.8505 (CI 0.7591–0.9302)
- offset=-0.01: AUC 0.8505 (CI 0.7591–0.9302)
- offset=-0.005: AUC 0.8505 (CI 0.7591–0.9302)
- offset=0.005: AUC 0.8505 (CI 0.7591–0.9302)
- offset=0.01: AUC 0.8505 (CI 0.7591–0.9302)
- offset=0.02: AUC 0.8505 (CI 0.7591–0.9302)
- scale=0.9: AUC 0.8505 (CI 0.7591–0.9302)
- scale=1.1: AUC 0.8505 (CI 0.7591–0.9302)
- scale=1.25: AUC 0.8505 (CI 0.7591–0.9302)