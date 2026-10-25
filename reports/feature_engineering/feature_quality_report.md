> **Historical / superseded Phase 6 report.** This copy has an unsupported statement that negative flight times were observed and preserved. Use the regenerated report at `reports/phase6_recovery/feature_engineering/`.

# Feature Engineering Quality & Integrity Report

**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  
**Dataset:** Tappy Extended Keystroke Dataset, Mendeley Data Version 3 (DOI: `10.17632/z39mhdsynx.3`)  
**Date:** 24 October 2026  
**Validation Verdict:** PASSED  

---

## 1. Executive Summary

- **Eligible Cohort Size:** 597 participants (100% reconciled with Phase 4 split).
- **Unique Participant Identifiers:** 597 (0 duplicates).
- **Approved Predictor Columns:** 153 typing-derived features.
- **Total Keystroke Events Aggregated:** 39,503,643 events.
- **Warning-Tagged Events Retained:** 298,271 events.
- **Date-Corrupted Columns (>100,000):** 0 (0 columns corrupted).
- **Infinite Values:** 0 (0 infinite values).

---

## 2. Cohort Re-Extraction Resolution (Comparison to Legacy Table)

| Metric | Legacy `features_clean.csv` | Phase 6 Re-Engineered Parquet | Integrity Status |
| :--- | :--- | :--- | :--- |
| **Cohort Coverage** | 495 participants (102 omitted) | **597 participants** | **RESOLVED (+102 participants)** |
| **Date Corruption (>100k)** | 28 corrupted columns | **0 corrupted columns** | **RESOLVED (100% clean)** |
| **Negative Flight Times** | Coerced to NaN / Clamped >= 0 | **Preserved (Rollover valid)** | **RESOLVED (Physiologically sound)** |
| **Predictor Separation** | IDs & Labels mixed with features | **Strict allowlist enforced** | **RESOLVED (No leakage)** |

---

## 3. Missingness & Zero-Variance Audit

- **Features with 0% Missingness:** 36 features.
- **Features with Missing Values:** 117 features (directional transitions with zero event occurrences).
- **Constant / Zero-Variance Features:** 0 features.

### Top Features by Missingness

| Feature Name | Missing Count (out of 597) | Missing Percentage | Interpretation |
| :--- | :--- | :--- | :--- |
| `flight_iqr_SS` | 75 | 12.56% | Zero transition occurrences in participant sessions |
| `flight_mean_SS` | 75 | 12.56% | Zero transition occurrences in participant sessions |
| `hold_iqr_SS` | 75 | 12.56% | Zero transition occurrences in participant sessions |
| `flight_std_SS` | 75 | 12.56% | Zero transition occurrences in participant sessions |
| `flight_median_SS` | 75 | 12.56% | Zero transition occurrences in participant sessions |
| `latency_iqr_SS` | 75 | 12.56% | Zero transition occurrences in participant sessions |
| `latency_mean_SS` | 75 | 12.56% | Zero transition occurrences in participant sessions |
| `latency_std_SS` | 75 | 12.56% | Zero transition occurrences in participant sessions |
| `latency_median_SS` | 75 | 12.56% | Zero transition occurrences in participant sessions |
| `hold_median_SS` | 75 | 12.56% | Zero transition occurrences in participant sessions |
| `hold_std_SS` | 75 | 12.56% | Zero transition occurrences in participant sessions |
| `hold_mean_SS` | 75 | 12.56% | Zero transition occurrences in participant sessions |
| `flight_median_SR` | 42 | 7.04% | Zero transition occurrences in participant sessions |
| `hold_median_SR` | 42 | 7.04% | Zero transition occurrences in participant sessions |
| `flight_mean_SR` | 42 | 7.04% | Zero transition occurrences in participant sessions |

---

## 4. Verification Check Details

- [x] Row count matches eligible cohort (597).
- [x] Participant IDs unique and disjoint.
- [x] Target labels complete, valid, and consistent with Phase 2 metadata.
- [x] Predictor allowlist strictly excludes identifiers and clinical covariates.
- [x] Date corruption (>100k) completely absent.
- [x] Negative flight times preserved for rollover typing.
- [x] Zero infinite values across all predictors.