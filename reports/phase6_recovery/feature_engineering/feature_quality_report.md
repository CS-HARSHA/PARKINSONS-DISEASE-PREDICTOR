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
- **Date-Corruption Signature Columns:** 0.
- **Predictor Columns Containing Infinite Values:** 0.
- **Participants with Negative Flight-Time Minimum:** 0 of 597.
- **Minimum Observed Participant Flight Time:** 1.0 ms (`flight_min`).

---

## 2. Cohort Re-Extraction Resolution (Comparison to Legacy Table)

| Metric | Legacy `features_clean.csv` | Phase 6 Re-Engineered Parquet | Integrity Status |
| :--- | :--- | :--- | :--- |
| **Cohort Coverage** | 495 participants (102 omitted) | **597 participants** | **RESOLVED (+102 participants)** |
| **Date Corruption (>100k)** | 28 corrupted columns | **0 corrupted columns** | **RESOLVED (100% clean)** |
| **Negative Flight Times** | Legacy-table issue requiring review | 0 of 597 participants have `flight_min < 0` | Empirical preservation is not demonstrated when no negative minima are present |
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
- Negative-flight observation: 0 participants have a negative participant-level `flight_min`; this does not prove negative event-level intervals were observed or preserved.
- [x] Zero infinite values across all predictors.