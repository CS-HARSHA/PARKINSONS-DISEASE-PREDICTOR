# Phase 6 Feature Engineering & Preprocessing Decision Log

**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  
**Date:** 24 October 2026  
**Working Branch:** `praneeth`  

---

## Decision D01: Re-Extraction from Clean Phase 2 Parquet Records (Case 2 Resolution)

- **Context:** Phase 4 data readiness gate and Phase 5 EDA identified that the legacy `features_clean.csv` was invalid: it omitted 102 participants (495 vs 597 eligible), had 28 columns corrupted with shifted YYMMDD date values (>100,000), and clamped negative flight times to NaN.
- **Decision:** Under Section 4 Case 2, re-extract participant-level features directly from the 2,143 accepted Parquet records in `data/phase2_recovery/interim/accepted/` covering the 597 eligible cohort participants.
- **Impact:** Restores the 597-participant cohort and removes legacy table date-corruption artifacts. The accepted-cohort audit observed no negative raw flight_ms values, so rollover preservation cannot be empirically demonstrated in this cohort.

---

## Decision D02: Negative Flight-Time Validation Policy and Cohort Observation

- **Context:** The Phase 2 validator permits signed plain-decimal flight_ms values. The accepted-cohort audit found zero negative values among 40,059,679 accepted flight_ms observations from 2,143 Parquet files.
- **Decision:** Negative flight times remain valid under the validator and are not clamped by that rule. Because no negative event-level values were observed in the accepted cohort, this dataset cannot empirically demonstrate negative-flight preservation.
- **Impact:** Keeps the processing policy compatible with negative values if they occur in another valid dataset; no negative-flight biomarker is empirically demonstrated by this cohort.

---

## Decision D03: Sample Size Thresholds for Higher-Order Distribution Statistics

- **Context:** Sample skewness and sample kurtosis suffer severe finite-sample bias when estimated from very few keystroke events.
- **Decision:** Enforce minimum sample size thresholds:
  - Skewness: minimum N >= 20 valid observations; returns NaN otherwise.
  - Kurtosis: minimum N >= 30 valid observations; returns NaN otherwise.
  - Coefficient of Variation (CV = std / mean): requires N >= 10 and strictly positive mean (mean > 1e-6); returns NaN otherwise.
- **Impact:** Prevents unstable outliers and extreme values caused by sample variance.

---

## Decision D04: Strict Separation of Feature Extraction from Model Preprocessing

- **Context:** Imputing or scaling across the entire participant cohort prior to cross-validation causes subtle data leakage and over-optimistic performance estimates.
- **Decision:** Pipeline A (Feature Extraction) produces unscaled, un-imputed participant features. Pipeline B (Preprocessing) packages reusable, **unfitted** scikit-learn transformers (`SimpleImputer`, `StandardScaler`, `RobustScaler`) that are fitted strictly inside cross-validation folds during Phase 7.
- **Impact:** Eliminates data leakage and guarantees strict methodological compliance with the Phase 3 protocol.
