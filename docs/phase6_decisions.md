> **Superseded copy.** Refer to `reports/phase6_recovery/docs/phase6_decisions.md` for the current Phase 6 decisions.

# Phase 6 Feature Engineering & Preprocessing Decision Log

**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  
**Date:** 24 October 2026  
**Working Branch:** `praneeth`  

---

## Decision D01: Re-Extraction from Clean Phase 2 Parquet Records (Case 2 Resolution)

- **Context:** Phase 4 data readiness gate and Phase 5 EDA identified that the legacy `features_clean.csv` was invalid: it omitted 102 participants (495 vs 597 eligible), had 28 columns corrupted with shifted YYMMDD date values (>100,000), and clamped negative flight times to NaN.
- **Decision:** Under Section 4 Case 2, re-extract participant-level features directly from the 2,143 accepted Parquet records in `data/interim/accepted/` covering all 597 eligible cohort participants.
- **Impact:** Restores 100% cohort coverage (597 participants: 432 PD, 165 Control), completely eliminates date corruption, and preserves natural rollover typing dynamics.

---

## Decision D02: Preservation of Negative Flight Times (Rollover Typing)

- **Context:** During natural fluid keyboard typing, typists frequently initiate a keystroke before releasing the preceding key ($T_{press, n+1} < T_{release, n}$).
- **Decision:** In strict alignment with Phase 2 Decision D04, flight times < 0.0 ms are recognized as mathematically valid physical quantities and preserved without truncation or clamping.
- **Impact:** Enables capture of typing fluency and motor coordination differences between PD and Control participants.

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
