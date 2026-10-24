# Phase 6 Feature Engineering & Preprocessing Decision Matrix

**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  
**Repository:** `CS-HARSHA/PARKINSONS-DISEASE-PREDICTOR`  
**Author:** Phase 5 Exploratory Data Analysis Hand-off  
**Date:** 24 October 2026  
**Primary Dataset:** Tappy Extended Keystroke Dataset, Mendeley Data Version 3 (DOI: `10.17632/z39mhdsynx.3`)  
**Context:** Grounded in Phase 4 Blocker Statement and Phase 5 EDA Evidence  

---

## 1. Overview and Purpose

Phase 5 Exploratory Data Analysis verified that the legacy table `data/processed/features_clean.csv` cannot be used for machine learning due to:
1. Shifted 6-digit YYMMDD date values contaminating 28 latency columns ($> 100,000$).
2. Total loss of typing overlap dynamics caused by coercing negative flight times ($FT \le 0$) to `NaN`.
3. Cohort under-coverage omitting 102 valid eligible participants (17.1% truncation).

This document establishes the concrete, prioritized decisions that Phase 6 must investigate and resolve when generating the clean feature table from validated Phase 2 records in [`data/interim/accepted/`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/data/interim/accepted/).

---

## 2. Ordered Decisions and Investigation Topics for Phase 6

| Priority | Decision Area | Phase 5 Empirical Finding | Concrete Phase 6 Implementation Requirement |
|---|---|---|---|
| **P0 (Critical)** | **Feature Re-extraction Source** | Legacy `features_clean.csv` is corrupted and incomplete (495 rows). | Re-aggregate timing features from scratch directly from verified Phase 2 Parquet files in [`data/interim/accepted/`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/data/interim/accepted/) across all **597 eligible participants**. |
| **P0 (Critical)** | **Preservation of Rollover (Negative Flight Time)** | Legacy table set `lat2_ms <= 0` to `NaN`, wiping out typing overlap and setting minimum flight time to $\ge 0$ ms. | Compute genuine flight time ($FT = t_{\text{down}, i+1} - t_{\text{up}, i}$), preserving negative values as physically valid rollover. Compute negative flight fraction (`rollover_ratio = (FT < 0).mean()`) as a primary coordination biomarker. |
| **P0 (Critical)** | **Elimination of Date Corruption** | Legacy table had 28 columns with max values $> 100,000$ (e.g. `171226.0`), inflating kurtosis to $> 720,000$. | Leverage Phase 2 rule `W05_SUSPECT_DATE_CODE_VALUE`. Ensure timing aggregates strictly exclude any suspect date codes or apply domain clipping prior to moment calculation. |
| **P1 (High)** | **Participant Coverage & Minimum Keystroke Sensitivity** | Legacy notebook dropped 102 eligible participants with $< 1000$ keystrokes without approval. | Evaluate minimum keystroke thresholding (e.g., full cohort $N=597$ vs $\ge 500$ keys vs $\ge 1000$ keys) within training cross-validation folds to verify stability of skewness and kurtosis estimates. |
| **P1 (High)** | **Handling Sparse Transition Directions** | 54 transition features (e.g., `SL`, `SS`) contain NaNs because participants rarely typed those transitions. | Implement inner-fold imputation (`SimpleImputer(strategy='median')`) within the pipeline, or aggregate directional transitions into higher-level groups: Unimanual (`LL`, `RR`) vs Bimanual (`LR`, `RL`) vs Spacebar (`*S`, `S*`). |
| **P1 (High)** | **Collinearity and Redundancy Mitigation** | Central tendencies (mean, median) and dispersion (std, IQR) have $|r| \ge 0.90$. | Formulate regularized linear models ($L_1$ / ElasticNet) to handle collinearity, or select a non-redundant feature subset based strictly on training-fold variance and domain interpretability. |
| **P2 (Medium)** | **Feature Scaling Strategy** | Timing distributions exhibit moderate right-skewness and heavy tails. | Evaluate `RobustScaler` (median / IQR based) versus `StandardScaler` inside cross-validation pipelines for distance-based models (SVM, KNN, Logistic Regression). Tree-based models (Random Forest) remain scale-invariant. |
| **P2 (Medium)** | **Class Imbalance Strategy** | Training set exhibits ~72% PD vs ~28% Control prevalence. | In accordance with Phase 3 protocol, use **Balanced Accuracy** as the primary selection metric, and evaluate cost-sensitive training (`class_weight='balanced'`) to prevent majority-class bias. |

---

## 3. Recommended Phase 6 Feature Allowlist Formulation

Phase 6 should generate an explicit, leakage-free feature table structured into three canonical families:

1. **Hold-Time Family ($HT$):**
   - `ht_mean`, `ht_std`, `ht_median`, `ht_iqr`, `ht_cv`, `ht_skew`, `ht_kurt`.
2. **Latency Family ($L$):**
   - `lat_mean`, `lat_std`, `lat_median`, `lat_iqr`, `lat_cv`, `lat_skew`, `lat_kurt`.
3. **Flight-Time Family ($FT$):**
   - `ft_mean`, `ft_std`, `ft_median`, `ft_iqr`, `ft_skew`, `ft_kurt`.
   - `ft_rollover_fraction` (fraction of transitions with $FT < 0$).
4. **Transition Direction Breakdown:**
   - Unimanual transition latency vs. bimanual transition latency.
   - Unimanual flight time vs. bimanual flight time.

*Prohibited Exclusions:* `participant_id` (traceability only), `birth_year`, `gender`, `updrs`, `tremors`, `diagnosis_year`, `levodopa`, `da`, `maob`, `other`, `sided`, `impact`.
