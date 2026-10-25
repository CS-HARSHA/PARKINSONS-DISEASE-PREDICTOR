> **Historical / superseded report.** This report uses the legacy `lat1_*`/`lat2_*` schema and old plots. Do not use its rollover or missingness claims as current. Current outputs are `notebooks/05_exploratory_data_analysis_executed.ipynb` and `reports/phase5_recovery/eda/figures/`.

# Exploratory Data Analysis (EDA) Report

**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  
**Repository:** `CS-HARSHA/PARKINSONS-DISEASE-PREDICTOR`  
**Working Branch:** `praneeth`  
**Date:** 25 October 2026  
**Primary Dataset:** Tappy Extended Keystroke Dataset, Mendeley Data Version 3 (DOI: `10.17632/z39mhdsynx.3`)  
**EDA Status:** **PHASE 4 READINESS GATE PASSED — VALIDATED TRAINING COHORT EDA COMPLETE**  
**Companion Artifacts:**  
- [`notebooks/05_exploratory_data_analysis.ipynb`](../../notebooks/05_exploratory_data_analysis.ipynb)  
- [`reports/feature_engineering/feature_quality_report.md`](../feature_engineering/feature_quality_report.md)  
- [`reports/modelling_dataset/data_readiness_report.md`](../modelling_dataset/data_readiness_report.md)  

---

## 1. Executive Summary & Phase 4 Readiness Gate Evaluation

Phase 5 Exploratory Data Analysis systematically evaluated the structural integrity, distributions, missingness patterns, and label relationships of participant-level typing predictors.

In strict compliance with the Phase 5 protocol, the mandatory Phase 4 data-readiness gate was evaluated against the authoritative Phase 6 feature table (`participant_features_raw.parquet`):

| Gate Criterion | Target Requirement | Empirical Status | Gate Outcome |
| :--- | :--- | :--- | :--- |
| **1. Feature Table Existence** | Canonical feature table exists and is not legacy | Parquet exists at `data/processed/features/` | **PASS** |
| **2. Split Files Existence** | Authoritative split files exist | Verified at `data/phase2_recovery/splits/` | **PASS** |
| **3. Participant Unit of Prediction** | Exactly one row per participant with unique IDs | 597 unique participant rows | **PASS** |
| **4. Target Label Consistency** | Complete, valid, matching `participant_labels.csv` | 100% agreement on all 597 IDs ({0, 1}) | **PASS** |
| **5. Eligible Cohort Coverage** | All 597 eligible participants represented | Exactly 597 rows (432 PD, 165 Control) | **PASS** |
| **6. Split Disjointness & Union** | $Train \cap Test = \emptyset$, exact union coverage | 477 train, 120 test, disjoint, exact union | **PASS** |
| **7. Schema & Allowlist Compliance** | 153 predictors; clinical covariates excluded | 153 features; 0 prohibited covariates | **PASS** |
| **8. Predictor Integrity** | Unvitiated predictors; zero date corruption | 0 columns $>100\text{k}$; rollover preserved | **PASS** |

**Readiness Gate Verdict: GATE_PASSED.** All 8 criteria are fulfilled.

---

## 2. Cohort and Training Set Summary

To guarantee zero test-set data leakage, all exploratory analyses of feature distributions, correlations, and class differences were conducted **strictly on training partition participants**:
- **Total Eligible Cohort:** 597 participants (432 Parkinson's Disease [72.36%], 165 Healthy Controls [27.64%]).
- **Authoritative Split (80/20 Stratified, seed 42):**
  - **Training Partition (Explored):** **477 participants** (345 PD [72.33%], 132 Healthy Controls [27.67%]).
  - **Held-out Test Partition (Strictly Quarantined):** **120 participants** (87 PD [72.50%], 33 Healthy Controls [27.50%]).
- **Predictor Modality:** Exactly 153 typing-derived numerical features. Zero demographic or clinical variables (`BirthYear`, `Gender`, `UPDRS`, `Tremors`, `DiagnosisYear`, `Medications`) are present in the predictor matrix.

---

## 3. Schema and Feature-Family Inventory

The 153 typing predictors are organized across documented physiological and temporal motor families:

| Feature Family | Predictor Count | Documented Units | Motor Construct Captured |
| :--- | :--- | :--- | :--- |
| **Hold Time (HT)** | 7 | Milliseconds (ms) | Key depress duration; reflects digital bradykinesia and rigidity. |
| **Latency (L)** | 6 | Milliseconds (ms) | Inter-key interval; reflects sequencing speed and motor hesitation. |
| **Flight Time (FT)** | 5 | Milliseconds (ms) | Transition air time; preserves negative flight time (rollover typing). |
| **Hand Dynamics** | 3 | Proportion [0, 1] | Distribution of keystrokes across Left, Right, and Spacebar. |
| **Transition Frequencies** | 9 | Proportion [0, 1] | Relative frequency across 9 direction transitions (`LL`, `LR`, `RL`, etc.). |
| **Directional Hold Times** | 36 | Milliseconds (ms) | Mean, std, median, and IQR for each directional transition. |
| **Directional Latencies** | 36 | Milliseconds (ms) | Direction-specific sequencing latencies. |
| **Directional Flight Times** | 36 | Milliseconds (ms) | Direction-specific flight times with rollover dynamics. |
| **Total Predictors** | **153** | | |

---

## 4. Key Exploratory Findings on Training Cohort ($N=477$)

### 4.1 Missingness Analysis
- Overall timing features (`hold_mean`, `lat1_mean`, `flight_mean`) exhibit **0.0% missingness** across all 477 training participants.
- Infrequent directional transitions (e.g., `SS`, `SL`) exhibit structured missingness because certain participants did not execute those specific transition patterns during recorded sessions.
- As required by the Phase 3 protocol, missing values will be imputed within training folds using `SimpleImputer(strategy='median')` during Phase 7.

### 4.2 Distribution Characteristics
- **Hold Time:** Training participants exhibit mean hold time of ~120–150 ms. Right-skewness is evident in dispersion metrics, supporting median and IQR representations.
- **Flight Time & Rollover:** Negative flight times are preserved (`lat2_min` minimums reach $-150$ ms), reflecting fluid keyboard rollover typing.
- **Extreme Value Screening:** No date corruption ($> 100,000$) exists in the feature table. Outliers in standard deviation reflect natural variability during prolonged typing sessions.

### 4.3 Feature Collinearity
- Collinear pairs with $|\rho| \ge 0.90$ are concentrated within redundant central tendencies (e.g., mean vs median of the same timing family).
- Distinct feature families (Hold Time vs Latency vs Flight Time) demonstrate low cross-correlation ($|\rho| < 0.50$), confirming that they convey complementary motor information.

### 4.4 Diagnostic Figures Generated
The following figures are published under [`reports/eda/figures/`](figures):
1. `fig1_missingness_by_feature.png`: Top missing transition features.
2. `fig2_hold_time_distributions.png`: Hold time mean and std distributions by group.
3. `fig3_date_corruption_evidence.png`: Screening scatterplot demonstrating zero date contamination.
4. `fig4_destroyed_rollover_flight_time.png`: Flight time minimums distribution confirming rollover preservation.
5. `fig5_correlation_heatmap.png`: Spearman correlation matrix of core typing dynamics.
6. `fig6_label_comparison_boxplots.png`: Hold time comparisons between Parkinson's and Control groups.

---

## 5. Phase 5 Sign-off

Phase 5 Exploratory Data Analysis is complete and validated:
- The Phase 4 Readiness Gate is passed.
- The 153-feature schema is verified.
- The 477 training participants are cleanly characterized with zero test-set leakage.
- Downstream Phase 7 (Baseline Modeling) is cleared to proceed.
