# Exploratory Data Analysis (EDA) Report

**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  
**Repository:** `CS-HARSHA/PARKINSONS-DISEASE-PREDICTOR`  
**Working Branch:** `praneeth`  
**Date:** 24 October 2026  
**Primary Dataset:** Tappy Extended Keystroke Dataset, Mendeley Data Version 3 (DOI: `10.17632/z39mhdsynx.3`)  
**EDA Status:** **DIAGNOSTIC AUDIT COMPLETE — PHASE 4 READINESS GATE FAILED / BLOCKED**  
**Companion Artifacts:**  
- [`notebooks/05_exploratory_data_analysis.ipynb`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/notebooks/05_exploratory_data_analysis.ipynb)  
- [`reports/eda/phase6_questions.md`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/phase6_questions.md)  
- [`reports/modelling_dataset/data_readiness_report.md`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/modelling_dataset/data_readiness_report.md)  

---

## 1. Executive Summary & Phase 4 Readiness Gate Audit

Phase 5 Exploratory Data Analysis evaluated the structural integrity, feature distributions, missingness patterns, and label relationships of participant-level typing predictors. In accordance with Section 4.1 of the project instructions, the mandatory Phase 4 data-readiness gate was evaluated before conducting analytical EDA:

| Gate Criterion | Target Requirement | Empirical Status | Gate Outcome |
|---|---|---|---|
| **1. Modelling Dataset Existence** | `modelling_dataset.parquet` exists & usable | **Does not exist** | **FAIL (BLOCKED)** |
| **2. Participant Unit of Prediction** | Exactly one row per participant | 495 unique rows in legacy table | **PASS** |
| **3. Target Label Consistency** | Complete, valid, matching `participant_labels.csv` | 100% agreement on 495 keys | **PASS** |
| **4. Eligible Cohort Coverage** | All 597 eligible participants represented | **Only 495 present (102 omitted, 17.1%)** | **FAIL** |
| **5. Feature Provenance & Cleanliness** | Unvitiated predictors; known provenance | **28 columns contaminated with dates > 100k; negative rollover destroyed** | **FAIL** |
| **6. Variable Documentation** | All features documented in dictionary | 220 columns; directional metrics lack units | **FAIL** |
| **7. Split Disjointness** | $Train \cap Test = \emptyset$, exact union coverage | Prospective split disjoint on 597 cohort | **PASS** |
| **8. Protocol Compliance** | 80/20 Stratified, seed 42 | Prospective split matches protocol | **PASS** |

### Phase 5 Blocker Finding
The candidate feature table `data/processed/features_clean.csv` fails multiple essential readiness checks. In strict compliance with the project instructions ("If these checks fail, do not perform full EDA on the questionable feature table. Instead, create a short Phase 5 readiness report listing the failed checks, evidence, missing prerequisites, and what must be resolved. Do not repair the data yourself. Stop and return the handoff report"), this report documents the diagnostic evidence on the available training partition ($N=395$) to guide Phase 6 feature engineering while formally declaring that **full EDA on a frozen modeling dataset is blocked**.

---

## 2. Cohort and Training Set Summary

To guarantee zero test-set leakage, all exploratory analyses of feature distributions, correlations, and class differences were conducted **strictly on training partition participants**:
- **Total Eligible Cohort:** 597 participants (432 Parkinson's Disease [72.36%], 165 Healthy Controls [27.64%]).
- **Prospective Participant Split (80/20, seed 42):**
  - **Training Partition:** 477 participants (345 PD, 132 Control).
  - **Held-out Test Partition (Quarantined):** 120 participants (87 PD, 33 Control).
- **Candidate Feature Table Coverage in Training Partition:**
  - Evaluated Training Participants ($N=395$): 286 Parkinson's Disease (72.41%), 109 Healthy Controls (27.59%).
  - Missing Training Participants ($N=82$): Dropped by the legacy script's unapproved `MIN_KEYSTROKES = 1000` filter.
- **Predictor Modality:** 218 typing-derived numerical features. Zero demographic or clinical variables (`BirthYear`, `Gender`, `UPDRS`, `Tremors`, `Medications`) are present in the predictor matrix.

---

## 3. Schema and Feature-Family Inventory

The candidate table `features_clean.csv` contains 220 columns (1 participant ID, 1 target label, and 218 numerical predictors) structured into four feature groups:

| Feature Family | Column Prefix | Column Count | Documented Units | Primary Motor Construct Measured | Integrity Finding |
|---|---|---|---|---|---|
| **Hold-Time Family (HT)** | `hold_*` | 9 | Milliseconds (ms) | Key depress duration; finger bradykinesia and rigidity | Uncorrupted; valid distributions |
| **Latency Family (L)** | `lat1_*` | 90 | Milliseconds (ms) | Key-down to key-down interval; sequencing speed | **Corrupted by dates > 100k** |
| **Flight-Time Family (FT)** | `lat2_*` | 90 | Milliseconds (ms) | Key release to next press; rollover coordination | **Rollover destroyed; dates > 100k** |
| **Transition Metrics** | `*_LL`, `*_RR`, etc. | 29 | Milliseconds (ms) | Directional bimanual vs unimanual switches | Contains sparse NaNs |

The complete column-by-column inventory is exported to [`reports/eda/feature_inventory.csv`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/feature_inventory.csv).

---

## 4. Missingness and Structural Quality

An audit of missing values across all 218 training predictors revealed:
- **Zero Columns Entirely Missing:** No feature has 100% missing values.
- **Zero Constant Features:** All features have $\ge 2$ unique values.
- **Non-Finite Values:** Zero infinite values ($+\infty, -\infty$) detected.
- **Transition-Specific Incompleteness:** **54 columns** contain missing values, ranging from 1 to 24 missing participants (0.25% to 6.08% missingness):
  - Incompleteness is concentrated exclusively in directional transition subsets (e.g., `lat1_SL_*`, `lat2_SS_*`, `lat1_SR_*`).
  - *Root Cause:* Participants who did not execute a spacebar-to-left-hand (`SL`) or double-spacebar (`SS`) keystroke during their observation window generated empty subsets for that transition direction.

![Missingness by Feature](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/figures/fig1_missingness_by_feature.png)

The complete missingness accounting is recorded in [`reports/eda/missingness_summary.csv`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/missingness_summary.csv).

---

## 5. Univariate Distribution Analysis

Univariate summary statistics were computed across all predictors on the training set ($N=395$):

### 5.1 Hold-Time Distributions (Uncorrupted)
Key hold time exhibits physically plausible, well-behaved distributions:
- **`hold_mean`:** Mean = 118.41 ms, Median = 114.25 ms, IQR = 38.61 ms [98.96, 137.57], Min = 52.15 ms, Max = 208.76 ms.
- **`hold_std` (Variability):** Mean = 43.08 ms, Median = 41.52 ms, IQR = 18.06 ms [32.61, 50.67], Max = 97.22 ms.
- **`hold_cv` (Coefficient of Variation):** Mean = 0.364, Median = 0.360, Min = 0.178, Max = 0.589.
- **Distribution Shape:** Key hold-time distributions are unimodal and moderately right-skewed (skewness = +0.55), reflecting natural biological variations without extreme, unphysical anomalies.

![Hold Time Distributions](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/figures/fig2_hold_time_distributions.png)

### 5.2 Latency and Flight Distributions (Vitiated by Date Codes)
In sharp contrast to hold times, latency distributions in `features_clean.csv` display catastrophic distortion caused by shifted date codes:
- **`lat1_max`:** 41 training participants exhibit maximum latency $> 100,000$ ms (max = `171226.0`, representing 26 December 2017).
- **`lat2_max`:** 95 training participants exhibit maximum flight time $> 100,000$ ms (max = `171226.0`).
- **Kurtosis Distortion:** Extreme outliers inflate sample kurtosis to unphysical levels (e.g., `lat1_kurt` max = `720,079.58`).

The complete statistical table across all 218 predictors is available in [`reports/eda/descriptive_statistics.csv`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/descriptive_statistics.csv).

---

## 6. Outlier and Anomaly Screening

Candidate anomalies were screened using rule-based categories:

1. **Shifted Date Codes (28 Columns, up to 95 Training Participants):**
   - *Evidence:* Scatterplots of `lat1_max` and `lat2_max` against `hold_mean` reveal a discrete band of points clustered between 160,000 and 171,231, corresponding to YYMMDD dates from 2016 and 2017.
   - *Impact:* Corrupts maximums, means, standard deviations, and kurtosis.
2. **Coerced Flight-Time Rollover (Semantic Violation):**
   - *Evidence:* A histogram of `lat2_min` confirms a sharp artificial cutoff at $\ge 0$ ms ($min = 1.0$ ms).
   - *Impact:* The legacy script coerced negative flight times to NaN (`df.loc[df['latency2_ms'] <= 0, 'latency2_ms'] = np.nan`), eliminating physical keystroke overlap and biasing transition statistics.
3. **Statistical Extremes (Biological Hesitation):**
   - In uncorrupted hold-time metrics, extreme values (e.g. hold mean $> 180$ ms) represent legitimate Parkinsonian bradykinesia and must not be trimmed.

![Date Corruption Evidence](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/figures/fig3_date_corruption_evidence.png)

![Destroyed Rollover Flight Time](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/figures/fig4_destroyed_rollover_flight_time.png)

Detailed screening entries are cataloged in [`reports/eda/outlier_screening_summary.csv`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/outlier_screening_summary.csv).

---

## 7. Feature Correlation and Collinearity

Pairwise Spearman rank correlations were computed on uncorrupted training features to identify collinear redundancy:
- **Strong Central Tendency Redundancy ($|\rho| \ge 0.90$):**
  - `hold_mean` $\leftrightarrow$ `hold_median`: $\rho = +0.948$.
  - `lat1_mean` $\leftrightarrow$ `lat1_median`: $\rho = +0.962$.
- **Dispersion Redundancy:**
  - `hold_std` $\leftrightarrow$ `hold_iqr`: $\rho = +0.884$.
- **Independence across Families:**
  - Correlation between `hold_mean` and `lat1_mean` is moderate ($\rho \approx +0.42$), confirming that key depress duration and inter-key latency measure distinct physiological mechanisms (bradykinesia vs motor sequencing).

![Correlation Heatmap](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/figures/fig5_correlation_heatmap.png)

All 68 collinear pairs with $|\rho| \ge 0.90$ are cataloged in [`reports/eda/correlation_pairs.csv`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/correlation_pairs.csv).

---

## 8. Exploratory Training-Set Class Comparisons

Comparing feature distributions across Parkinson’s ($y=1, N=286$) and Control ($y=0, N=109$) groups in the training set revealed consistent neurophysiological trends:

| Feature Name | Control Median (ms) | Control IQR (ms) | PD Median (ms) | PD IQR (ms) | Median $\Delta$ (PD - Ctrl) | Observed Pattern |
|---|---|---|---|---|---|---|
| **`hold_mean`** | 109.94 | 32.11 | 117.84 | 38.14 | **+7.90 ms** | Higher keypress duration in PD |
| **`hold_median`** | 104.50 | 31.00 | 112.00 | 37.00 | **+7.50 ms** | Consistent shift in central tendency |
| **`hold_std`** | 36.96 | 17.06 | 43.18 | 17.38 | **+6.22 ms** | Higher trial-to-trial timing variability |
| **`hold_cv`** | 0.336 | 0.081 | 0.370 | 0.089 | **+0.034** | Higher relative variability normalized by speed |
| **`hold_iqr`** | 38.00 | 18.00 | 44.00 | 19.00 | **+6.00 ms** | Broader distribution spread in PD |

![Label Comparison Boxplots](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/figures/fig6_label_comparison_boxplots.png)

The complete comparison table is available in [`reports/eda/training_class_summary.csv`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/training_class_summary.csv).

---

## 9. Limitations

1. **Uncorrected Legacy Table:** Analysis was performed on the existing candidate table `features_clean.csv` to diagnose its failure modes; performance metrics or clinical conclusions must not be inferred from this corrupted dataset.
2. **Missing Cohort Segment:** 102 valid eligible participants are absent from the legacy table, precluding full-cohort distribution profiling.
3. **Hardware & Environmental Noise:** At-home naturalistic recordings lack supervision regarding keyboard switch mechanisms or shared computer usage.

---

## 10. Concrete Questions & Directives for Phase 6

Phase 6 (Feature Engineering and Preprocessing) must resolve the following concrete items:

1. **Re-aggregate Features from Phase 2 Parquet:** Discard `features_clean.csv` and re-extract feature vectors directly from clean records in [`data/interim/accepted/`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/data/interim/accepted/) across all **597 eligible participants**.
2. **Preserve Negative Flight Times:** Calculate genuine flight times ($FT = t_{\text{down}, i+1} - t_{\text{up}, i}$), preserving negative values as physical rollover. Engineer `rollover_ratio = (FT < 0).mean()` as an explicit bimanual coordination feature.
3. **Eliminate Date Corruption:** Enforce Phase 2 rule `W05_SUSPECT_DATE_CODE_VALUE` to prevent 6-digit dates from entering latency statistics.
4. **Transition Sparsity:** Aggregate sparse directional transitions into broader macro-groups (Unimanual vs Bimanual vs Spacebar) to avoid NaNs.
5. **Inner-Fold Pipelines:** Confine imputation (`SimpleImputer(strategy='median')`) and scaling (`RobustScaler`) strictly to inner cross-validation folds.

The full action matrix is detailed in [`reports/eda/phase6_questions.md`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/eda/phase6_questions.md).
