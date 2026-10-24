# Prospective Experimental Protocol: Machine Learning for Parkinson’s Disease Prediction from Keystroke Dynamics

**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  
**Repository:** `CS-HARSHA/PARKINSONS-DISEASE-PREDICTOR`  
**Working Branch:** `praneeth`  
**Document Type:** Prospective Experimental & Methodological Protocol  
**Protocol Version:** 1.0 (Phase 3 Deliverable)  
**Date:** 24 October 2026
**Primary Dataset:** Tappy Extended Keystroke Dataset, Mendeley Data, Version 3 (DOI: `10.17632/z39mhdsynx.3`)
**Companion Review Document:** [`docs/literature_review.md`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/docs/literature_review.md)
**Status:** Frozen Prospective Protocol (Governing Phases 4–10)

---

## 1. Title, Date, Version, and Protocol Status

- **Protocol Title:** Prospective Machine-Learning Experimental Protocol for Parkinson’s Disease Classification Using Computer Keystroke Dynamics on the Tappy Extended Dataset.
- **Protocol Date:** 24 October 2026.
- **Protocol Version:** 1.0.
- **Protocol Status:** **Frozen Prospective Protocol** `[Verified]`.
  - All experimental rules, evaluation splits, candidate model architectures, primary selection metrics, and leakage safeguards defined in this document are established *a priori* based on Phase 2 empirical data audit evidence and literature review findings.
  - This protocol serves as a formal pre-registration contract. The primary model comparison rules and selection criteria specified herein must not be altered post-hoc after observing experimental results in subsequent phases. Any later exploratory deviations must be explicitly declared as post-hoc secondary analyses.

---

## 2. Study Objective and Research Scope

### 2.1 Primary Objective `[Proposed]`
The primary objective of this study is to determine whether supervised machine-learning models trained on statistical features derived exclusively from natural, unconstrained computer keyboard interactions can discriminate between individuals diagnosed with Parkinson’s disease and non-Parkinsonian control participants significantly better than non-informative baselines under a strict, participant-disjoint validation framework.

### 2.2 Core Methodological Tenets `[Verified]`
1. **Prediction Unit:** The unit of observation, prediction, and evaluation is strictly **one individual participant** ($N=1$ row per participant). Individual keystroke events or short typing sessions from the same individual must never be treated as independent observations.
2. **Predictor Modality:** The primary feature space is strictly restricted to **typing-derived characteristics** (timing statistics of hold time, inter-key latency, flight time, and key transitions).
3. **Strict Covariate Isolation:** Demographic variables (birth year, age, gender) and clinical metadata (MDS-UPDRS motor scores, presence of tremors, diagnosis year, levodopa/dopamine agonist medication status) are strictly quarantined from the primary feature matrix to avoid confounding and data leakage.
4. **Target Provenance:** The prediction target is the binary label extracted from verified participant metadata files (`Parkinsons: True` $\to 1$, `Parkinsons: False` $\to 0$).
5. **Regulatory & Research Disclaimer:** This project develops an academic research prototype intended to evaluate the informational capacity of keystroke timing signals. It is not an FDA-cleared, CE-marked, or clinically validated medical diagnostic device, and it must never be interpreted as providing clinical medical diagnosis or guiding therapeutic decisions.

---

## 3. Dataset, Version, and Target-Label Provenance

### 3.1 Primary Dataset Identity `[Verified]`
The sole primary data source for this study is:
- **Title:** Keystroke dataset for 'The detection of movement-related disease by changes in keystroke characteristics while typing'
- **Author:** Warwick Adams (2023)
- **Repository:** Mendeley Data, Version 3
- **Persistent Identifier (DOI):** `10.17632/z39mhdsynx.3`
- **Official Dataset URL:** https://data.mendeley.com/datasets/z39mhdsynx/3
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)

This dataset must be designated unambiguously as **Tappy Extended Dataset (Mendeley Data V3)** throughout all subsequent project phases. It must not be confused with, nor substituted by, the historical 2016–2017 PhysioNet Tappy release, which contained an earlier and smaller cohort.

### 3.2 Target Label Provenance `[Verified]`
Participant diagnosis labels originate from 646 individual text files located in `data/raw/participant_metadata/User_KEY.txt`. In Phase 2, these files were parsed and validated into [`data/interim/participant_labels.csv`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/data/interim/participant_labels.csv):
- `Parkinsons: True` (case-insensitive) $\to$ Class `1` (Parkinson's Disease / PD).
- `Parkinsons: False` (case-insensitive) $\to$ Class `0` (Control / Healthy Comparison).
- Missing, blank, or ambiguous entries $\to$ `MISSING_LABEL` / `INVALID_LABEL` (excluded from eligible cohort).

---

## 4. Verified Phase 2 Cohort Baseline and Known Limitations

### 4.1 Empirical Cohort Verification `[Verified]`
Phase 2 completed an exhaustive, byte-level scan and reconciliation of all raw files, recorded in [`reports/data_quality/summary.json`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/data_quality/summary.json) and [`reports/data_quality/cohort_reconciliation.csv`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/data_quality/cohort_reconciliation.csv), establishing the definitive participant hierarchy:

```text
Total Unique Participant Keys: 791
├── In Raw Keystroke Logs: 754
│   ├── In Metadata Files: 609
│   │   ├── Valid Labels (True/False): 606 (438 True, 168 False)
│   │   │   ├── At least 1 usable accepted record: 597 (ELIGIBLE COHORT: 432 True, 165 False)
│   │   │   └── Zero usable records (empty / quarantined): 9 (EXCLUDED_NO_USABLE_RECORDS: 6 True, 3 False)
│   │   └── Missing Label: 3 (FXD2HCNIPJ, PVADT62JUM, QIHPY6RIK3) (EXCLUDED_NO_VALID_LABEL)
│   └── Logs Only (Missing Metadata): 145 (EXCLUDED_NO_METADATA)
└── Metadata Only (No Keystroke Logs): 37 (EXCLUDED_NO_LOGS: 17 True, 20 False)
```

### 4.2 Cohort Discrepancy Accounting `[Verified]`
- **The 609 to 606 Labeled Gap:** Exactly 3 participants present in both logs and metadata have empty metadata fields where `Parkinsons:` is blank (`FXD2HCNIPJ`, `PVADT62JUM`, `QIHPY6RIK3`).
- **The 606 to 597 Usable Gap:** Exactly 9 labeled participants have zero usable records in `data/interim/accepted/`:
  - 2 participants have 0-byte log files: `DRQKANH3Q0` (`DRQKANH3Q0_1709.txt`), `JORVLMISQA` (`JORVLMISQA_1709.txt`).
  - 7 participants contributed monthly logs containing 100% comma-decimal formatted numbers (e.g., `0066,4` from European Windows locales) which failed plain-decimal validation and were quarantined: `EIFBUG1US3`, `H9PJHE5OTR`, `HQW5YE0PUK`, `KM4FAMJWJ3`, `OPACZPPV1Q`, `PDKF0GKH59`, `TOC5D4KGNO`.
- **The Legacy 495-Row Table (`features_clean.csv`):** An earlier exploratory table in the repository contained 495 rows. This legacy artifact omitted 102 validly labeled, usable participants without documentation. As mandated by Phase 2 rules, this 495-row table must not be trusted or utilized for modeling.

---

## 5. Research Questions and Hypotheses

### 5.1 RQ1 — Predictive Signal under Strict Participant Disjointness `[Proposed]`
- **Research Question:** Can supervised machine-learning models trained on participant-level keystroke timing statistics predict binary Parkinson’s disease labels with discriminatory performance superior to a non-informative majority-class baseline under participant-disjoint evaluation?
- **Hypothesis 1 ($H_1$):** At least one candidate model will achieve a cross-validated balanced accuracy on training participants that is statistically superior to the prior/dummy baseline ($> 50.0\%$).
- **Scientific Rationale:** Basal ganglia dopamine depletion causes physical bradykinesia and rigidity, systematically lengthening key hold time and increasing transition variance.

### 5.2 RQ2 — Algorithmic Comparison `[Proposed]`
- **Research Question:** How do classical linear models (Logistic Regression, Linear SVM), kernel machines (RBF SVM), instance-based classifiers (KNN), and tree ensembles (Random Forest) compare in classification performance, calibration, and stability on keystroke feature representations?
- **Hypothesis 2 ($H_2$):** Non-linear models (Random Forest, RBF SVM) will achieve higher balanced accuracy than linear classifiers due to non-linear interactions between hold-time variance and transition latencies.

### 5.3 RQ3 — Relative Contribution of Feature Families `[Proposed]`
- **Research Question:** What is the relative predictive capability of hold-time features versus inter-key latency and flight-time features, and does their combination yield synergistic predictive performance?
- **Hypothesis 3 ($H_3$):** Combined feature sets ($HT + L + FT$) will demonstrate superior discriminatory power compared to hold-time features alone or latency/flight features alone.

### 5.4 RQ4 — Robustness to Preprocessing and Imbalance Adjustments `[Proposed]`
- **Research Question:** How sensitive are model rankings, discrimination metrics, and decision thresholds to feature standardization strategies (standard scaling vs. robust scaling) and class-imbalance mitigations (class weighting)?

---

## 6. Inclusion and Exclusion Criteria

To prevent selection bias and maintain complete transparency, participant inclusion in Phase 4 must follow deterministic criteria recorded in [`data/interim/participant_inventory.csv`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/data/interim/participant_inventory.csv):

### 6.1 Inclusion Criteria `[Proposed]`
A participant is included in the primary modeling cohort if and only if **all** of the following conditions are satisfied:
1. `in_logs == True`: At least one monthly keystroke log file (`KEY_YYMM.txt`) exists in `data/raw/keystroke_logs/`.
2. `in_metadata == True`: A corresponding metadata file (`User_KEY.txt`) exists in `data/raw/participant_metadata/`.
3. `label_status == 'VALID_LABEL'`: The metadata file contains an unambiguous boolean target label (`Parkinsons: True` or `Parkinsons: False`).
4. `(n_accepted + n_warned) > 0`: The participant contributes at least one valid, accepted keystroke record in `data/interim/accepted/KEY_YYMM.parquet`.

Participants meeting all four criteria are assigned primary status `cohort_status = 'ELIGIBLE'`.

### 6.2 Exclusion Criteria `[Proposed]`
Participants are excluded from the primary predictive model if any of the following apply:
1. **Missing Metadata (`EXCLUDED_NO_METADATA`):** Present in keystroke logs but lacking a metadata file ($N=145$).
2. **Missing Logs (`EXCLUDED_NO_LOGS`):** Present in metadata but lacking any log files ($N=37$).
3. **Ambiguous Label (`EXCLUDED_NO_VALID_LABEL`):** Metadata present but `Parkinsons:` value is missing or corrupted ($N=3$).
4. **No Usable Typing Records (`EXCLUDED_NO_USABLE_RECORDS`):** Present in both sources with a valid label, but all records are 0 bytes or quarantined ($N=9$).
5. **Secondary Minimum-Keystroke Threshold (`[Deferred to Phase 4]`):** Any potential threshold requiring a minimum typing volume (e.g., $\ge 500$ or $\ge 1,000$ keystrokes to ensure statistical stability of higher-order moments) must be formally evaluated in Phase 4 before freezing the modeling dataset.

---

## 7. Participant-Level Data Flow

```text
Raw Keystroke Logs (2,165 files, 40.5M lines) ──► Phase 2 Parser & Cleaner
                                                           │
                                                           ▼
                             data/interim/accepted/KEY_YYMM.parquet (39.7M clean + 314k warned)
                                                           │
                                                           ▼ [Phase 4 / 6]
                             Feature Aggregation: Group by user_key
                             (Calculate Mean, Median, Std, IQR, Skew, Kurt for HT, L, FT)
                                                           │
                                                           ▼ [Phase 4]
                             Join with data/interim/participant_labels.csv
                                                           │
                                                           ▼
                             Master Modelling Table (1 row per eligible participant, N=597)
                                                           │
                               ┌───────────────────────────┴───────────────────────────┐
                               ▼                                                       ▼
                 Stratified Training Split (80%, N≈478)                   Held-out Test Split (20%, N≈119)
                 - Stratified 5-Fold Cross-Validation                     - FROZEN UNTIL PHASE 10
                 - Inner-Fold Imputation & Scaling                        - Final Independent Evaluation
                 - Model Comparison & Hyperparameter Tuning
```

---

## 8. Predictor Definitions and Leakage Exclusions

### 8.1 Primary Typing Predictor Families `[Proposed]`
Feature engineering in Phase 6 will generate aggregate predictors organized into three distinct families:
1. **Hold-Time Family ($HT$):** Mean, standard deviation, median, interquartile range ($IQR$), coefficient of variation ($CV$), skewness, and kurtosis of key hold duration across all valid keystrokes. Reflects localized digital bradykinesia and finger release hesitation.
2. **Latency Family ($L$):** Mean, standard deviation, median, $IQR$, $CV$, skewness, and kurtosis of key-down to key-down intervals. Reflects inter-key transition delays and motor hesitation.
3. **Flight-Time Family ($FT$):** Mean, standard deviation, median, $IQR$, negative flight fraction (rollover percentage), and skewness of key-up to key-down intervals. Reflects motor coordination, bimanual transitions, and chording dynamics.
4. **Transition Directional Metrics:** Directional breakdown of timing across unimanual (`LL`, `RR`) versus bimanual (`LR`, `RL`, `LS`, `SL`, `RS`, `SR`, `SS`) transitions.

### 8.2 Strict Exclusion Allowlist `[Verified]`
The following columns must be strictly excluded from the feature matrix $X$:
- `user_key` / participant identifiers (prevents author memorization).
- Target labels (`Parkinsons`, `parkinsons_raw`, `label`).
- `source_file`, `first_yymm`, `last_yymm` (prevents temporal or batch artifacts).
- Clinical metadata: `BirthYear`, `Gender`, `Tremors`, `DiagnosisYear`, `Sided`, `UPDRS`, `Impact`, `Levodopa`, `DA`, `MAOB`, `Other`.

---

## 9. Train/Test Split and Cross-Validation Strategy

### 9.1 Outer Holdout Test Partition `[Proposed]`
- **Splitting Strategy:** Stratified, participant-disjoint random split.
- **Split Ratio:** **80% Training Set, 20% Held-Out Test Set**.
- **Stratification Variable:** Target label (`label`: PD vs. Control).
- **Random Seed:** Strictly fixed at `random_state = 42`.
- **Integrity Rule:** The 20% holdout test set is immediately quarantined upon generation in Phase 4. It must never be consulted during feature selection, model exploration, hyperparameter tuning, or threshold selection.

### 9.2 Inner Validation Scheme `[Proposed]`
- **Cross-Validation:** **5-Fold Stratified Cross-Validation** repeated exclusively within the 80% training set.
- **Fold Partitions:** Each training fold must maintain the training set's empirical class ratio (~72% PD, ~28% Control).
- **Pipeline Wrapping:** Every transformation (imputer, scaler, feature selector) must be encapsulated in a `scikit-learn` `Pipeline` and fitted strictly on the $k-1$ training folds, then transformed on the $k$-th validation fold.

---

## 10. Candidate Machine Learning Models

Model selection is prospective. Exploration in Phase 7 is restricted to six well-defined candidate model architectures:

1. **Baseline — DummyClassifier `[Proposed]`:**
   - Strategy: `strategy='prior'` (predicts class probability based on empirical training class prevalence).
   - Purpose: Establishes the non-informative performance floor. Any model failing to outperform this baseline lacks predictive signal.
2. **Logistic Regression `[Proposed]`:**
   - Specification: `LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)`.
   - Purpose: Transparent, linear baseline exposing log-odds coefficients for each feature family.
3. **Linear Support Vector Classifier `[Proposed]`:**
   - Specification: `LinearSVC(dual=False, max_iter=2000)`.
   - Purpose: Assesses maximum-margin linear separability in standardized feature space.
4. **Support Vector Machine with RBF Kernel `[Proposed]`:**
   - Specification: `SVC(kernel='rbf', probability=True)`.
   - Purpose: Evaluates non-linear decision boundaries and complex feature interactions.
5. **K-Nearest Neighbors (KNN) `[Proposed]`:**
   - Specification: `KNeighborsClassifier(metric='euclidean')`.
   - Purpose: Non-parametric instance-based clustering benchmark.
6. **Random Forest Classifier `[Proposed]`:**
   - Specification: `RandomForestClassifier(n_estimators=200, random_state=42)`.
   - Purpose: Non-linear tree ensemble robust to feature scale, non-normality, and outliers, providing Gini and permutation feature importances.

*Exclusion of Complex Architectures `[Verified]`:* Deep neural networks, gradient-boosted ensembles with massive hyperparameter grids, automated machine learning (AutoML), and multi-layer stacking are explicitly prohibited in the baseline phase to prevent small-sample overfitting and maintain interpretability.

---

## 11. Preprocessing Principles Reserved for Phase 6

Phase 3 establishes preprocessing rules while strictly deferring implementation to Phase 6:
1. **Missing Value Handling `[Proposed]`:** Missing feature values must be imputed using median values computed solely within the training folds (`SimpleImputer(strategy='median')`). No participant row may be deleted during cross-validation due to missing features.
2. **Feature Scaling `[Proposed]`:** Distance-based and gradient-based models (LogReg, SVM, KNN) require feature standardization. A `StandardScaler` or `RobustScaler` must be fitted within each fold pipeline. Tree models (Random Forest) will be evaluated on unscaled and scaled features to verify invariance.
3. **Outlier Mitigation `[Proposed]`:** Unusually large values (e.g., extreme hold times $> 2,000$ ms) are legitimate behavioral anomalies (freezing of movement, distractions) and must not be deleted. If clipping or winsorization is deemed necessary in Phase 6, thresholds must be fitted strictly inside training folds.
4. **No Synthetic Resampling on Test Data `[Proposed]`:** If SMOTE or random oversampling is explored in secondary analyses, it must be applied strictly to inner training folds (`imblearn.pipeline.Pipeline`). The test set and validation folds must reflect the natural empirical class distribution.

---

## 12. Primary and Secondary Evaluation Metrics

### 12.1 Primary Model-Selection Metric `[Proposed]`
- **Primary Metric:** **Balanced Accuracy**
  $$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2} = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)$$
- **Justification:** Because the eligible cohort has an unbalanced class distribution (432 PD vs. 165 Control, ~72.4% PD), standard accuracy is deceptive; a naive model predicting 100% PD achieves ~72.4% accuracy with only 50.0% balanced accuracy. Balanced accuracy weights sensitivity and specificity equally, penalizing majority-class bias.

### 12.2 Comprehensive Secondary Metrics `[Proposed]`
Every model evaluation must report:
1. **Sensitivity (Recall / True Positive Rate):** $\frac{TP}{TP + FN}$ (critical for assessing screening efficacy).
2. **Specificity (True Negative Rate):** $\frac{TN}{TN + FP}$ (critical for minimizing false alarms in healthy controls).
3. **Precision (Positive Predictive Value):** $\frac{TP}{TP + FP}$.
4. **F1-Score (Macro and Weighted):** Harmonic mean of precision and recall.
5. **ROC-AUC:** Area under the Receiver Operating Characteristic curve (evaluates threshold-independent ranking ability).
6. **Average Precision (PR-AUC):** Area under the Precision-Recall curve (highly informative under class imbalance).
7. **Overall Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$ (reported for completeness, but never used as the selection criterion).
8. **Confusion Matrix:** Full tabular breakdown of $TP$, $FP$, $TN$, $FN$.

---

## 13. Model Selection and Decision Threshold Rules

### 13.1 Prospective Model-Selection Rule `[Proposed]`
1. **Evaluation Phase:** Candidate models will be ranked based on their **mean 5-fold cross-validated Balanced Accuracy** on the training partition.
2. **Tie-Breaking Rule:** If two models achieve mean balanced accuracy within $\pm 0.005$ (0.5%), the model with the higher cross-validated **ROC-AUC** will be selected. If still tied, the simpler model (fewer parameters: Logistic Regression over SVM; Linear SVM over RBF SVM; SVM over Random Forest) will be selected following Occam's razor.
3. **Decision Threshold Freezing:** By default, the classification decision threshold is fixed at $P \ge 0.50$. If threshold tuning is justified in secondary experiments, the optimal threshold must be determined via Youden's $J$ index ($J = \text{Sensitivity} + \text{Specificity} - 1$) solely within the training cross-validation folds and frozen prior to holdout test set evaluation.

---

## 14. Feature-Family and Robustness Analyses

To answer RQ3 and RQ4, the modeling workflow will execute structured comparative experiments:
1. **Ablation by Feature Family `[Proposed]`:**
   - Experiment A: Models trained on Hold-Time ($HT$) features only ($N_{\text{feat}} \approx 7$).
   - Experiment B: Models trained on Latency/Flight ($L + FT$) features only ($N_{\text{feat}} \approx 14$).
   - Experiment C: Models trained on Full Combined Typing features ($N_{\text{feat}} \approx 21$).
2. **Class Imbalance Sensitivity `[Proposed]`:**
   - Compare unweighted classifiers against cost-sensitive classifiers (`class_weight='balanced'`).
3. **Interpretability & Feature Importance `[Proposed]`:**
   - Permutation feature importance on the frozen model to identify the top 5 most influential predictors.
   - Verify whether hold-time variance or inter-key latency dominates classification.
   - Confirm that model decisions are not driven by a single suspicious or degenerate feature.

---

## 15. Sample-Size Limitations and Statistical Uncertainty

1. **Finite Cohort Constraints `[Verified]`:**
   - With an eligible cohort of 597 participants (432 PD, 165 Control), an 80/20 train/test split allocates approximately 478 participants to training (~346 PD, ~132 Control) and 119 participants to the holdout test set (~86 PD, ~33 Control).
   - In a test set of 119 participants with 33 controls, each individual misclassification in the control group alters specificity by approximately $\frac{1}{33} \approx 3.03\%$.
2. **Confidence Intervals `[Proposed]`:**
   - Final holdout metrics must be reported with 95% bootstrap confidence intervals (1,000 bootstrap resamples of the test set) to convey statistical uncertainty honestly.
3. **No Over-Interpretation `[Verified]`:**
   - Claims of "near-perfect detection" or "clinical diagnosis" are strictly unscientific and prohibited.

---

## 16. Reproducibility, Manifest Tracking, and Experiment Logging

To guarantee complete reproducibility:
1. **Deterministic Execution `[Proposed]`:** Fixed random seeds (`random_state = 42`) across all splits, cross-validation folds, and stochastic estimators.
2. **Artifact Manifest `[Proposed]`:** Every generated feature table, split manifest, and model artifact must record:
   - Git commit SHA.
   - Python environment and package versions (`scikit-learn`, `pandas`, `numpy`, `pyarrow`).
   - SHA-256 hashes of input interim datasets.
3. **Scripted Pipeline Execution `[Proposed]`:** All model comparisons must be runnable via standalone CLI modules (`src/modelling/`) and Jupyter notebooks.

---

## 17. Reporting Requirements (TRIPOD+AI Alignment)

Reporting in Phase 10 and Phase 12 will adhere to the **TRIPOD+AI Statement** (Collins et al., 2024) for clinical prediction models:
- **Title and Abstract:** Explicitly state the prediction problem, study population, study design, and target label.
- **Source of Data:** Clearly identify the Tappy Extended dataset, Mendeley Data V3.
- **Participants:** State full eligibility, inclusion, and exclusion criteria with a flow diagram.
- **Outcome:** Detail the provenance and verification of the Parkinson's disease target label.
- **Predictors:** Provide a comprehensive data dictionary defining every feature, its units, and calculation logic.
- **Sample Size:** Report total participants, class distribution, and justification.
- **Missing Data:** Detail the handling and imputation of missing values within folds.
- **Model Specification:** Document all candidate model hyperparameters and selection rules.
- **Performance Evaluation:** Report discrimination (Balanced Accuracy, ROC-AUC, PR-AUC), calibration where applicable, confusion matrices, and sensitivity/specificity with confidence intervals.
- **Limitations:** Discuss hardware heterogeneity, uncontrolled home recording, absence of clinical neurologist re-examination, and lack of external cohort validation.

---

## 18. Ethical, Bias, and Interpretation Limitations

1. **Spectrum Bias `[Verified]`:** At-home volunteer typing cohorts may under-represent severe, advanced-stage Parkinson's patients who are physically unable to operate standard computer keyboards, and conversely under-represent healthy individuals with arthritis or other non-Parkinsonian motor tremors.
2. **Demographic Confounding `[Verified]`:** If the Parkinson's cohort differs systematically in age or gender from the control cohort, typing slowing could reflect normal aging rather than pathology. Secondary covariate analyses in Phase 9 must investigate whether typing features correlate strongly with age.
3. **No Clinical Substitution `[Verified]`:** Keystroke analytics is a passive digital monitoring biomarker, not a substitute for comprehensive neurological examination, neuroimaging, or dopamine transporter (DaT) SPECT imaging.

---

## 19. Decisions Deferred to Subsequent Phases & Protocol Decision Table

### 19.1 Decisions Deferred to Phase 4, Phase 5, and Phase 6 `[Verified]`
To maintain strict phase boundary discipline, the following decisions and implementations are explicitly deferred:

| Deferred Decision / Task | Deferred To | Rationale |
|---|---|---|
| **Authoritative Modelling Table Assembly** | Phase 4 | Requires executing participant-level joins and reconciling the legacy 495-row table against the 597 eligible cohort. |
| **Minimum Typing Volume Threshold** | Phase 4 | Requires empirical inspection of keystroke counts per participant to establish whether participants with $< 500$ keystrokes produce degenerate higher-order moments. |
| **Exploratory Feature Distribution Analysis (EDA)** | Phase 5 | Prohibited in Phase 3; requires inspecting feature correlations, skewness, and outliers on the assembled modeling dataset. |
| **Mathematical Formulation of Feature Formulas** | Phase 6 | Formal coding of feature extractors (skewness, kurtosis, coefficients of variation) and preprocessing pipelines. |
| **Model Training & Cross-Validation Execution** | Phase 7 | Actual fitting of scikit-learn models and hyperparameter tuning. |
| **Model Selection & Final Threshold Optimization** | Phase 8 | Execution of model selection criteria defined in this protocol. |
| **Holdout Test Set Evaluation** | Phase 10 | The quarantined 20% test set remains untouched until Phase 10. |

### 19.2 Explicit Protocol Decision Summary Table
Every key decision in this protocol is classified as **`[Verified]`** (empirically confirmed fact), **`[Proposed]`** (predeclared future rule), or **`[Unresolved]`** (open question to resolve in subsequent phases):

| Decision Topic | Status | Proposed Value / Rule | Methodological Rationale | Conditions for Reconsideration |
|---|---|---|---|---|
| **Primary Dataset** | `[Verified]` | Tappy Extended, Mendeley Data V3 (DOI: `10.17632/z39mhdsynx.3`) | Verified by Phase 2 profiling of all 2,165 logs and 40.5M records. | Never. Canonical primary dataset. |
| **Prediction Unit** | `[Verified]` | Exactly one observation per participant | Prevents record-level data leakage and author fingerprinting. | Never. Fundamental leakage rule. |
| **Primary Predictor Set** | `[Verified]` | Keystroke dynamics features only ($HT, L, FT$) | Evaluates motor typing signal without clinical confounding. | Never for primary experiment. |
| **Clinical Covariates** | `[Verified]` | Quarantined from primary feature matrix | Prevents target and metadata leakage. | Only in separate secondary analysis. |
| **Target Label Definition** | `[Verified]` | Binary: `Parkinsons: True` $\to 1$, `False` $\to 0$ | Verified against Phase 2 metadata extraction. | Never. Documented dataset label. |
| **Eligible Cohort Base** | `[Verified]` | 597 eligible participants (432 PD, 165 Control) | Reconciled in Phase 2 `cohort_reconciliation.csv`. | If Phase 4 imposes a minimum typing count. |
| **Train/Test Partitioning** | `[Proposed]` | Stratified 80% Training / 20% Held-Out Test | Balances statistical power in training with test evaluation size. | If class counts become too sparse in test set. |
| **Validation Scheme** | `[Proposed]` | 5-Fold Stratified Cross-Validation on Training | Standard unbiased validation design within training set. | If inner fold test counts fall below 10 controls. |
| **Random State** | `[Proposed]` | Fixed seed: `random_state = 42` | Ensures deterministic, reproducible partition assignment. | Never. Essential for reproducibility. |
| **Primary Metric** | `[Proposed]` | Balanced Accuracy | Penalizes majority-class bias on imbalanced classes. | Must be formally amended prior to Phase 7. |
| **Candidate Models** | `[Proposed]` | Dummy, LogReg, Linear SVM, RBF SVM, KNN, Random Forest | Diverse linear, non-linear, and ensemble models with established inductive bias. | If baseline models fail to train cleanly. |
| **Minimum Keystroke Count** | `[Unresolved]` | Evaluated in Phase 4 (provisional candidate: $\ge 500$ keystrokes) | Ensures statistical validity of skewness and kurtosis estimates. | Decided and logged in Phase 4 before splitting. |
| **Feature Transformation Choice** | `[Unresolved]` | RobustScaler vs. StandardScaler | Resolved in Phase 6 based on feature skewness and outlier frequency. | Selected inside cross-validation in Phase 6. |

---

## 20. References

1. **Adams, W. (2017).** High-accuracy detection of early Parkinson's Disease using multiple characteristics of finger movement while typing. *PLOS ONE*, 12(11), e0188226. https://doi.org/10.1371/journal.pone.0188226
2. **Adams, W. (2023).** Keystroke dataset for 'The detection of movement-related disease by changes in keystroke characteristics while typing'. *Mendeley Data*, Version 3. https://doi.org/10.17632/z39mhdsynx.3
3. **Arroyo-Gallego, T., Ledesma-Carbayo, M. J., Sanchez-Ferro, A., et al. (2018).** Detection of motor impairment in early Parkinson's disease via natural typing interaction with keyboards: validation of the neuroQWERTY approach in an uncontrolled at-home setting. *Journal of Medical Internet Research*, 20(3), e89. https://doi.org/10.2196/jmir.9462
4. **Collins, G. S., Moons, K. G. M., Dhiman, P., et al. (2024).** TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *The BMJ*, 385, e078378. https://doi.org/10.1136/bmj-2023-078378
5. **Giancardo, L., Sánchez-Ferro, Á., Arroyo-Gallego, T., et al. (2016).** Computer keyboard interaction as an indicator of early Parkinson’s disease. *Scientific Reports*, 6, 34468. https://doi.org/10.1038/srep34468
6. **Goetz, C. G., Tilley, B. C., Shaftman, S. R., et al. (2008).** Movement Disorder Society-sponsored revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS): Process, format, and clinimetric testing plan. *Movement Disorders*, 23(15), 2129–2170. https://doi.org/10.1002/mds.22340
7. **Iakovakis, D., Hadjidimitriou, S., Charisis, V., et al. (2018).** Touchscreen typing-pattern analysis for detecting fine motor skills decline in early-stage Parkinson's disease. *Scientific Reports*, 8, 7663. https://doi.org/10.1038/s41598-018-25999-0
8. **Iakovakis, D., Hadjidimitriou, S., Charisis, V., et al. (2022).** Diagnostic accuracy of keystroke dynamics as digital biomarkers for fine motor decline in neuropsychiatric disorders: a systematic review and meta-analysis. *Scientific Reports*, 12, 8094. https://doi.org/10.1038/s41598-022-11865-7
9. **Tripathi, S., Arroyo-Gallego, T., & Giancardo, L. (2023).** Keystroke-dynamics for Parkinson's disease signs detection in an at-home uncontrolled population: A new benchmark and method. *IEEE Transactions on Biomedical Engineering*, 70(1), 182–192. https://doi.org/10.1109/TBME.2022.3187309
10. **Tripathi, S., Acien, A., Holmes, A. A., Arroyo-Gallego, T., & Giancardo, L. (2024).** Generalizing Parkinson’s disease detection using keystroke dynamics: a self-supervised approach. *Journal of the American Medical Informatics Association*, 31(6), 1239–1246. https://doi.org/10.1093/jamia/ocae050
