# Phase 7 — Baseline Model Evaluation Report

**Project:** Parkinson's Disease Prediction Using Typing Characteristics  
**Repository:** `PARKINSONS-DISEASE-PREDICTOR`  
**Working Branch:** `praneeth`  
**Dataset:** Tappy Extended Keystroke Dataset, Mendeley Data Version 3 (DOI: `10.17632/z39mhdsynx.3`)  
**Evaluation Methodology:** 5-Fold Stratified Cross-Validation on Training Cohort ($N=477$)  
**Held-Out Test Set:** Quarantined ($N=120$, 87 PD / 33 Control), untouched and unblinded  

---

## 1. Executive Summary

Phase 7 executes the prospective baseline machine-learning evaluation protocol formally established in Phase 3 (`docs/experimental_protocol.md`). Using the certified, clean Phase 6 feature table (`participant_features_raw.parquet` with 153 typing-derived predictors) and the authoritative Phase 4 training split (`data/processed/splits/train_ids.csv`), six candidate model architectures were evaluated across 5 stratified folds.

### Key Conclusions:
1. **The Deception of Standard Accuracy Exposed:**  
   The non-informative `Dummy Prior` baseline achieves an apparent **72.33% standard accuracy** simply by predicting the majority Parkinson's disease label for every participant. However, its **Balanced Accuracy is exactly 50.00%**, with **0.0% specificity** (identifying zero healthy controls). This confirms the Phase 3 protocol mandate that **Balanced Accuracy must serve as the primary model-selection criterion** under 72.4% / 27.6% class imbalance.
2. **Prospective Champion Identified:**  
   **Random Forest** achieved the highest 5-fold cross-validated Balanced Accuracy of **0.5938 ± 0.0350**, outperforming the Dummy chance floor by +9.38% and the linear baseline (Logistic Regression) by +4.06%. Random Forest also attained the highest discrimination metrics: **ROC-AUC = 0.6664 ± 0.0181** and **PR-AUC = 0.8373 ± 0.0192**.
3. **Model Selection Protocol Resolution:**  
   Because the performance margin between Random Forest ($0.5938$) and Logistic Regression ($0.5532$) is **+0.0406**, which well exceeds the protocol tie-breaking margin of $\pm 0.005$ ($\pm 0.5\%$), **Random Forest is prospectively selected as the champion baseline architecture** without requiring tie-break invocation.
4. **Leakage-Free Execution & Test-Set Isolation:**  
   All imputation (`SimpleImputer(strategy='median')`) and feature scaling (`StandardScaler`) transformers were fitted strictly within training folds inside scikit-learn `Pipeline` objects. The prospective holdout test set ($N=120$) was completely quarantined from disk read operations and model fitting.

---

## 2. Experimental Setup & Preprocessing Protocol

### 2.1 Cohort & Feature Matrix
- **Training Cohort:** Exactly 477 participants (345 Parkinson's Disease [72.33%], 132 Healthy Control [27.67%]).
- **Predictors ($P=153$):** 
  - Hold-time summary statistics across keypress subsets (mean, std, median, min, max, q25, q75, iqr, skew, kurtosis, coefficient of variation).
  - Latency and Flight-time distributions across consecutive keystroke pairs.
  - Hand transition frequencies (Left-Left, Left-Right, Right-Left, Right-Right).
- **Quality Checks:** Zero infinite values; all missing values handled inside cross-validation fold pipelines.

### 2.2 Pipeline Design & Leakage Prevention
```
Fold Training Split (N ≈ 381-382)
       │
       ▼
SimpleImputer(strategy='median')  <-- Fit on training fold only
       │
       ▼
StandardScaler() [if applicable] <-- Fit on training fold only
       │
       ▼
Classifier.fit(X_train_fold, y_train_fold)
       │
       ▼
Evaluate on Quarantined Fold Validation Set (N ≈ 95-96)
```

---

## 3. Comprehensive Baseline Cross-Validation Results

The table below summarizes the 5-fold cross-validated performance (mean ± standard deviation across 5 folds) for all six prospective baseline architectures, sorted in descending order of the primary metric (**Balanced Accuracy**).

| Candidate Architecture | Scaler Applied | Balanced Accuracy (Primary) | Sensitivity (PD Recall) | Specificity (Ctrl Recall) | Precision (PPV) | F1-Macro | Accuracy (Secondary) | ROC-AUC | Average Precision (PR-AUC) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | None | **0.5938 ± 0.0350** | 0.9304 ± 0.0361 | 0.2573 ± 0.0611 | 0.7661 ± 0.0155 | 0.5983 ± 0.0458 | 0.7443 ± 0.0309 | **0.6664 ± 0.0181** | **0.8373 ± 0.0192** |
| **Logistic Regression** | StandardScaler | **0.5532 ± 0.0421** | 0.7884 ± 0.0476 | 0.3179 ± 0.1128 | 0.7525 ± 0.0233 | 0.5512 ± 0.0449 | 0.6583 ± 0.0240 | 0.6161 ± 0.0415 | 0.8031 ± 0.0268 |
| **K-Nearest Neighbors** | StandardScaler | **0.5425 ± 0.0419** | 0.8580 ± 0.0215 | 0.2271 ± 0.0785 | 0.7440 ± 0.0213 | 0.5391 ± 0.0497 | 0.6835 ± 0.0292 | 0.5793 ± 0.0526 | 0.7639 ± 0.0353 |
| **Linear SVC** | StandardScaler | **0.5421 ± 0.0289** | 0.7275 ± 0.0919 | 0.3567 ± 0.1407 | 0.7495 ± 0.0234 | 0.5350 ± 0.0222 | 0.6248 ± 0.0315 | 0.5771 ± 0.0424 | 0.7853 ± 0.0278 |
| **RBF SVC** | StandardScaler | **0.5324 ± 0.0124** | 0.9739 ± 0.0189 | 0.0909 ± 0.0341 | 0.7369 ± 0.0075 | 0.4967 ± 0.0259 | 0.7296 ± 0.0124 | 0.6091 ± 0.0345 | 0.7773 ± 0.0212 |
| **Dummy Prior** | None | **0.5000 ± 0.0000** | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.7233 ± 0.0041 | 0.4197 ± 0.0014 | 0.7233 ± 0.0041 | 0.5000 ± 0.0000 | 0.7233 ± 0.0041 |

*Note: All metrics computed exclusively on the $N=477$ training cohort across 5 stratified folds. Input SHA256 hashes recorded in `reports/modelling/baseline_cv_manifest.json`.*

---

## 4. Architectural Analysis & Diagnostic Findings

### 4.1 Why Random Forest Wins
1. **Non-Linear Interactions:** Keystroke features possess non-linear, correlated dynamics (e.g., flight times and hold times interacting during dual-hand key sequences). Decision trees isolate sub-populations with motor latency without requiring monotonic linearity.
2. **Robustness to Extreme Outliers:** Even after domain clipping ($10\text{ s}$ hold, $100\text{ s}$ latency), motor pause distributions are heavily right-skewed. Tree partitioning is invariant to monotonic feature scaling and extreme tails.
3. **Superior Discrimination:** Random Forest achieves a mean ROC-AUC of **0.6664**, providing meaningful discriminative signal over chance ($0.5000$).

### 4.2 Why Linear SVC and Logistic Regression Struggle
- Linear models achieved modest sensitivity ($72.8\% - 78.8\%$) and slightly higher specificity ($31.8\% - 35.7\%$), but had higher variance across folds ($\text{std} = 0.11 - 0.14$ in specificity).
- With 153 features on 381 training instances per fold, standard $L_2$ regularization without feature selection produces weight shrinkage that struggles to isolate the subtle motor biomarkers amidst redundant keystroke statistics.

### 4.3 Why RBF SVC Collapses
- RBF SVC exhibits severe majority-class collapse: sensitivity is $97.4\%$ while specificity plunges to $9.1\%$.
- In a 153-dimensional standardized Euclidean space with class imbalance ($72.3\%$ positive), default kernel bandwidth ($\gamma = \frac{1}{P}$) causes the decision boundary to wrap tightly around the minority control instances, classifying nearly the entire space as positive.

---

## 5. Diagnostic Figures

The following diagnostic figures have been generated and saved under `reports/modelling/figures/`:

1. **`fig1_balanced_accuracy_comparison.png`:**  
   Bar chart comparing mean 5-fold Balanced Accuracy with $\pm 1 \text{ SD}$ error bars against the 0.50 non-informative chance floor.
2. **`fig2_sensitivity_vs_specificity.png`:**  
   Paired bar comparison illustrating the trade-off between Parkinson's recall (Sensitivity) and Healthy Control recall (Specificity) across models.
3. **`fig3_roc_auc_vs_balanced_acc.png`:**  
   Scatter plot demonstrating the correlation between threshold-independent discrimination (ROC-AUC) and threshold-fixed classification (Balanced Accuracy).
4. **`fig4_rf_aggregate_confusion_matrix.png`:**  
   Out-of-fold aggregate confusion matrix for the champion Random Forest model across all 477 training participants ($TP=321$, $FN=24$, $TN=34$, $FP=98$).

---

## 6. TRIPOD+AI Compliance & Next Phase Gate

- **TRIPOD+AI Section 10 (Model Development):** Documented full architecture specifications, parameter handling, and zero leakage.
- **TRIPOD+AI Section 12 (Validation Strategy):** Preserved complete holdout isolation. Test data ($N=120$) remains strictly unseen.
- **Phase 8 Readiness Gate: PASSED.**  
  Baseline performance establishes an empirical benchmark of **0.5938 Balanced Accuracy** and **0.6664 ROC-AUC**. Subsequent work is authorized to proceed to **Phase 8 (Model Comparison, Feature Selection, and Hyperparameter Tuning)**.
