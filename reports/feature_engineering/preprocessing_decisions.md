# Preprocessing Pipeline Design & Component Decisions

**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  
**Phase:** Phase 6 — Feature Engineering and Preprocessing  
**Date:** 24 October 2026  

---

## 1. Summary of Implemented Preprocessing Transformers

In accordance with Section 9 and Section 10 of the Phase 6 specification, preprocessing components are modularized into reusable scikit-learn transformers via `src.feature_engineering.preprocessing.build_preprocessing_pipeline`.

| Component | Default Strategy | Supported Options | Downstream Model Alignment | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Missing Value Imputer** | `SimpleImputer(strategy='median')` | `'median'`, `'mean'`, optional `add_indicator=True` | All models | Robust to right-skewed timing distributions; preserves feature medians without distributional shift. |
| **Standard Scaler** | `StandardScaler()` | `StandardScaler()`, `RobustScaler()`, `None` | Logistic Regression, SVM, KNN | Normalizes feature scales (mu=0, sigma=1) required for distance and gradient-based estimators. |
| **Robust Scaler** | Configurable alternative | `RobustScaler()` | Linear models under heavy tails | Centers by median and scales by IQR; immune to extreme typing outliers. |
| **Passthrough Scaler** | `'none'` | No scaling | Random Forest, ExtraTrees, XGBoost | Tree-based models are invariant to monotonic scaling; unscaled features preserve clinical interpretability. |
| **Power Transformer** | `PowerTransformer(method='yeo-johnson')` | Optional flag | Gaussian-assuming linear models | Symmetrizes moderately skewed timing distributions; supports negative values (flight times). |

---

## 2. Train-Fold-Safety & Leakage Prevention

1. **Zero Global Fitting:** Preprocessing transformers in Phase 6 remain strictly **unfitted**.
2. **CV Integration:** In Phase 7, the pipeline will be combined with estimators inside `sklearn.pipeline.Pipeline` or `cross_validate`, ensuring that statistics (imputation medians, scaling means/standard deviations) are estimated exclusively on training folds.
3. **Holdout Test Isolation:** Test partition participants ($N=120$) will never contribute to learned transformer parameters.
