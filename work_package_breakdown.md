# Parkinson's Keystroke ML Pipeline: Work Package Breakdown

This document outlines the 10 Work Packages (WPs) required to build the Parkinson's Keystroke Screening project, along with their respective goals and assigned files.

## WP1: Data and cleaning
**Goal:** Handle the ingestion of the raw PhysioNet dataset, map subject IDs to clinical labels, apply cleaning rules, and output standardized session data.
*   **Source files:** `src/config.py`, `src/data_loading.py`, `src/cleaning.py`
*   **Scripts:** `scripts/01_prepare_data.py`
*   **Tests:** `tests/test_cleaning.py`
*   **Docs:** `DATA_DICTIONARY.md`

## WP2: EDA and feature tables
**Goal:** Perform Exploratory Data Analysis, generate hold-time distributions and summary statistics, and build the core feature tables for downstream models.
*   **Source files:** `src/eda.py`, `src/feature_extraction.py`
*   **Scripts & Notebooks:** `scripts/02_eda.py`, `scripts/03_extract_features.py`, `notebooks/01_explore_data.ipynb`
*   **Tests:** `tests/test_feature_extraction.py`
*   **Docs:** `reports/eda.md`, `docs_GLOSSARY.md`, `TEAM_WORKPLAN.md` (shared with WP8)

## WP3: MACD, corridor sensitivity, feature variants
**Goal:** Write and prove the mathematical logic for the Mean Absolute Consecutive Change (MACD) feature, and test its robustness against different quantile corridors and transformations.
*   **Source files:** `src/features.py`
*   **Scripts:** `scripts/05_corridor_sensitivity.py`, `scripts/06_feature_variants.py`
*   **Tests:** `tests/test_features.py`

## WP4: Simple baselines and nQi window features
**Goal:** Establish basic single-feature benchmarks (mean hold, standard deviation) and extract specific time-window features required for the nQi-style baseline.
*   **Source files:** `src/baselines_simple.py`
*   **Scripts:** `scripts/07_baselines_simple.py`
*   **Tests:** `tests/test_baselines.py`

## WP5: nQi-style SVR ensemble
**Goal:** Build a complex baseline model mimicking a previous study by training an ensemble of 200 Support Vector Regression (SVR) models on bootstrap samples of typing windows.
*   **Source files:** `src/baselines_nqi.py`
*   **Scripts:** `scripts/08_baselines_nqi.py`
*   **Tests:** `tests/test_baselines_nqi.py`

## WP6: FRESH part A (extraction and selection)
**Goal:** Introduce automated feature engineering using `tsfresh` to extract time-series features and rank them by relevance, with strict tests to prevent data leakage during selection.
*   **Source files:** `src/fresh_features.py`
*   **Scripts:** `scripts/09_fresh_features.py`
*   **Tests:** `tests/test_fresh_selection.py`

## WP7: FRESH part B (classifier and analysis)
**Goal:** Train logistic regression models on the top features selected in WP6, plot AUC curves, and analyze feature overlap between cohorts to check for overfitting.
*   **Source files:** `src/fresh_eval.py`
*   **Scripts:** `scripts/10_fresh_eval.py`

## WP8: Evaluation framework, comparison, integration
**Goal:** Build the strict cross-cohort evaluation logic (preventing subject leakage), aggregate results into master tables, and maintain the main pipeline execution script.
*   **Source files:** `src/evaluation.py`, `src/compare.py`
*   **Scripts:** `scripts/04_evaluate_macd.py`, `scripts/02_reproduce_macd.py`, `scripts/14_compare_models.py`, `scripts/run_pipeline.sh`
*   **Tests:** `tests/conftest.py`, `tests/test_evaluation.py`
*   **Docs:** `README.md`, `TEAM_WORKPLAN.md`, `reports/replication_report.md`, `reports/model_comparison.md`, slide template

## WP9: Model ablation and robustness
**Goal:** Test the baseline MACD model's limits by swapping classifiers (RF, SVM, KNN), testing truncation (number of keystrokes needed), and injecting artificial timing noise.
*   **Source files:** `src/models.py`, `src/robustness.py`
*   **Scripts:** `scripts/11_model_ablation.py`, `scripts/12_keystroke_curve.py`, `scripts/13_noise_injection.py`
*   **Tests:** `tests/test_robustness.py`
*   **Docs:** `reports/robustness.md` (shared with WP3)

## WP10: Export and web app
**Goal:** Export the retrained model as JSON, build a privacy-first HTML/JS browser demo, and write parity tests to ensure the JavaScript math matches the Python backend.
*   **Source files:** `src/export_model.py`, `webapp/` (all HTML, CSS, JS files)
*   **Scripts:** `scripts/15_export_model.py`
*   **Tests:** `tests/parity_check.mjs`, `tests/test_js_parity.md`, `tests/fixtures/macd_parity.json`
*   **Outputs:** `models/macd_model.json`