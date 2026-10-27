# MACD replication report

## Purpose

This report documents the explainable baseline and its leakage-safe evaluation. It is a research demonstration, not a diagnostic report.

## Dataset and cleaning

Source: PhysioNet neuroQWERTY MIT-CSXPD. The cleaning stage follows the original PhysioNet loader: invalid timestamps, negative or five-second-and-longer holds, press-time reversals, mouse events, long modifier keys, and BackSpace are excluded.

## Feature

MACD is the mean absolute difference between consecutive hold times when both hold times are in the session's 60th–100th percentile corridor.

## Evaluation

The model is evaluated cross-cohort: train on MIT-CS1PD and test on MIT-CS2PD, then reverse. All sessions from one subject remain together. The pooled ROC uses the common MACD score scale. Logistic probabilities are stored separately because independently fitted fold probabilities have different intercepts.

## Results

Run `./scripts/run_pipeline.sh` to refresh preparation, EDA, feature extraction, evaluation, corridor sensitivity, model export, and tests. The generated JSON/CSV files are the source of truth; numbers should not be copied manually into this report.

## Limitations

The dataset is small, clinic-collected, and not representative of all devices or typing contexts. Medication status and collection environment limit deployment claims. A high AUC does not establish clinical utility or diagnosis.
