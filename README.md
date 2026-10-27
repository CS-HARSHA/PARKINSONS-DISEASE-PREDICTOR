# Parkinson's Keystroke ML Pipeline

This repository is a deliberately simple and explainable research pipeline for Parkinson's screening from keystroke hold times. The current baseline is the MACD feature from Milne, Farrahi, and Nicolaou: the mean absolute change between consecutive cleaned hold times when both values lie in the session's 60th–100th percentile corridor. The model is a one-feature logistic regression, with the pooled ROC reported on the common MACD score scale.

This is not a diagnostic system and must not be used to diagnose Parkinson's disease or guide treatment.

## Pipeline layers

The repository separates responsibilities so each part can be developed and reviewed independently. `src/data_loading.py` finds the PhysioNet archive, joins ground-truth metadata to files, and creates `Session` objects. `src/cleaning.py` applies the original loader's sanity checks and records every removal category. `src/eda.py` summarizes cohort sizes, labels, session lengths, and feature ranges. `src/features.py` contains mathematical feature definitions only. `src/feature_extraction.py` turns sessions into session-level and subject-level tables. `src/evaluation.py` owns subject-safe cross-cohort evaluation, AUC, Youden cutoffs, and bootstrap intervals. Reporting scripts write JSON/CSV artifacts; the future web app should consume only exported model artifacts and never raw text.

## What cross-cohort means

A cohort is a study subgroup collected under one experiment. `MIT-CS1PD` is the early-PD cohort with 31 subjects, and `MIT-CS2PD` is the de novo, untreated-PD cohort with 54 subjects. Cross-cohort evaluation trains on all eligible subjects in one cohort and tests on subjects in the other cohort; then it swaps the direction. The two held-out prediction sets are pooled for one ROC. This is stricter than a random split over all rows because it tests transfer across collection cohorts and prevents sessions from the same subject appearing in both train and test.

## Run the modular pipeline

```bash
python3 -m pip install -r requirements.txt
python3 scripts/01_prepare_data.py
python3 scripts/02_eda.py
python3 scripts/03_extract_features.py
python3 scripts/04_evaluate_macd.py
python3 scripts/05_corridor_sensitivity.py
python3 scripts/06_export_model.py
python3 scripts/07_benchmark_models.py
pytest -q
```

The raw archive belongs in `data/raw/neuroQWERTY.zip`. Raw and processed data are gitignored. The preparation stage writes cleaned sessions to `data/processed/sessions.json` and an audit log to `data/processed/cleaning_summary.json`. EDA writes `results/metrics/eda_summary.json` and `reports/eda.md`. Feature extraction writes `data/processed/session_features.csv` and `data/processed/subject_features.csv`. Evaluation writes the MACD metrics JSON, subject predictions CSV, and ROC figure.

The validated replication currently gives approximately **AUC 0.8505**, with a subject-bootstrap 95% interval of approximately **0.7577–0.9290**. The reported number comes from the generated metrics JSON, not a manually maintained report table.

The benchmark stage compares the frozen MACD baseline with standard deviation, mean hold time, MACD plus standard deviation, and MACD plus simple distributional features. It evaluates logistic regression, a small random forest, and an RBF SVM using the same two cross-cohort test folds. The primary comparison is the mean of the two held-out fold AUCs; pooled model outputs are marked descriptive because independently fitted model scores may not share one calibration scale. A benchmark result is not considered a real improvement unless it is evaluated without selecting features or hyperparameters on the held-out cohort.

On the current data, the exploratory leader is random forest using MACD alone, with mean held-out fold AUC **0.8650** versus **0.8513** for logistic regression. This is promising but not yet a definitive improvement claim because there are only two cross-cohort folds and no independent external dataset. The generated comparison is documented in `reports/model_benchmark.md`.

After running the pipeline, the later webapp scaffold can be served from the repository root with `python3 -m http.server 8000`. Open `http://localhost:8000/webapp/`. The browser app is intentionally a research scaffold: it captures timing values locally, does not store typed text, and should not be treated as a clinical tool.

## Project coordination

See [TEAM_WORKPLAN.md](TEAM_WORKPLAN.md) for ten separable workstreams, [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for table schemas, and [docs_GLOSSARY.md](docs_GLOSSARY.md) for the terminology used in reports and meetings. Every contributor should add tests for public interfaces and record the command used to produce any reported metric.

## Sources

- PhysioNet neuroQWERTY MIT-CSXPD: https://physionet.org/content/nqmitcsxpd/
- Milne, Farrahi & Nicolaou, *Less is More*: https://research.gold.ac.uk/id/eprint/24111/1/Less%20is%20More%20Univariate%20Modelling%20to%20Detect%20Early%20Parkinson%27s%20Disease%20from%20Keystroke%20Dynamics.pdf
