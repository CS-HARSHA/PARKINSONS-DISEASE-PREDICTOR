# Parkinson's Keystroke Screening Pipeline

This repository implements a simple, readable, leakage-aware research pipeline for Parkinson's screening from keystroke timing. It replicates the paper-compatible MACD feature, adds simple, nQi-style, FRESH, classifier, keystroke-count, corridor, feature-variant, and browser-noise comparisons, and exports a static browser demo.

This is a screening/research demonstration only. It is not a diagnostic system and must not guide treatment.

## Primary measured result

The primary model is MACD with the paper-compatible corridor (`ql=0.60`, `qh=1.00`) and logistic regression. Subject-safe cross-cohort evaluation produced:

- Pooled AUC: **0.8505**
- Subject-bootstrap 95% CI: **0.7591–0.9302**
- MIT-CS1PD → MIT-CS2PD: **0.8222**
- MIT-CS2PD → MIT-CS1PD: **0.8803**

These values are generated from `results/metrics/macd_cross_cohort.json`. Paper values are references, not targets.

## Data and contracts

The PhysioNet neuroQWERTY MIT-CSXPD archive is placed in `data/raw/`. The two cohorts are MIT-CS1PD (early PD) and MIT-CS2PD (de novo/untreated PD), with 85 subjects and 116 valid sessions in this run.

Every processed session follows the shared schema:

```text
session_id, subject_id, cohort, label,
hold_times_s, press_times_s, n_keystrokes, source_file
```

Hold times and JSON values are seconds. Browser event timestamps are milliseconds and are converted to seconds immediately.

## Full pipeline

Run the complete pipeline from the repository root:

```bash
python3 -m pip install -r requirements.txt
./scripts/run_pipeline.sh
```

The runner executes:

```text
01_prepare_data.py       clean and save sessions
02_eda.py                descriptive EDA and figures
03_extract_features.py  session and subject feature tables
04_evaluate_macd.py     primary MACD evaluation
05_corridor_sensitivity.py
06_feature_variants.py
07_baselines_simple.py
08_baselines_nqi.py
09_fresh_features.py
10_fresh_eval.py
11_model_ablation.py
12_keystroke_curve.py
13_noise_injection.py
14_compare_models.py
15_export_model.py
pytest -q
```

The raw archive must exist before stage 01. The pipeline writes the processed session schema, cleaning audit, feature tables, saved metrics, figures, reports, and `models/macd_model.json`.

## Key outputs

- `data/processed/sessions.json`: cleaned session records
- `data/processed/cleaning_summary.json`: cleaning audit
- `data/processed/session_features.csv`: one row per session
- `data/processed/subject_features.csv`: sessions averaged per subject
- `data/processed/nqi_window_features.csv`: nQi-style window features
- `data/processed/fresh_features.csv`: cached FRESH/tsfresh features
- `results/metrics/`: all generated numeric results
- `results/figures/`: generated plots
- `reports/eda.md`: EDA and volatility motivation
- `reports/replication_report.md`: primary MACD result
- `reports/model_comparison.md`: model and baseline comparison
- `reports/robustness.md`: corridor, keystroke-count, and timing-perturbation analysis
- `models/macd_model.json`: browser-exported MACD logistic model

## Evaluation rules

All main evaluation is cross-cohort and subject-safe. A fresh model is fitted for each direction. Session scores are averaged per subject before AUC. Supervised feature selection and scaling are fitted only on training sessions. The paper-compatible primary pooled AUC uses the common MACD feature ranking so separately fitted fold intercepts cannot distort the pooled ROC scale; direction-specific AUCs are also saved.

## Webapp

After model export, serve the repository root:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000/webapp/`. The static webapp captures keyboard timing in memory, never reads or transmits typed text, computes the same MACD as Python, loads `models/macd_model.json`, and displays a relative research band plus an experimental model output. It is not validated on browser timings and does not provide a diagnosis.

Run the browser parity test with:

```bash
node tests/parity_check.mjs
```

## Code organization

`src/` contains reusable scientific and engineering modules. `scripts/` contains executable pipeline entry points. `tests/` protects cleaning, MACD, aggregation, leakage, and browser parity contracts. `notebooks/01_explore_data.ipynb` provides an interactive view over the canonical `src/` functions.

## Sources

- PhysioNet neuroQWERTY MIT-CSXPD: https://physionet.org/content/nqmitcsxpd/
- Milne, Farrahi & Nicolaou, *Less is More*: https://research.gold.ac.uk/id/eprint/24111/1/Less%20is%20More%20Univariate%20Modelling%20to%20Detect%20Early%20Parkinson%27s%20Disease%20from%20Keystroke%20Dynamics.pdf
