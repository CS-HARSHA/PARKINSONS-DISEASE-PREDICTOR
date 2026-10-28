# Replication report

## Primary result

The primary model is MACD with the paper-compatible corridor (`ql=0.60`, `qh=1.00`) and logistic regression. Evaluation is subject-safe and cross-cohort: each direction trains on one cohort and tests on the other; session scores are averaged per subject.

- Pooled subject AUC: **0.8505**
- Bootstrap 95% CI: **0.7591–0.9302**
- MIT-CS1PD → MIT-CS2PD: **0.8222**
- MIT-CS2PD → MIT-CS1PD: **0.8803**

These are the results produced by the current code and dataset. Paper values are references, not targets.

## Interpretation

The result is consistent with the approximate paper-level MACD reference, but the confidence interval is wide because the dataset contains only 85 subjects. This is a screening/research demonstration and not a clinical validation.

## Leakage controls

No subject is present in both cohorts or in both sides of a train/test direction. MACD is computed from each session, and subject scores are formed by averaging session scores.
