# Parkinson's Keystroke Screening — Slide Template

## 1. Problem and motivation
- Parkinsonian motor changes may affect keystroke timing consistency.
- Goal: evaluate an explainable timing-based screening feature.

## 2. Dataset and cohorts
- PhysioNet neuroQWERTY MIT-CS1PD and MIT-CS2PD.
- Explain subjects, sessions, labels, and cross-cohort evaluation.

## 3. Data cleaning
- Invalid timestamps, time reversals, mouse/modifier/backspace filtering.
- Show the cleaning audit and final session schema.

## 4. EDA
- Hold-time distributions.
- Consecutive timing change.
- Subject-level hold-time versus volatility scatter plot.

## 5. MACD feature
- Define the corridor and mean absolute consecutive change.
- Explain why the feature is interpretable.

## 6. Evaluation design
- Train on one cohort and test on the other.
- Keep subjects separated.
- Aggregate session predictions by subject.

## 7. Results
- ROC curve and AUC with bootstrap 95% CI.
- Report both cohort directions.

## 8. Baselines and ablations
- Mean hold, standard deviation, nQi-style model, FRESH, and classifier comparison.

## 9. Robustness
- Corridor sweep, keystroke-count curve, timing jitter, quantization, offsets, and scaling.

## 10. Export and webapp architecture
- Python pipeline trains and validates offline.
- `macd_model.json` is consumed by the browser for inference.
- Typed text is not stored or transmitted.

## 11. Limitations
- Small cohort count, browser timing mismatch, no clinical validation, and no diagnostic use.

## 12. Conclusion
- Summarize the primary result and the next validation step.
