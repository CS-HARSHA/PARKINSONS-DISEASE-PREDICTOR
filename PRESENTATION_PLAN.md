# Presentation Plan

## Project title

**An Explainable Machine Learning Pipeline for Parkinson’s Screening Using Keystroke Dynamics**

**Subtitle:** Cross-cohort evaluation of hold-time variability and MACD

This presentation should explain the full machine-learning story: why keystroke timing may contain a motor-control signal, how the data are cleaned, what the EDA shows, why MACD is selected, how the model is evaluated, and what the exploratory model comparison found.

The focus should remain on the machine-learning pipeline. The webapp should be presented as future integration rather than the central contribution.

Recommended length: 12–18 minutes and approximately 12–17 slides.

---

## Slide 1 — Title

### Title

**An Explainable Machine Learning Pipeline for Parkinson’s Screening Using Keystroke Dynamics**

### Subtitle

**Cross-cohort evaluation of hold-time variability and MACD**

### Include

- Team members
- Institution, course, or department
- PhysioNet neuroQWERTY MIT-CSXPD dataset
- Presentation date

### Speaker message

Introduce this as a research and machine-learning demonstration, not as a medical diagnostic system. The project investigates whether timing patterns in keyboard typing can provide an interpretable signal associated with Parkinson’s disease.

Do not begin with the final AUC. First explain the problem and scientific motivation.

---

## Slide 2 — Problem motivation

### Title

**Why keystroke timing?**

Typing contains motor-control information beyond the words being typed. This project focuses on:

- Key hold time
- Change in hold time between consecutive keys
- Timing variability
- Differences between controls and PD subjects

### Suggested visual

```text
Key A pressed ───── released
                 hold time = 0.12 s

Key B pressed ─────── released
                   hold time = 0.18 s

Absolute change = |0.18 - 0.12| = 0.06 s
```

### Speaker message

The project does not analyse the words typed by participants. It focuses on timing information only. The hypothesis is that motor impairment may appear as reduced consistency or larger changes in successive key timings.

---

## Slide 3 — Research question and hypothesis

### Research question

> Can simple, interpretable features extracted from keystroke hold times distinguish PD subjects from control subjects across independent study cohorts?

### Hypothesis

PD subjects may demonstrate:

- Higher mean hold times
- Greater timing variability
- Larger absolute changes between consecutive hold times
- Higher MACD values

Make clear that this is an association and screening research question, not a diagnosis question.

---

## Slide 4 — Dataset

### Title

**Dataset: PhysioNet neuroQWERTY MIT-CSXPD**

| Cohort | Description | Subjects | Sessions |
|---|---|---:|---:|
| MIT-CS1PD | Early-PD cohort | 31 | 62 |
| MIT-CS2PD | De novo, untreated-PD cohort | 54 | 54 |
| Total | — | 85 | 116 |

The data contain keyboard event timings and hold durations in seconds, together with subject-level PD/control labels.

### Suggested visual

```text
PhysioNet neuroQWERTY
          │
 ┌────────┴────────┐
 │                 │
MIT-CS1PD       MIT-CS2PD
Early PD        De novo PD
31 subjects     54 subjects
```

### Speaker message

A cohort is a group of subjects collected under one study protocol. The project keeps the cohorts separate during evaluation to test whether the learned signal transfers between study populations.

---

## Slide 5 — End-to-end pipeline

### Title

**Modular machine-learning pipeline**

```text
Raw PhysioNet files
        ↓
Data loading
        ↓
Cleaning and quality checks
        ↓
Exploratory data analysis
        ↓
Feature extraction
        ↓
Subject-level aggregation
        ↓
Cross-cohort evaluation
        ↓
Model and feature benchmark
        ↓
Exported model for future webapp
```

### Code mapping

| Stage | Code |
|---|---|
| Loading | `src/data_loading.py` |
| Cleaning | `src/cleaning.py` |
| EDA | `src/eda.py` |
| Features | `src/features.py`, `src/feature_sets.py` |
| Aggregation | `src/feature_extraction.py` |
| Modeling | `src/modeling.py` |
| Evaluation | `src/evaluation.py` |
| Benchmarking | `src/benchmark.py` |
| Execution | `scripts/` |

### Speaker message

The `src/` directory contains reusable implementation. The `scripts/` directory contains execution entry points that run the stages and save outputs.

---

## Slide 6 — Data cleaning

### Title

**Cleaning and quality control**

The pipeline removes:

- Invalid or non-numeric rows
- Negative hold times
- Unrealistically long hold times
- Invalid timestamps
- Time-order reversals
- Mouse events
- Modifier keys such as Shift, Alt, and Control
- Backspace events
- Sessions with fewer than 20 valid keystrokes

### Suggested visual

```text
Raw key events
      ↓
Remove invalid timing rows
      ↓
Remove non-typing events
      ↓
Check timestamps and durations
      ↓
Valid hold-time sequence
```

The cleaning stage writes an audit file:

```text
data/processed/cleaning_summary.json
```

### Speaker message

Cleaning is separated from feature engineering so the team can inspect what was removed before calculating the model feature.

---

## Slide 7 — EDA: hold-time comparison

### Title

**EDA: PD subjects show different hold-time behavior**

Show the generated figure:

```text
results/figures/hold_time_by_label.png
```

### Current results

| Cohort | Control mean hold | PD mean hold |
|---|---:|---:|
| MIT-CS1PD | 0.1065 s | 0.1427 s |
| MIT-CS2PD | 0.1049 s | 0.1299 s |

### Interpretation

PD subjects have higher average hold times in both cohorts.

Use careful language:

> In this dataset, PD subjects show higher subject-level average hold times. This is an observed association and may be influenced by cohort composition, device, medication, or collection conditions.

Do not say that Parkinson’s disease causes longer hold times based on this analysis.

---

## Slide 8 — EDA: consecutive changes and MACD motivation

### Title

**Why focus on consecutive timing changes?**

Show:

- `results/figures/consecutive_change_by_label.png`
- `results/figures/macd_by_label.png`

### Consecutive-change results

| Cohort | Control consecutive change | PD consecutive change |
|---|---:|---:|
| MIT-CS1PD | 0.0313 s | 0.0511 s |
| MIT-CS2PD | 0.0266 s | 0.0429 s |

### MACD results

| Cohort | Control MACD | PD MACD |
|---|---:|---:|
| MIT-CS1PD | 0.0242 s | 0.0469 s |
| MIT-CS2PD | 0.0221 s | 0.0368 s |

### Main message

The PD groups show larger subject-level changes between consecutive hold times. MACD summarizes this behaviour into one interpretable feature.

MACD is not simply the standard deviation of hold times. It measures the average absolute change between consecutive hold times within a defined corridor.

---

## Slide 9 — MACD feature definition

### Title

**The MACD feature**

For hold times \(h_1, h_2, ..., h_n\):

1. Calculate the session’s 60th percentile.
2. Keep hold times between the 60th percentile and the session maximum.
3. Consider adjacent pairs only when both values are in that corridor.
4. Calculate the mean absolute difference.

\[
\text{MACD} = \frac{1}{K} \sum_{i \in \text{valid pairs}} |h_{i+1} - h_i|
\]

### Plain-language version

```text
MACD = average absolute change between valid consecutive hold times
```

### Example

```text
Valid hold times:
0.12, 0.18, 0.15, 0.22

Absolute changes:
0.06, 0.03, 0.07

MACD:
(0.06 + 0.03 + 0.07) / 3 = 0.0533 seconds
```

### Why use a corridor?

The corridor focuses the feature on the upper part of the hold-time distribution, where the original paper found useful signal, while reducing the influence of very short or potentially noisy events.

This is the central technical slide and should be explained slowly.

---

## Slide 10 — Subject-level aggregation

### Title

**From sessions to subjects**

```text
Keystrokes
    ↓
Session-level hold-time sequence
    ↓
Session-level MACD
    ↓
Subject-level average MACD
    ↓
Model input
```

A subject may contribute multiple sessions. The pipeline therefore:

- Calculates MACD per session
- Groups sessions by subject
- Averages session-level values
- Performs evaluation at the subject level

### Leakage rule

All sessions belonging to one subject stay in the same train or test split.

### Speaker message

This prevents the model from learning subject-specific patterns from one session and being tested on another session from the same subject.

---

## Slide 11 — Cross-cohort evaluation

### Title

**Evaluation design: cross-cohort testing**

```text
Fold 1:
Train: MIT-CS1PD
Test:  MIT-CS2PD

Fold 2:
Train: MIT-CS2PD
Test:  MIT-CS1PD
```

### Why this is important

A random split may place similar study-specific patterns in both training and testing. Cross-cohort evaluation asks whether the signal transfers to a different study cohort.

### Safeguards

- Subject-disjoint splits
- No session leakage
- No random mixing of cohorts
- Held-out cohort used only for testing
- Bootstrap confidence intervals resampled at the subject level

---

## Slide 12 — Baseline results

### Title

**MACD baseline performance**

Show:

```text
results/figures/roc_macd.png
```

### Results

```text
AUC: 0.8505
Bootstrap 95% CI: 0.7577–0.9290
```

The bootstrap interval was calculated by repeatedly resampling subjects with replacement and recomputing AUC.

The interval describes uncertainty in the estimated AUC. It is not the range of performance for individual subjects.

Because the sample is relatively small, the interval is wide. The result should not be presented as proof of clinical readiness.

---

## Slide 13 — Model and feature benchmark

### Title

**Can the baseline be improved?**

| Feature/model | Mean held-out fold AUC |
|---|---:|
| MACD + random forest | 0.8650 |
| MACD + logistic regression | 0.8513 |
| MACD + standard deviation + logistic regression | 0.8451 |
| MACD + standard deviation + mean hold + logistic regression | 0.8382 |
| Standard deviation + logistic regression | 0.8321 |
| Mean hold time + logistic regression | 0.7481 |

The small random forest using MACD alone was the exploratory leader.

### Caution

- Only two cross-cohort folds are available.
- Several models and feature sets were compared.
- The random forest was selected after looking at the benchmark.
- No independent external test dataset has confirmed the result.

Use the phrase **promising exploratory candidate**, not **proven improvement**.

---

## Slide 14 — Explainability and model choice

### Logistic regression

Advantages:

- Easy to explain
- One coefficient
- Direct relationship between MACD and score
- Easy to export to JSON
- Easy to reproduce in JavaScript

### Random forest

Advantages:

- Can capture nonlinear relationships
- Higher exploratory AUC in the current benchmark
- More interpretable than a deep neural network

Disadvantages:

- More complex
- More prone to overfitting on a small dataset
- Harder to explain in a browser
- Requires stronger validation before deployment

### Recommended position

Retain logistic regression as the paper replication and transparent baseline. Treat the random forest as an exploratory benchmark until independently validated.

---

## Slide 15 — Future webapp plan

### Title

**Future webapp architecture**

```text
Browser keydown/keyup events
          ↓
Hold-time extraction
          ↓
JavaScript MACD
          ↓
Exported model JSON
          ↓
Research screening score
```

### Design principles

- Capture timing only
- Do not store typed text
- Ignore repeated keydown events
- Require a minimum number of observations
- Warn when the value is outside the training range
- Show a research-only disclaimer
- Do not provide a diagnosis

The Python and JavaScript MACD implementations should remain equivalent through parity tests.

---

## Slide 16 — Limitations

### Dataset limitations

- Small sample size
- Only two cohorts
- Clinic-collected data
- Limited representation of devices and typing contexts
- Cohort differences may contain study-specific effects

### Modeling limitations

- Only two cross-cohort evaluation folds
- Exploratory model benchmark
- No independent external test dataset
- Confidence intervals are relatively wide
- Feature selection and model comparison need stronger validation

### Deployment limitations

- Browser timing differs from the PhysioNet logger
- Keyboard and device differences may affect hold times
- Medication status may affect motor timing
- The project must not guide clinical decisions

This slide increases credibility and should not be omitted.

---

## Slide 17 — Conclusion

Use three main points:

1. **EDA supports the feature motivation.** PD subjects showed higher average hold times, larger consecutive timing changes, and higher MACD in both cohorts.
2. **The MACD baseline is interpretable and reproducible.** It achieved approximately AUC 0.8505 under cross-cohort evaluation.
3. **The random forest is a promising exploratory candidate.** It reached mean held-out fold AUC 0.8650, but this needs stronger validation before being considered a confirmed improvement.

### Final statement

> The contribution is not only a model score. It is a reproducible, explainable, leakage-aware pipeline connecting EDA motivation to feature engineering, evaluation, benchmarking, and future web deployment.

---

## Recommended presentation flow

```text
Problem
  ↓
Why timing might matter
  ↓
Dataset
  ↓
Cleaning
  ↓
EDA evidence
  ↓
MACD definition
  ↓
Subject-level aggregation
  ↓
Cross-cohort evaluation
  ↓
Baseline results
  ↓
Model comparison
  ↓
Webapp plan
  ↓
Limitations
  ↓
Conclusion
```

Do not begin with the AUC. Begin with the motivation and EDA evidence.

---

## Division among ten people

| Person | Responsibility |
|---|---|
| 1 | Problem motivation and medical context |
| 2 | Dataset and cohort explanation |
| 3 | Data loading and cleaning |
| 4 | EDA: hold-time comparison |
| 5 | EDA: consecutive changes and MACD motivation |
| 6 | MACD mathematics and feature extraction |
| 7 | Subject aggregation and leakage prevention |
| 8 | Cross-cohort evaluation and bootstrap confidence interval |
| 9 | Model and feature benchmark |
| 10 | Webapp plan, limitations, and conclusion |

Each person should explain both the scientific meaning of their component and the corresponding code module.

---

## Presentation language rules

Use **associated with**, not **causes**.

Use **screening research score**, not **diagnosis**.

Use **exploratory improvement**, not **proven improvement**.

Always distinguish descriptive EDA from held-out model evaluation, and distinguish the paper replication baseline from the exploratory model benchmark.

---

## Most important visuals

1. End-to-end pipeline diagram
2. Hold-time comparison
3. Consecutive-change comparison
4. MACD comparison
5. MACD formula and numerical example
6. Cross-cohort split diagram
7. ROC curve
8. Model benchmark table
9. Future webapp architecture
