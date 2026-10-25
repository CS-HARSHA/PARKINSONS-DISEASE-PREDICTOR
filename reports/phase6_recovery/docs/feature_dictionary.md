# Typing Feature Dictionary

**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  
**Primary Dataset:** Tappy Extended Keystroke Dataset, Mendeley Data Version 3 (DOI: `10.17632/z39mhdsynx.3`)  
**Phase:** Phase 6 — Feature Engineering & Preprocessing  
**Total Features Cataloged:** 153  
**Generated:** 2026-10-25 08:03:26 UTC  

---

## 1. Feature Taxonomy Overview

Typing features are derived deterministically from validated event-level keystroke records under the Phase 2 specification:
- **Hold Time (HT):** Duration of individual keypress ($T_{release} - T_{press}$) in milliseconds.
- **Latency (L):** Interval between consecutive key-press events ($T_{press, n+1} - T_{press, n}$) in milliseconds.
- **Flight Time (FT):** Interval between key-release and consecutive key-press ($T_{press, n+1}- T_{release, n}$) in milliseconds. The Phase 2 validator permits negative values; the accepted-cohort audit observed none, so rollover preservation was not empirically demonstrated.
- **Hand Dynamics:** Relative proportions of keys struck by left hand (`L`), right hand (`R`), or spacebar (`S`).
- **Transition Dynamics:** Relative proportions and directional timing statistics across 9 transitions: `LL`, `LR`, `RL`, `RR`, `LS`, `RS`, `SL`, `SR`, `SS`.

---

## 2. Complete Feature Inventory

| Feature Name | Family | Documented Unit | Domain Constraint | Formula / Calculation Method | Predictor Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `hold_mean` | Hold Time (HT) | ms | [0.0, Unbounded] | Key depression duration overall mean across all keystroke events | APPROVED_PREDICTOR |
| `hold_std` | Hold Time (HT) | ms | [0.0, Unbounded] | Key depression duration overall std across all keystroke events | APPROVED_PREDICTOR |
| `hold_median` | Hold Time (HT) | ms | [0.0, Unbounded] | Key depression duration overall median across all keystroke events | APPROVED_PREDICTOR |
| `hold_min` | Hold Time (HT) | ms | [0.0, Unbounded] | Key depression duration overall min across all keystroke events | APPROVED_PREDICTOR |
| `hold_max` | Hold Time (HT) | ms | [0.0, Unbounded] | Key depression duration overall max across all keystroke events | APPROVED_PREDICTOR |
| `hold_q25` | Hold Time (HT) | ms | [0.0, Unbounded] | Key depression duration overall q25 across all keystroke events | APPROVED_PREDICTOR |
| `hold_q75` | Hold Time (HT) | ms | [0.0, Unbounded] | Key depression duration overall q75 across all keystroke events | APPROVED_PREDICTOR |
| `hold_iqr` | Hold Time (HT) | ms | [0.0, Unbounded] | Key depression duration overall iqr across all keystroke events | APPROVED_PREDICTOR |
| `hold_skew` | Hold Time (HT) | unitless | Real (R) | Key depression duration overall skew across all keystroke events | APPROVED_PREDICTOR |
| `hold_kurt` | Hold Time (HT) | unitless | Real (R) | Key depression duration overall kurt across all keystroke events | APPROVED_PREDICTOR |
| `hold_cv` | Hold Time (HT) | ratio | [0.0, Unbounded] | Key depression duration overall cv across all keystroke events | APPROVED_PREDICTOR |
| `latency_mean` | Latency (L) | ms | [0.0, Unbounded] | Inter-key press interval overall mean across all keystroke events | APPROVED_PREDICTOR |
| `latency_std` | Latency (L) | ms | [0.0, Unbounded] | Inter-key press interval overall std across all keystroke events | APPROVED_PREDICTOR |
| `latency_median` | Latency (L) | ms | [0.0, Unbounded] | Inter-key press interval overall median across all keystroke events | APPROVED_PREDICTOR |
| `latency_min` | Latency (L) | ms | [0.0, Unbounded] | Inter-key press interval overall min across all keystroke events | APPROVED_PREDICTOR |
| `latency_max` | Latency (L) | ms | [0.0, Unbounded] | Inter-key press interval overall max across all keystroke events | APPROVED_PREDICTOR |
| `latency_q25` | Latency (L) | ms | [0.0, Unbounded] | Inter-key press interval overall q25 across all keystroke events | APPROVED_PREDICTOR |
| `latency_q75` | Latency (L) | ms | [0.0, Unbounded] | Inter-key press interval overall q75 across all keystroke events | APPROVED_PREDICTOR |
| `latency_iqr` | Latency (L) | ms | [0.0, Unbounded] | Inter-key press interval overall iqr across all keystroke events | APPROVED_PREDICTOR |
| `latency_skew` | Latency (L) | unitless | Real (R) | Inter-key press interval overall skew across all keystroke events | APPROVED_PREDICTOR |
| `latency_kurt` | Latency (L) | unitless | Real (R) | Inter-key press interval overall kurt across all keystroke events | APPROVED_PREDICTOR |
| `latency_cv` | Latency (L) | ratio | [0.0, Unbounded] | Inter-key press interval overall cv across all keystroke events | APPROVED_PREDICTOR |
| `flight_mean` | Flight Time (FT) | ms | Real (R) | Key release to next key press interval overall mean (negative values valid) | APPROVED_PREDICTOR |
| `flight_std` | Flight Time (FT) | ms | Real (R) | Key release to next key press interval overall std (negative values valid) | APPROVED_PREDICTOR |
| `flight_median` | Flight Time (FT) | ms | Real (R) | Key release to next key press interval overall median (negative values valid) | APPROVED_PREDICTOR |
| `flight_min` | Flight Time (FT) | ms | Real (R) | Key release to next key press interval overall min (negative values valid) | APPROVED_PREDICTOR |
| `flight_max` | Flight Time (FT) | ms | Real (R) | Key release to next key press interval overall max (negative values valid) | APPROVED_PREDICTOR |
| `flight_q25` | Flight Time (FT) | ms | Real (R) | Key release to next key press interval overall q25 (negative values valid) | APPROVED_PREDICTOR |
| `flight_q75` | Flight Time (FT) | ms | Real (R) | Key release to next key press interval overall q75 (negative values valid) | APPROVED_PREDICTOR |
| `flight_iqr` | Flight Time (FT) | ms | Real (R) | Key release to next key press interval overall iqr (negative values valid) | APPROVED_PREDICTOR |
| `flight_skew` | Flight Time (FT) | unitless | Real (R) | Key release to next key press interval overall skew (negative values valid) | APPROVED_PREDICTOR |
| `flight_kurt` | Flight Time (FT) | unitless | Real (R) | Key release to next key press interval overall kurt (negative values valid) | APPROVED_PREDICTOR |
| `flight_cv` | Flight Time (FT) | ratio | Real (R) | Key release to next key press interval overall cv (negative values valid) | APPROVED_PREDICTOR |
| `hand_freq_L` | Hand Dynamics | proportion [0, 1] | [0.0, 1.0] | Proportion of keystrokes executed by hand category L | APPROVED_PREDICTOR |
| `hand_freq_R` | Hand Dynamics | proportion [0, 1] | [0.0, 1.0] | Proportion of keystrokes executed by hand category R | APPROVED_PREDICTOR |
| `hand_freq_S` | Hand Dynamics | proportion [0, 1] | [0.0, 1.0] | Proportion of keystrokes executed by hand category S | APPROVED_PREDICTOR |
| `dir_freq_LL` | Transition Dynamics | proportion [0, 1] | [0.0, 1.0] | Proportion of transitions matching direction pattern LL | APPROVED_PREDICTOR |
| `dir_freq_LR` | Transition Dynamics | proportion [0, 1] | [0.0, 1.0] | Proportion of transitions matching direction pattern LR | APPROVED_PREDICTOR |
| `dir_freq_RL` | Transition Dynamics | proportion [0, 1] | [0.0, 1.0] | Proportion of transitions matching direction pattern RL | APPROVED_PREDICTOR |
| `dir_freq_RR` | Transition Dynamics | proportion [0, 1] | [0.0, 1.0] | Proportion of transitions matching direction pattern RR | APPROVED_PREDICTOR |
| `dir_freq_LS` | Transition Dynamics | proportion [0, 1] | [0.0, 1.0] | Proportion of transitions matching direction pattern LS | APPROVED_PREDICTOR |
| `dir_freq_RS` | Transition Dynamics | proportion [0, 1] | [0.0, 1.0] | Proportion of transitions matching direction pattern RS | APPROVED_PREDICTOR |
| `dir_freq_SL` | Transition Dynamics | proportion [0, 1] | [0.0, 1.0] | Proportion of transitions matching direction pattern SL | APPROVED_PREDICTOR |
| `dir_freq_SR` | Transition Dynamics | proportion [0, 1] | [0.0, 1.0] | Proportion of transitions matching direction pattern SR | APPROVED_PREDICTOR |
| `dir_freq_SS` | Transition Dynamics | proportion [0, 1] | [0.0, 1.0] | Proportion of transitions matching direction pattern SS | APPROVED_PREDICTOR |
| `hold_mean_LL` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration mean for LL transitions | APPROVED_PREDICTOR |
| `latency_mean_LL` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency mean for LL transitions | APPROVED_PREDICTOR |
| `flight_mean_LL` | Directional Flight Time | ms | Real (R) | Flight duration mean for LL transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_std_LL` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration std for LL transitions | APPROVED_PREDICTOR |
| `latency_std_LL` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency std for LL transitions | APPROVED_PREDICTOR |
| `flight_std_LL` | Directional Flight Time | ms | Real (R) | Flight duration std for LL transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_median_LL` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration median for LL transitions | APPROVED_PREDICTOR |
| `latency_median_LL` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency median for LL transitions | APPROVED_PREDICTOR |
| `flight_median_LL` | Directional Flight Time | ms | Real (R) | Flight duration median for LL transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_iqr_LL` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration iqr for LL transitions | APPROVED_PREDICTOR |
| `latency_iqr_LL` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency iqr for LL transitions | APPROVED_PREDICTOR |
| `flight_iqr_LL` | Directional Flight Time | ms | Real (R) | Flight duration iqr for LL transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_mean_LR` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration mean for LR transitions | APPROVED_PREDICTOR |
| `latency_mean_LR` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency mean for LR transitions | APPROVED_PREDICTOR |
| `flight_mean_LR` | Directional Flight Time | ms | Real (R) | Flight duration mean for LR transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_std_LR` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration std for LR transitions | APPROVED_PREDICTOR |
| `latency_std_LR` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency std for LR transitions | APPROVED_PREDICTOR |
| `flight_std_LR` | Directional Flight Time | ms | Real (R) | Flight duration std for LR transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_median_LR` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration median for LR transitions | APPROVED_PREDICTOR |
| `latency_median_LR` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency median for LR transitions | APPROVED_PREDICTOR |
| `flight_median_LR` | Directional Flight Time | ms | Real (R) | Flight duration median for LR transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_iqr_LR` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration iqr for LR transitions | APPROVED_PREDICTOR |
| `latency_iqr_LR` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency iqr for LR transitions | APPROVED_PREDICTOR |
| `flight_iqr_LR` | Directional Flight Time | ms | Real (R) | Flight duration iqr for LR transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_mean_RL` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration mean for RL transitions | APPROVED_PREDICTOR |
| `latency_mean_RL` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency mean for RL transitions | APPROVED_PREDICTOR |
| `flight_mean_RL` | Directional Flight Time | ms | Real (R) | Flight duration mean for RL transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_std_RL` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration std for RL transitions | APPROVED_PREDICTOR |
| `latency_std_RL` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency std for RL transitions | APPROVED_PREDICTOR |
| `flight_std_RL` | Directional Flight Time | ms | Real (R) | Flight duration std for RL transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_median_RL` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration median for RL transitions | APPROVED_PREDICTOR |
| `latency_median_RL` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency median for RL transitions | APPROVED_PREDICTOR |
| `flight_median_RL` | Directional Flight Time | ms | Real (R) | Flight duration median for RL transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_iqr_RL` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration iqr for RL transitions | APPROVED_PREDICTOR |
| `latency_iqr_RL` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency iqr for RL transitions | APPROVED_PREDICTOR |
| `flight_iqr_RL` | Directional Flight Time | ms | Real (R) | Flight duration iqr for RL transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_mean_RR` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration mean for RR transitions | APPROVED_PREDICTOR |
| `latency_mean_RR` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency mean for RR transitions | APPROVED_PREDICTOR |
| `flight_mean_RR` | Directional Flight Time | ms | Real (R) | Flight duration mean for RR transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_std_RR` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration std for RR transitions | APPROVED_PREDICTOR |
| `latency_std_RR` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency std for RR transitions | APPROVED_PREDICTOR |
| `flight_std_RR` | Directional Flight Time | ms | Real (R) | Flight duration std for RR transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_median_RR` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration median for RR transitions | APPROVED_PREDICTOR |
| `latency_median_RR` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency median for RR transitions | APPROVED_PREDICTOR |
| `flight_median_RR` | Directional Flight Time | ms | Real (R) | Flight duration median for RR transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_iqr_RR` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration iqr for RR transitions | APPROVED_PREDICTOR |
| `latency_iqr_RR` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency iqr for RR transitions | APPROVED_PREDICTOR |
| `flight_iqr_RR` | Directional Flight Time | ms | Real (R) | Flight duration iqr for RR transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_mean_LS` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration mean for LS transitions | APPROVED_PREDICTOR |
| `latency_mean_LS` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency mean for LS transitions | APPROVED_PREDICTOR |
| `flight_mean_LS` | Directional Flight Time | ms | Real (R) | Flight duration mean for LS transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_std_LS` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration std for LS transitions | APPROVED_PREDICTOR |
| `latency_std_LS` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency std for LS transitions | APPROVED_PREDICTOR |
| `flight_std_LS` | Directional Flight Time | ms | Real (R) | Flight duration std for LS transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_median_LS` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration median for LS transitions | APPROVED_PREDICTOR |
| `latency_median_LS` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency median for LS transitions | APPROVED_PREDICTOR |
| `flight_median_LS` | Directional Flight Time | ms | Real (R) | Flight duration median for LS transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_iqr_LS` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration iqr for LS transitions | APPROVED_PREDICTOR |
| `latency_iqr_LS` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency iqr for LS transitions | APPROVED_PREDICTOR |
| `flight_iqr_LS` | Directional Flight Time | ms | Real (R) | Flight duration iqr for LS transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_mean_RS` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration mean for RS transitions | APPROVED_PREDICTOR |
| `latency_mean_RS` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency mean for RS transitions | APPROVED_PREDICTOR |
| `flight_mean_RS` | Directional Flight Time | ms | Real (R) | Flight duration mean for RS transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_std_RS` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration std for RS transitions | APPROVED_PREDICTOR |
| `latency_std_RS` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency std for RS transitions | APPROVED_PREDICTOR |
| `flight_std_RS` | Directional Flight Time | ms | Real (R) | Flight duration std for RS transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_median_RS` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration median for RS transitions | APPROVED_PREDICTOR |
| `latency_median_RS` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency median for RS transitions | APPROVED_PREDICTOR |
| `flight_median_RS` | Directional Flight Time | ms | Real (R) | Flight duration median for RS transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_iqr_RS` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration iqr for RS transitions | APPROVED_PREDICTOR |
| `latency_iqr_RS` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency iqr for RS transitions | APPROVED_PREDICTOR |
| `flight_iqr_RS` | Directional Flight Time | ms | Real (R) | Flight duration iqr for RS transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_mean_SL` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration mean for SL transitions | APPROVED_PREDICTOR |
| `latency_mean_SL` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency mean for SL transitions | APPROVED_PREDICTOR |
| `flight_mean_SL` | Directional Flight Time | ms | Real (R) | Flight duration mean for SL transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_std_SL` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration std for SL transitions | APPROVED_PREDICTOR |
| `latency_std_SL` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency std for SL transitions | APPROVED_PREDICTOR |
| `flight_std_SL` | Directional Flight Time | ms | Real (R) | Flight duration std for SL transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_median_SL` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration median for SL transitions | APPROVED_PREDICTOR |
| `latency_median_SL` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency median for SL transitions | APPROVED_PREDICTOR |
| `flight_median_SL` | Directional Flight Time | ms | Real (R) | Flight duration median for SL transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_iqr_SL` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration iqr for SL transitions | APPROVED_PREDICTOR |
| `latency_iqr_SL` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency iqr for SL transitions | APPROVED_PREDICTOR |
| `flight_iqr_SL` | Directional Flight Time | ms | Real (R) | Flight duration iqr for SL transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_mean_SR` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration mean for SR transitions | APPROVED_PREDICTOR |
| `latency_mean_SR` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency mean for SR transitions | APPROVED_PREDICTOR |
| `flight_mean_SR` | Directional Flight Time | ms | Real (R) | Flight duration mean for SR transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_std_SR` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration std for SR transitions | APPROVED_PREDICTOR |
| `latency_std_SR` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency std for SR transitions | APPROVED_PREDICTOR |
| `flight_std_SR` | Directional Flight Time | ms | Real (R) | Flight duration std for SR transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_median_SR` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration median for SR transitions | APPROVED_PREDICTOR |
| `latency_median_SR` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency median for SR transitions | APPROVED_PREDICTOR |
| `flight_median_SR` | Directional Flight Time | ms | Real (R) | Flight duration median for SR transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_iqr_SR` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration iqr for SR transitions | APPROVED_PREDICTOR |
| `latency_iqr_SR` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency iqr for SR transitions | APPROVED_PREDICTOR |
| `flight_iqr_SR` | Directional Flight Time | ms | Real (R) | Flight duration iqr for SR transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_mean_SS` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration mean for SS transitions | APPROVED_PREDICTOR |
| `latency_mean_SS` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency mean for SS transitions | APPROVED_PREDICTOR |
| `flight_mean_SS` | Directional Flight Time | ms | Real (R) | Flight duration mean for SS transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_std_SS` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration std for SS transitions | APPROVED_PREDICTOR |
| `latency_std_SS` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency std for SS transitions | APPROVED_PREDICTOR |
| `flight_std_SS` | Directional Flight Time | ms | Real (R) | Flight duration std for SS transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_median_SS` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration median for SS transitions | APPROVED_PREDICTOR |
| `latency_median_SS` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency median for SS transitions | APPROVED_PREDICTOR |
| `flight_median_SS` | Directional Flight Time | ms | Real (R) | Flight duration median for SS transitions (negative values valid) | APPROVED_PREDICTOR |
| `hold_iqr_SS` | Directional Hold Time | ms | [0.0, Unbounded] | Key hold duration iqr for SS transitions | APPROVED_PREDICTOR |
| `latency_iqr_SS` | Directional Latency | ms | [0.0, Unbounded] | Inter-key latency iqr for SS transitions | APPROVED_PREDICTOR |
| `flight_iqr_SS` | Directional Flight Time | ms | Real (R) | Flight duration iqr for SS transitions (negative values valid) | APPROVED_PREDICTOR |

---

## 3. Governance and Non-Predictor Allowlist

The following columns are strictly excluded from the predictor matrix:
- `participant_id`: 10-character anonymized participant identifier (traceability key only).
- `parkinsons_label`: Binary ground truth target (0 = Control, 1 = Parkinson's Disease). Joined only for model evaluation within CV loops.
- Clinical & Demographic covariates: `Age`, `Gender`, `UPDRS`, `Tremor`, `DiagnosisYear`, `Levodopa`, etc., are maintained in `data/interim/participant_covariates.csv` and are **prohibited** from entering the primary typing-only feature matrix.
