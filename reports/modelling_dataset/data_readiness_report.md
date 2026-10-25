> **Historical Phase 4 snapshot / superseded for current outputs.** Detailed corruption and rollover statements refer to the legacy `features_clean.csv`, not the regenerated Phase 6 feature table. Use `reports/phase6_recovery/` and the executed Phase 5 notebook for current artifacts.

# Phase 4 Data Readiness and Cohort Verification Report

**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  
**Repository:** `CS-HARSHA/PARKINSONS-DISEASE-PREDICTOR`  
**Working Branch:** `praneeth`  
**Date:** 24 October 2026  
**Primary Dataset:** Tappy Extended Keystroke Dataset, Mendeley Data Version 3 (DOI: `10.17632/z39mhdsynx.3`)  
**Phase Status:** **BLOCKED FROM CONSTRUCTING FINAL MODELLING PARQUET** (Readiness & Audit Complete)  

---

## 1. Executive Summary

Phase 4 evaluated the empirical readiness of the dataset for participant-level machine learning by reconciling the participant cohort from verified Phase 2 interim outputs, auditing candidate feature tables, and validating participant-disjoint splits.

Key findings:
1. **Definitive Eligible Cohort Frozen:** The authoritative eligible modeling cohort is strictly **597 participants** (432 Parkinson's Disease [72.36%], 165 Healthy Controls [27.64%]), verified against [`data/phase2_recovery/interim/participant_inventory.csv`](../../data/phase2_recovery/interim/participant_inventory.csv) and [`data/phase2_recovery/interim/participant_labels.csv`](../../data/phase2_recovery/interim/participant_labels.csv).
2. **Reconciliation of Historical Figures:** The historical 606 labeled count differs from the 597 eligible count by exactly **-9 participants** who contributed zero usable records (2 with 0-byte log files and 7 whose monthly logs contained 100% comma-decimal formatted numbers quarantined in Phase 2).
3. **Severe Integrity Failures in Legacy Feature Table:** The historical candidate table `data/processed/features_clean.csv` (495 rows) fails validation on multiple critical criteria:
   - **Participant Coverage Gap:** It omits **102 valid eligible participants** (17.1% cohort truncation) due to an unapproved `MIN_KEYSTROKES = 1000` rule hardcoded in legacy notebook `03_data_cleaning.ipynb`.
   - **Shifted Date-Code Corruption:** **28 columns** contain corrupted values $> 100,000$ (affecting up to 126 participants in `lat2_max` and 52 in `lat1_max`, with kurtosis values reaching ~720,000) caused by 6-digit YYMMDD dates read into latency columns.
   - **Destroyed Rollover Dynamics:** All negative flight times ($FT \le 0$, representing genuine typing overlap and rollover) were erroneously coerced to NaN.
4. **Formal Phase 4 Blocker Declared:** In accordance with non-negotiable Phase 4 rules ("Phase 4 must not create new typing features from raw events. Feature engineering belongs to Phase 6. If no trustworthy feature table is available, document the blocker and stop instead of generating features prematurely"), **Phase 4 is formally blocked from constructing `modelling_dataset.parquet`**. No fabricated, imputed, or clipped modeling table was created from corrupted inputs.
5. **Prospective Split Validated:** A stratified participant-disjoint 80/20 split on all 597 eligible participants was validated under `data/phase2_recovery/splits/` (477 training: 345 PD, 132 control; 120 held-out test: 87 PD, 33 control). The historical split (396/99) remains rejected because it partitions the truncated 495-row subset.

---

## 2. Verified Cohort Flow and Accounting

The multi-stage participant flow establishes complete, lossless accounting across all 791 unique keys discovered:

| Stage | Stage Description | Total Participants | PD (True) | Control (False) | Unlabeled | Delta |
|---|---|---|---|---|---|---|
| 1 | Total Unique Participant Keys | 791 | N/A | N/A | 791 | Baseline |
| 2 | Keys in Keystroke Logs | 754 | N/A | N/A | 754 | -37 (Metadata only) |
| 3 | Joint Intersection (Logs & Metadata) | 609 | N/A | N/A | 609 | -145 (Logs only) |
| 4 | Joint Keys with Valid Diagnosis Label | 606 | 438 | 168 | 0 | -3 (Missing labels) |
| 5 | **Eligible Modeling Cohort (Usable Records)** | **597** | **432** | **165** | **0** | **-9 (Zero usable records)** |
| 6 | Legacy Feature Table (`features_clean.csv`) | 495 | 357 | 138 | 0 | -102 (Dropped: < 1000 keys) |

### Detailed Discrepancy Reconciliation
- **37 Metadata Keys without Logs:** Participants who submitted survey questionnaires but contributed zero keystroke log files.
- **145 Log Keys without Metadata:** Participants who logged typing sessions but lack a `User_KEY.txt` metadata file.
- **3 Joint Keys Missing Labels:** `FXD2HCNIPJ`, `PVADT62JUM`, `QIHPY6RIK3` have empty metadata files where `Parkinsons:` is blank.
- **9 Joint Labeled Keys with No Usable Records:**
  - 2 zero-byte log files: `DRQKANH3Q0`, `JORVLMISQA`.
  - 7 participants with 100% comma-decimal quarantined lines: `EIFBUG1US3`, `H9PJHE5OTR`, `HQW5YE0PUK`, `KM4FAMJWJ3`, `OPACZPPV1Q`, `PDKF0GKH59`, `TOC5D4KGNO`.
- **102 Eligible Participants Omitted by Legacy Table:** All 102 omitted participants have valid labels (75 PD, 27 Control) and between 1 and 931 valid keystrokes. They were silently excluded by the legacy notebook's `MIN_KEYSTROKES = 1000` filter.

---

## 3. Label Validation Results

- **Authoritative Source:** [`data/phase2_recovery/interim/participant_labels.csv`](../../data/phase2_recovery/interim/participant_labels.csv).
- **Identifier Uniqueness:** 100.0% unique `user_key` entries; zero duplicates.
- **Mapping Verification:** Binary mapping confirmed:
  - `Parkinsons: True` (case-insensitive) $\to 1.0$ (Parkinson's Disease).
  - `Parkinsons: False` (case-insensitive) $\to 0.0$ (Healthy Control).
- **Consistency across Joins:** Zero label discrepancies between interim tables and candidate tables.
- **Class Imbalance:**
  - Full Eligible Cohort: 432 PD (72.36%) vs. 165 Control (27.64%).
  - Predeclared evaluation metric in [`docs/experimental_protocol.md`](../../docs/experimental_protocol.md): **Balanced Accuracy** (required to penalize majority-class bias).

---

## 4. Feature Table Provenance and Integrity Audit

### 4.1 Provenance of `features_clean.csv`
- **Origin:** Generated in commit `80375cc` via notebook `notebooks/03_data_cleaning.ipynb`.
- **Pre-Phase 2 Lineage:** The script read raw log text files directly using `pd.read_csv(sep='\t')` before the Phase 2 strict byte-level parser, CRLF line iterator, and rule engine were built.
- **Target Inclusion:** Target label `parkinsons_label` was joined directly into the feature table.

### 4.2 Critical Integrity Failures
1. **Date Corruption in Latency Fields (28 Columns Affected):**
   - In raw logs, occasional rows had shifted delimiters where the 6-digit `Date` field (e.g., `160730`, `171226`) appeared in timing columns.
   - The legacy script called `pd.to_numeric(errors='coerce')` without checking for date-value ranges. As a result, dates $> 100,000$ were parsed as millisecond timings.
   - `lat2_max` has **126 participants** with values $> 100,000$ (max: `171226.0`).
   - `lat1_max` has **52 participants** with values $> 100,000$ (max: `171226.0`).
   - Kurtosis columns were catastrophically inflated (e.g., `lat1_kurt` max: `720,079.58`).
2. **Destruction of Negative Flight Times (Rollover Dynamics):**
   - In Cell 18 of `03_data_cleaning.ipynb`, the author wrote: `df.loc[df['latency2_ms'] <= 0, 'latency2_ms'] = np.nan`.
   - Field 7 is physical flight time ($FT$). Negative values represent genuine keystroke rollover where key $i+1$ is depressed before key $i$ is released. Coercing negative values to NaN eliminated typing overlap signals and biased distribution statistics.
3. **Missing Value Incompleteness:**
   - **54 columns** contain NaN values, primarily due to participants lacking specific bimanual transition directions (e.g., `SL`, `SR`).
4. **Severe Cohort Under-Coverage:**
   - Omitted 102 eligible participants (17.1% of the eligible cohort).

### 4.3 Feature Audit Verdict
**VERDICT: REJECTED & BLOCKED FOR MODELING.**  
The table `features_clean.csv` cannot be used as an input to machine learning. Feeding date-corrupted numbers and distorted flight times into classifiers would produce spurious, unscientific results.

---

## 5. Audit of Participant Splits

### 5.1 Historical Split Files (`data/processed/train_ids.csv`, `data/processed/test_ids.csv`)
- **Historical Counts:** 396 Training IDs, 99 Test IDs (Total: 495).
- **Audit Outcome:** **REJECTED FOR MODELING.** These split files partition the truncated 495-row subset and completely omit 102 valid eligible participants. They are retained in `data/processed/` unchanged strictly for historical provenance and audit traceability.
- **Rule:** Downstream pipelines must never load these files as the authoritative split.

### 5.2 Authoritative 80/20 Stratified Participant Split
In strict adherence to [`docs/experimental_protocol.md`](../../docs/experimental_protocol.md) Section 9:
- **Eligible Cohort Partitioned:** Exactly 597 participants (432 PD, 165 Control).
- **Split Ratio:** 80% Training / 20% Held-out Test.
- **Stratification Variable:** Target label (`label`: PD vs Control).
- **Random Seed:** `random_state = 42`.
- **Partition Allocation:**
  - **Training Partition:** **477 participants** (345 PD [72.33%], 132 Control [27.67%]).
  - **Held-out Test Partition:** **120 participants** (87 PD [72.50%], 33 Control [27.50%]).
- **Authoritative Split File Locations:**
  - **Recovery Archive:** [`data/phase2_recovery/splits/train_ids.csv`](../../data/phase2_recovery/splits/train_ids.csv), [`data/phase2_recovery/splits/test_ids.csv`](../../data/phase2_recovery/splits/test_ids.csv), and [`data/phase2_recovery/splits/split_manifest.json`](../../data/phase2_recovery/splits/split_manifest.json).
  - **Runtime Working Mirror:** `data/phase2_recovery/splits/train_ids.csv` and `data/phase2_recovery/splits/test_ids.csv` (Git-ignored per `.gitignore`).
- **Validation:**
  - Zero partition overlap confirmed ($train \cap test = \emptyset$).
  - Exact union coverage confirmed ($train \cup test = \text{eligible cohort}$).
  - Deterministic reproducibility verified via `generate_prospective_split(random_state=42)`.

---

## 6. Phase 4 Blocker Statement (Historical Snapshot) and Phase 6 Handoff

> **Phase 1?6 closure status (25 October 2026):** The blocker statement below records the position at the Phase 4 review; it is not the current project status. Phase 6 subsequently generated `data/phase6_recovery/features/participant_features_raw.parquet` and `participant_feature_quality.parquet` for all 597 eligible participants. The feature table contains 153 approved predictors, plus participant ID and target. The independent Phase 6 audit passed its 16 checks. Phase 5 EDA also executed with no notebook cell errors and produced six diagnostic figures in `reports/phase5_recovery/eda/figures/`. The authoritative recovery split remains `data/phase2_recovery/splits/` (477 train / 120 test, zero overlap). These recovery outputs are verified in this working tree; their availability in another clone depends on their being packaged or regenerated.


> [!IMPORTANT]
> **PHASE 4 BLOCKER STATEMENT:**  
> A trustworthy participant-level feature table for the 597 eligible cohort does not currently exist in the repository. The legacy table `features_clean.csv` is corrupted and covers only 495 participants.  
> Under Phase 4 non-negotiable rules, feature extraction from raw event records belongs strictly to **Phase 6 (Feature Engineering)**. Phase 4 is therefore **complete in its auditing, cohort verification, and prospective split generation responsibilities**, but **formally blocked from assembling `modelling_dataset.parquet`**.

### Prerequisites for Phase 6 Feature Engineering
When Phase 6 is authorized, the feature pipeline must:
1. Aggregate timing features directly from validated Phase 2 records in [`data/phase2_recovery/interim/accepted/`](../../data/phase2_recovery/interim/accepted).
2. Process all **597 eligible participants** identified in [`reports/modelling_dataset/split_manifest.json`](split_manifest.json).
3. Compute mathematically sound features:
   - Hold time statistics ($HT \ge 0$).
   - Inter-key latency statistics ($L \ge 0$).
   - Genuine flight time statistics, preserving negative rollover values ($FT < 0$).
   - Directional transition metrics.
4. Exclude all clinical and demographic metadata from predictor matrix $X$.
5. Evaluate sensitivity to minimum keystroke thresholds (e.g., comparing full cohort vs $\ge 500$ keys) within the prospective cross-validation folds.

---

## 7. Strict Phase Boundary Confirmation (Phase 4 Point-in-Time Record)

- No exploratory data analysis (EDA) or distribution plots were created (Phase 5).
- No new features were generated from raw event records (Phase 6).
- No feature scaling, normalization, imputation, or clipping was applied (Phase 6).
- No machine learning classifiers were trained (Phase 7).
- Phase 2 outputs and Phase 3 protocol documents remain completely untouched.
