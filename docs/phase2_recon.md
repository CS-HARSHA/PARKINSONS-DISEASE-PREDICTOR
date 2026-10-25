# Phase 2 Reconnaissance Report

**Project:** Parkinson’s Disease Predictor  
**Repository:** `CS-HARSHA/PARKINSONS-DISEASE-PREDICTOR`  
**Date:** 24 October 2026  
**Branch:** `praneeth`  
**Execution Environment:** Windows PowerShell, Python 3.13.7 (64-bit)

---

## 1. Initial Git Status & Working Tree

- **Current Branch:** `praneeth` (confirmed via `git status` and user instruction).
- **Working Tree:** Clean (no uncommitted user changes).
- **Recent Git Log:**
  - `815af99 Updated Audit`
  - `80375cc Cleaned data`
  - `1aa8006 EDA`
  - `7d2949b Merge branch 'main' of https://github.com/CS-HARSHA/PARKINSONS-DISEASE-PREDICTOR`
  - `db3e8e6 updated project structure`
  - `01cc36a 2nd`
  - `6e155ff Data`
  - `f698e14 Initial commit`

---

## 2. Tracked Data Status

Checking `git ls-files data` revealed **2,818 tracked files** under `data/`:
- `data/raw/keystroke_logs/`: 2,165 text log files + 1 ZIP archive (`Archived Users.zip`).
- `data/raw/participant_metadata/`: 646 metadata files (`User_*.txt`).
- `data/processed/`: 4 files (`features_clean.csv`, `participant_info.csv`, `train_ids.csv`, `test_ids.csv`) + `.gitkeep`.
- `data/interim/`: 1 file (`.gitkeep`).

> **Rule & Safety Observation:** Raw data is tracked in the repository. As per non-negotiable rule 1 and section 11, `data/raw` is strictly read-only, and Git history must not be rewritten. The files will remain tracked as-is, with SHA-256 pre- and post-run hashes verified.

---

## 3. Purpose of Existing Root & Notebook Files

- `README.md`: Describes repository structure and notes that raw logs and metadata are tab-delimited.
- `onefile.py`: Minimal single-line script (`print("Well just to intialiize repo :|")`) used to initialize the GitHub repository.
- `notebooks/01_dataset_audit.ipynb`: Phase 1 diagnostic notebook that scanned all 2,165 logs and 646 metadata files, generating the audit reports in `Reports/local_audit/`.
- `notebooks/02_eda_and_cleaning.ipynb` & `03_data_cleaning.ipynb`: Downstream exploratory/legacy cleaning notebooks (out of scope for Phase 2).
- `Reports/local_audit/`: 26 comprehensive audit reports documenting Phase 1 baseline metrics and structural findings.

---

## 4. Recovered Phase 1 Baseline Metrics

Recovered from `Reports/local_audit/audit_summary.json`, `audit_count_reconciliation_summary.json`, `audit_count_reconciliation_detail.csv`, and `diagnosis_signoff.md`:

| Metric | Phase 1 Recovered Value | Source Reference |
|---|---|---|
| Monthly log files scanned | 2,165 | `audit_summary.json` |
| Non-log files in keystroke_logs | 1 (`Archived Users.zip`) | Directory scan / `log_manifest.csv` |
| Unique log participants | 754 | `audit_summary.json` |
| Metadata files | 646 | `audit_summary.json` |
| Unique metadata participants | 646 | `audit_summary.json` |
| Participants in both sources | 609 | `audit_summary.json` |
| Log participants without metadata | 145 | `audit_summary.json` |
| Metadata participants without logs | 37 | `audit_summary.json` |
| Participants with valid labels and logs | 606 | `audit_summary.json` |
| Label distribution (eligible) | 438 True (PD=1), 168 False (Control=0) | `audit_summary.json` |
| Total physical event lines scanned | 40,524,886 | `audit_summary.json` |
| Blank lines | 0 | `audit_summary.json` |
| Legacy malformed rows (whitespace split) | 7 | `audit_count_reconciliation_summary.json` |
| Strict tab-split malformed (fields != 8) | 4 | `audit_count_reconciliation_summary.json` |
| Legacy participant ID mismatches | 384 | `audit_count_reconciliation_summary.json` |
| Strict tab-split participant ID mismatches | 387 | `audit_count_reconciliation_summary.json` |
| Invalid date rows | 291 | `audit_summary.json` |
| Invalid time rows | 196,341 | `audit_summary.json` |
| Invalid hand rows | 1,043 | `audit_summary.json` |
| Timing parse error rows | 464,803 | `audit_summary.json` |
| Nonpositive hold rows | 18,101 | `audit_summary.json` |
| Negative latency rows | 0 | `audit_summary.json` |
| Negative flight rows (Phase 1 flag) | 0 | `audit_summary.json` |
| Files with NUL bytes | 34 | `diagnosis_signoff.md` |
| Records with NUL bytes | 92 | `diagnosis_signoff.md` |
| NUL byte count | 36,040 | `diagnosis_signoff.md` |
| Backward timestamp transitions | 44,748 (541 > 1 sec) | `diagnosis_signoff.md` |

---

## 5. Key Reconciliations Established

### 5.1 Reconciliation of 609 Joint Participants, 606 with Valid Labels, and 597 Eligible Participants
Inspecting `Reports/local_audit/participant_audit.csv` and the actual metadata files revealed that exactly **3 participants** present in both logs and metadata have empty/unspecified metadata records:
1. `FXD2HCNIPJ` (`data/raw/participant_metadata/User_FXD2HCNIPJ.txt`)
2. `PVADT62JUM` (`data/raw/participant_metadata/User_PVADT62JUM.txt`)
3. `QIHPY6RIK3` (`data/raw/participant_metadata/User_QIHPY6RIK3.txt`)

Each of these three files contains field labels with blank values (e.g. `Parkinsons: ` with no boolean). The three missing labels leave 606 joint participants with valid labels (`438 True + 168 False = 606`). Of those, 9 have no usable accepted or accepted-with-warning records (6 `True`, 3 `False`). Therefore, the final eligible cohort contains 597 participants (`432 True + 165 False = 597`).

### 5.2 Discrepancy Between Legacy Malformed Rows (7 vs 4) and ID Mismatches (384 vs 387)
Phase 1 detailed audit resolved this exact discrepancy:
- 4 records have 7 tab-separated fields (missing a field).
- 3 records start with a leading tab `\t`:
  - `2JTCBKUP8T_1705.txt` lines 28871, 29176, 29525.
  - When split on whitespace (legacy), the leading tab is discarded and 7 fields remain, flagging it as malformed while skipping the ID check.
  - When split strictly on tabs (`\t`), field 0 is empty `""` (giving 8 fields), causing a participant ID mismatch (`"" != "2JTCBKUP8T"`).
  - Hence: $384 + 3 = 387$ ID mismatches.

---

## 6. Open Questions & Strategy for Phase 2 Tasks

1. **Delimiter & Canonical Shape (T2):**
   - The dataset documentation claims comma-delimited, while all Phase 1 checks show tab-delimited records with a trailing tab.
   - T2 will empirically verify all 2,165 log files for field count distribution, line endings (CRLF vs LF vs CR), trailing delimiters, and character encoding.
2. **Quarantine Lossless Storage:**
   - Quarantined records will retain `raw_line_bytes` as raw binary bytes, plus `raw_line_display` decoded with utf-8 replacement characters for viewing.
3. **Environment:**
   - Python 3.13.7 is the active environment. `tomllib` is built-in; `pandas` 3.0.1, `pyarrow` 25.0.1, `numpy` 2.3.5 are installed. `pytest` will be added in T1.
