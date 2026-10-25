# Phase 2 Interim Data Dictionary

**Project:** Parkinson’s Disease Predictor  
**Pipeline:** Phase 2 (Parsing, Validation, and Cleaning)  
**Date:** 24 October 2026  

---

## 1. Accepted Typing Records (`data/phase2_recovery/interim/accepted/KEY_YYMM.parquet`)

Contains all validated keystroke records that satisfied the canonical record contract, including clean records and records flagged with non-fatal review warnings.

| Column Name | PyArrow Dtype | Nullable | Units / Format | Description |
|---|---|---|---|---|
| `source_file` | `string` | No | `KEY_YYMM.txt` | Name of the source monthly keystroke log file. |
| `line_no` | `int64` | No | 1-based integer | Physical line number in the source file. |
| `user_key` | `string` | No | `[A-Z0-9]{10}` | 10-character alphanumeric participant identifier. |
| `event_time` | `timestamp[ms]` | No | UTC timestamp | Combined session date and wall-clock timestamp with millisecond resolution. |
| `hand` | `string` | No | `{'L', 'R', 'S'}` | Hand used for keystroke: `L` (Left), `R` (Right), `S` (Spacebar). |
| `hold_ms` | `float64` | No | Milliseconds (ms) | Duration the key was depressed (`>= 0.0`). Plain decimal format. |
| `direction` | `string` | No | 2-char string | Transition direction between consecutive keypresses (`LL`, `LR`, `RR`, `RL`, `LS`, `RS`, `SR`, `SL`, `SS`). |
| `latency_ms` | `float64` | No | Milliseconds (ms) | Time elapsed between consecutive key-down events (`>= 0.0`). Plain decimal format. |
| `flight_ms` | `float64` | No | Milliseconds (ms) | Time elapsed between key release and next key press. Negative values represent physical keypress overlap. Plain decimal format. |
| `warning_codes` | `string` | No | Comma-separated | Applicable warning codes (`W01_BACKWARD_TIMESTAMP`, `W02_DUPLICATE_RECORD`, `W03_DATE_FILENAME_MISMATCH`, `W04_EXTREME_HOLD`, `W05_SUSPECT_DATE_CODE_VALUE`), or empty string `""` if clean. |

---

## 2. Quarantined Records (`data/interim/quarantine/KEY_YYMM.parquet`)

Generated only for log files containing one or more corrupted, invalid, or undecodable lines.

| Column Name | PyArrow Dtype | Nullable | Format | Description |
|---|---|---|---|---|
| `source_file` | `string` | No | `KEY_YYMM.txt` | Source log filename. |
| `line_no` | `int64` | No | 1-based integer | Physical line number where corruption was detected. |
| `raw_line_bytes` | `binary` | No | Raw bytes | Exact, lossless byte sequence of the physical source line (excluding stripped line terminator). |
| `raw_line_display` | `string` | No | UTF-8 string | String decoded with replacement characters (`\ufffd`) for inspection. |
| `reason_codes` | `string` | No | Comma-separated | Applicable error codes (`E01`–`E11`). |
| `detail` | `string` | No | Semicolon-separated | Descriptive summary of validation failures. |

---

## 3. Participant Labels (`data/interim/participant_labels.csv`)

Participant diagnosis labels parsed from `data/raw/participant_metadata/User_*.txt`.

| Column Name | Dtype | Allowed Values | Description |
|---|---|---|---|
| `user_key` | `string` | `[A-Z0-9]{10}` | 10-character participant identifier. |
| `parkinsons_raw` | `string` | `"True"`, `"False"`, `""` | Raw string value extracted from `Parkinsons:` line. |
| `label` | `Int64` (nullable) | `1`, `0`, `NaN` | Binary diagnosis label: `1` = Parkinson's Disease (PD), `0` = Healthy Control, `NaN` = missing/unverified. |
| `label_status` | `string` | Category | `VALID_LABEL`, `MISSING_LABEL`, `INVALID_LABEL`, or `INVALID_DUPLICATE_FIELD`. |

---

## 4. Participant Covariates (`data/interim/participant_covariates.csv`)

Raw clinical covariates kept separate from typing time-series data.

| Column Name | Dtype | Description |
|---|---|---|
| `user_key` | `string` | 10-character participant identifier. |
| `BirthYear` | `string` | Reported birth year (e.g. `1955`). |
| `Gender` | `string` | Reported gender (e.g. `Male`, `Female`). |
| `Parkinsons` | `string` | Raw Parkinson's response. |
| `Tremors` | `string` | Reported presence of tremors (`True`/`False`). |
| `DiagnosisYear` | `string` | Year of official Parkinson's diagnosis. |
| `Sided` | `string` | Dominant or most affected side (`None`, `Left`, `Right`). |
| `UPDRS` | `string` | Unified Parkinson's Disease Rating Scale score. |
| `Impact` | `string` | Self-reported disease impact severity. |
| `Levodopa` | `string` | Levodopa medication usage (`True`/`False`). |
| `DA` | `string` | Dopamine agonist medication usage (`True`/`False`). |
| `MAOB` | `string` | MAO-B inhibitor medication usage (`True`/`False`). |
| `Other` | `string` | Other antiparkinsonian medication usage (`True`/`False`). |

---

## 5. Participant Inventory (`data/interim/participant_inventory.csv`)

Master reconciliation of all 791 distinct participants observed across logs and metadata.

| Column Name | Dtype | Allowed Values | Description |
|---|---|---|---|
| `user_key` | `string` | `[A-Z0-9]{10}` | Participant identifier. |
| `in_logs` | `bool` | `True`, `False` | True if at least one monthly log file exists. |
| `n_log_files` | `int64` | `>= 0` | Total recognized monthly log files for this participant. |
| `first_yymm` | `string` | `YYMM` or `""` | Earliest log session month. |
| `last_yymm` | `string` | `YYMM` or `""` | Latest log session month. |
| `in_metadata` | `bool` | `True`, `False` | True if `User_KEY.txt` metadata file exists. |
| `label_status` | `string` | Category | Label parsing status or `NO_METADATA`. |
| `n_lines` | `int64` | `>= 0` | Total physical lines across all participant logs. |
| `n_accepted` | `int64` | `>= 0` | Accepted lines without warnings. |
| `n_warned` | `int64` | `>= 0` | Accepted lines with warnings. |
| `n_quarantined` | `int64` | `>= 0` | Quarantined lines with errors. |
| `n_blank` | `int64` | `>= 0` | Blank physical lines. |
| `cohort_status` | `string` | Category | Primary status: `ELIGIBLE`, `EXCLUDED_NO_METADATA`, `EXCLUDED_NO_LOGS`, `EXCLUDED_NO_VALID_LABEL`, `EXCLUDED_NO_USABLE_RECORDS`. |
| `exclusion_reasons` | `string` | Semicolon list | Exhaustive list of all applicable exclusion reasons, or empty string `""` for eligible cohort members. |
