# Phase 2 Engineering & Validation Decision Log

**Project:** Parkinson’s Disease Predictor  
**Pipeline:** Phase 2 (Parsing, Validation, and Cleaning Pipeline)  
**Date:** 24 October 2026  

---

## Decision D01: Empirical Verification of Tab Delimiter with Single Trailing Delimiter

- **Date:** 24 October 2026
- **Context:** Mendeley Data V3 public documentation stated that log files are comma-delimited. Phase 1 local audit observed tab-separated fields.
- **Decision:** Profile all 2,165 log files at the byte level before writing the parser. Confirmed that 100.0% of records use `\t` (40,524,886 lines) with exactly one trailing `\t` before CRLF.
- **Rule:** Strict delimiter splitting on `\t` with removal of the single expected trailing delimiter. Regex splitting or whitespace collapsing is prohibited.
- **Impact:** 40,524,882 of 40,524,886 records (99.99999%) conform exactly to the 8-data-field modal shape.
- **Owner:** Data Pipeline Lead

---

## Decision D02: CRLF Line Iteration & Lossless Physical Line Numbering

- **Date:** 24 October 2026
- **Context:** Ensuring 1-to-1 physical line number stability across all operating systems without universal-newline artifacts.
- **Decision:** Implemented pure byte-level streaming using `re.finditer(rb"\r\n|\r|\n", data)`. Confirmed 100.0% of non-empty log files use DOS/Windows CRLF (`\r\n`). Zero mixed endings or lone CRs exist in raw logs. Unterminated last lines are supported without phantom rows.
- **Impact:** Exact preservation of 1-based physical line numbers for accepted and quarantined records.
- **Owner:** Data Pipeline Lead

---

## Decision D03: Lossless Quarantine Storage for Corrupted Lines

- **Date:** 24 October 2026
- **Context:** Corrupted records may contain invalid UTF-8 bytes or embedded NUL bytes (`\x00`). Python replacement decoding (`\ufffd`) loses the original byte sequence.
- **Decision:** `quarantine/KEY_YYMM.parquet` stores `raw_line_bytes` as raw binary (`pa.binary()`), retaining the exact physical byte stream losslessly. A companion column `raw_line_display` provides replacement-decoded text for human inspection.
- **Impact:** Complete audit traceability; zero raw data loss.
- **Owner:** Data Pipeline Lead

---

## Decision D04: Permitting Negative Flight Times

- **Date:** 24 October 2026
- **Context:** Flight time is measured from key release to consecutive key press. In natural fast typing, rollover and chord keypresses produce negative flight times where the next key is pressed before the previous key is released.
- **Decision:** Negative flight time is mathematically valid and physically expected. Rule `E11_BAD_FLIGHT` rejects only non-decimal strings, explicitly permitting negative numbers.
- **Impact:** Preserves genuine fast-typing dynamics without corrupting feature distributions.
- **Owner:** Data Pipeline Lead

---

## Decision D05: Treatment of Comma-Formatted Numeric Locales

- **Date:** 24 October 2026
- **Context:** Phase 1 audit recorded 464,803 timing parse errors. Investigation during Task T2 revealed that participants running European/comma-decimal Windows locales produced values like `0066,4`.
- **Decision:** Follow Non-negotiable Rule 2 ("Never delete, repair, impute, round, reorder or deduplicate source values") and Section 5.2 ("Plain decimal: optional minus sign, at least one digit, and optionally a decimal point followed by at least one digit. Reject whitespace, plus signs, exponent notation, underscores, NaN and infinity"). Comma decimals are not auto-converted to dots during Phase 2; they fail the plain decimal contract and are safely quarantined under `E08_BAD_HOLD`, `E10_BAD_LATENCY`, and `E11_BAD_FLIGHT`.
- **Impact:** Exactly 464,803 rows quarantined with complete lossless retention in `quarantine/`, leaving accepted data 100% compliant with standard numeric types.
- **Owner:** Data Pipeline Lead

---

## Decision D06: Cohort Reconciliation (609 Joint $\to$ 606 Labeled $\to$ 597 Eligible)

- **Date:** 24 October 2026
- **Context:** An earlier audit reported 606 participants with valid labels and logs. Reconcile this with the 609 joint participants and the Phase 2 eligible cohort.
- **Decision:**
  1. Exactly 3 participants present in both sources (`FXD2HCNIPJ`, `PVADT62JUM`, `QIHPY6RIK3`) have blank metadata files with missing Parkinson's diagnosis labels (`609 - 3 = 606`).
  2. Exactly 9 participants have either 0-byte log files (2 participants: `DRQKANH3Q0`, `JORVLMISQA`) or 100% of their log records quarantined due to comma-decimal locale formatting (7 participants: `EIFBUG1US3`, `H9PJHE5OTR`, `HQW5YE0PUK`, `KM4FAMJWJ3`, `OPACZPPV1Q`, `PDKF0GKH59`, `TOC5D4KGNO`).
  3. Under the explicit Phase 2 contract ("ELIGIBLE means the participant has logs, metadata, a valid label, and at least one usable accepted or accepted-with-warning record"), these 9 participants receive primary status `EXCLUDED_NO_USABLE_RECORDS`, resulting in **597 eligible participants**.
- **Impact:** Reconciled key-by-key in `reports/data_quality/cohort_reconciliation.csv`.
- **Owner:** Data Pipeline Lead

---

## Decision D07: Safe Staging and Atomic Publishing Strategy

- **Date:** 24 October 2026
- **Context:** Preventing partial or corrupted outputs if a run aborts or if pre/post raw file hashes disagree.
- **Decision:** The pipeline processes into isolated staging directories (`staging_interim_*` and `staging_reports_*`). Invariants and raw file SHA-256 hashes are verified. Publishing to final `data/interim/` and `reports/data_quality/` occurs only after all checks pass, using an atomic directory swap with backup rollback.
- **Impact:** Previous successful outputs remain untouched upon failure; zero risk of partial or inconsistent artifacts.
- **Owner:** Data Pipeline Lead
