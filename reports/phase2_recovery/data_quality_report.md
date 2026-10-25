# Phase 2 Data Quality & Validation Report

**Project:** Parkinson’s Disease Predictor  
**Pipeline Run Date:** 2026-10-24  
**Git Commit:** `420f1159e7e440cf3d8ff49979e171042f72fad4` (Dirty: False)  
**Python Environment:** 3.13.7  
**Raw Files Integrity:** VERIFIED IDENTICAL  

---

## 1. Physical Line Outcomes

| Outcome Category | Unique Physical Lines | Percentage |
|---|---|---|
| **Accepted (clean)** | 39,745,054 | 98.076% |
| **Accepted (with warning)** | 314,625 | 0.776% |
| **Quarantined** | 465,207 | 1.148% |
| **Blank Lines** | 0 | 0.000% |
| **Total Physical Lines** | **40,524,886** | **100.000%** |

*Invariant Verification:* `physical_lines == accepted + accepted_with_warning + quarantined + blank` holds for all 2165 files.

---

## 2. Participant Inventory & Cohort Reconciliation

- **Total Unique Participants:** 791
- **Participants in Keystroke Logs:** 754
- **Participants in Metadata:** 646
- **Participants in Both Sources:** 609
- **Log Participants without Metadata:** 145
- **Metadata Participants without Logs:** 37
- **Eligible Cohort (Usable Logs + Valid Labels):** **597**

Among the 609 participants present in both sources, 3 have blank metadata labels: `FXD2HCNIPJ`, `PVADT62JUM`, and `QIHPY6RIK3`. The other 606 participants have valid labels (438 `True`, 168 `False`). Of those 606, 9 have no usable accepted or accepted-with-warning records (6 `True`, 3 `False`), leaving the eligible cohort of 597 participants (432 `True`, 165 `False`).

---

## 3. Anomaly & Rule Code Breakdown

*Reporting both total code occurrences and unique affected physical lines:*

| Code | Type | Total Occurrences | Unique Lines Affected | Accepted Lines | Quarantined Lines | Description |
|---|---|---|---|---|---|---|
| `E01_FIELD_COUNT` | ERROR | 0 | 0 | 0 | 0 | Number of fields differs from canonical shape |
| `E02_BAD_ENCODING` | ERROR | 0 | 0 | 0 | 0 | Source-line bytes are not valid UTF-8 |
| `E03_BAD_USERKEY` | ERROR | 387 | 387 | 0 | 387 | Key is not exactly 10 uppercase letters or digits |
| `E04_USERKEY_MISMATCH` | ERROR | 387 | 387 | 0 | 387 | Record key differs from filename key |
| `E05_BAD_DATE` | ERROR | 291 | 291 | 0 | 291 | Date is not 6 digits or not a valid calendar date |
| `E06_BAD_TIME` | ERROR | 196,341 | 196,341 | 0 | 196,341 | Timestamp is not HH:MM:SS.SSS or out of range |
| `E07_BAD_HAND` | ERROR | 1,043 | 1,043 | 0 | 1,043 | Hand is outside allowed set {L, R, S} |
| `E08_BAD_HOLD` | ERROR | 463,687 | 463,687 | 0 | 463,687 | Hold time is not a plain decimal or is negative |
| `E09_BAD_DIRECTION` | ERROR | 1,489 | 1,489 | 0 | 1,489 | Direction is outside the 9 allowed transitions |
| `E10_BAD_LATENCY` | ERROR | 464,119 | 464,119 | 0 | 464,119 | Latency is not a plain decimal or is negative |
| `E11_BAD_FLIGHT` | ERROR | 463,533 | 463,533 | 0 | 463,533 | Flight time is not a plain decimal |
| `W01_BACKWARD_TIMESTAMP` | WARNING | 44,748 | 44,748 | 44,346 | 402 | Datetime earlier than preceding parseable record |
| `W02_DUPLICATE_RECORD` | WARNING | 258,100 | 258,100 | 257,943 | 157 | Record fields identical to an earlier row in file |
| `W03_DATE_FILENAME_MISMATCH` | WARNING | 217 | 217 | 0 | 217 | Record YYMM differs from filename YYMM |
| `W04_EXTREME_HOLD` | WARNING | 13,261 | 13,261 | 13,185 | 76 | Hold time exceeds threshold (> 2000 ms) |
| `W05_SUSPECT_DATE_CODE_VALUE` | WARNING | 348 | 348 | 0 | 348 | Timing value numerically equals YYMMDD date |

---

## 4. Phase 1 Audit Reconciliation Summary

Every recoverable Phase 1 metric was audited and compared against the Phase 2 pipeline results in `audit_reconciliation.csv`:
- **Files scanned:** Exactly 2,165 logs.
- **Physical lines scanned:** Exactly 40,524,886 rows.
- **Zero blank lines** confirmed.
- **Strict tab field count != 8:** Exactly 4 rows.
- **Tab participant ID mismatches:** Exactly 387 rows.
- **Invalid dates:** Exactly 291 rows.
- **Invalid times:** Exactly 196,341 rows.
- **Invalid hands:** Exactly 1,043 rows.
- **Timing errors (comma decimals):** Exactly 464,803 rows quarantined.
- **Backward transitions:** Flagged and preserved with exact millisecond deltas in `accepted/` Parquet.
