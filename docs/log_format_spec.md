# Keystroke Log & Participant Metadata Format Specification

**Project:** Parkinson’s Disease Predictor  
**Repository:** `CS-HARSHA/PARKINSONS-DISEASE-PREDICTOR`  
**Date:** 24 October 2026  
**Status:** Empirically verified across all 2,165 keystroke log files and 646 participant metadata files.

---

## 1. Executive Summary & Format Discrepancy Resolution

The public Mendeley Data V3 description states that log files are comma-separated. **Empirical profiling of all 2,165 local files confirms this description is inaccurate for this repository copy.**

- **Actual Delimiter:** Tab character (`\t`, ASCII `0x09`). Exactly 40,524,886 of 40,524,886 lines (100.0%) use tab delimiters. Exactly 0 lines use commas as delimiters.
- **Trailing Delimiter:** Every nonblank line has a trailing tab (`\t`) immediately preceding the line terminator `\r\n`.
- **Line Endings:** Every non-empty log file (2,160 files) uses DOS/Windows `CRLF` (`\r\n`, `0x0D 0x0A`). No mixed line endings, lone `\r`, or lone `\n` were detected.
- **Empty Files:** Exactly 5 files have 0 bytes:
  1. `1SFTNRT0XB_1703.txt`
  2. `DRQKANH3Q0_1709.txt`
  3. `JORVLMISQA_1709.txt`
  4. `UURLOSDUOA_1605.txt`
  5. `ZI1KGKLCD5_1612.txt`
- **Non-log Files:** Exactly 1 non-log archive is present in `data/raw/keystroke_logs/`: `Archived Users.zip`. It must not be parsed as a log file.
- **Encoding & BOM:** All files are valid UTF-8. Exactly 0 log files contain a byte-order mark (BOM `\xef\xbb\xbf`).
- **NUL Bytes:** 34 log files contain NUL bytes (`\x00`), totaling 36,040 NUL bytes across 92 physical lines.

---

## 2. Canonical Log Record Specification

### 2.1 Modal Record Shape

A physical line consists of 8 data fields followed by a single trailing tab delimiter and CRLF:

```text
UserKey \t Date \t Timestamp \t Hand \t HoldTime \t Direction \t Latency \t Flight \t \r\n
```

When split strictly by `\t`, this produces 9 tokens where token index 8 is empty (`""`). Stripping the trailing delimiter produces exactly 8 data fields:

| Field Index | Field Name | Type / Format | Example | Description |
|---|---|---|---|---|
| 0 | `UserKey` | `^[A-Z0-9]{10}$` | `0EA27ICBLF` | 10-character alphanumeric participant ID. Must match the filename ID. |
| 1 | `Date` | `^\d{6}$` (`YYMMDD`) | `160723` | Session date. `YY` in `['16', '17']` maps to `2016` or `2017`. |
| 2 | `Timestamp` | `^\d{2}:\d{2}:\d{2}\.\d{3}$` | `17:35:20.516` | Wall-clock time (`HH:MM:SS.SSS`). |
| 3 | `Hand` | `^[LRS]$` | `L` | Keypress hand: `L` (Left), `R` (Right), `S` (Space). |
| 4 | `HoldTime` | Plain decimal (ms) | `0199.2` | Duration key was held down. Must be non-negative. |
| 5 | `Direction` | 2-char string | `LL` | Transition direction between successive keys. |
| 6 | `Latency` | Plain decimal (ms) | `0453.1` | Time between consecutive key down events. Must be non-negative. |
| 7 | `Flight` | Plain decimal (ms) | `0281.3` | Time between key release and next key press. Negative values are valid. |

### 2.2 Field-Count Histogram

Across all 40,524,886 physical lines:
- **Strict tab split (`line.split('\t')`):**
  - 9 tokens (8 data fields + trailing empty string): 40,524,882 (99.99999%)
  - 8 tokens (missing trailing tab / 7 data fields): 4 lines
- **After removing trailing delimiter:**
  - 8 data fields: 40,524,882 (99.99999%)
  - 7 data fields: 4 lines

The 8-data-field modal record shape covers **>99.999%** of nonblank lines, well exceeding the 99% stop-condition threshold.

---

## 3. Allowed Value Sets

### 3.1 Hand Field (`Field 3`)
- **Allowed Set:** `{'L', 'R', 'S'}`
- **Observed Distribution:**
  - `L`: 19,050,797 (47.01%)
  - `R`: 14,607,954 (36.05%)
  - `S`: 6,865,092 (16.94%)
  - Corrupt / shifted strings (dates, IDs, times): 1,043 occurrences.
- Any value outside `{'L', 'R', 'S'}` receives error code `E07_BAD_HAND`.

### 3.2 Direction Field (`Field 5`)
- **Allowed Set:** All 9 combinations of `L`, `R`, and `S`:
  `{'LL', 'LR', 'RR', 'RL', 'LS', 'RS', 'SR', 'SL', 'SS'}`
- **Observed Distribution:**
  - `LL`: 8,441,255
  - `RL`: 7,182,671
  - `LR`: 6,878,696
  - `RR`: 5,682,703
  - `LS`: 4,061,714
  - `SL`: 3,426,631
  - `RS`: 2,466,417
  - `SR`: 2,046,391
  - `SS`: 336,919
  - Corrupt / shifted values (e.g. `160725`, `L`, `0156.3`): 1,489 occurrences.
- Any value outside this 9-element set receives error code `E09_BAD_DIRECTION`.

### 3.3 Numeric Fields & Plain Decimal Contract
- `HoldTime`, `Latency`, and `Flight` must match the plain decimal pattern:
  `^-?\d+(\.\d+)?$`
- **HoldTime:** Must be non-negative (`>= 0.0`). Zero hold time (`0000.0`) is non-negative and valid. Negative values produce `E08_BAD_HOLD`.
- **Latency:** Must be non-negative (`>= 0.0`). Negative values produce `E10_BAD_LATENCY`.
- **Flight:** May be negative (`flight < 0.0` is valid in physical typing overlaps). Non-decimal values produce `E11_BAD_FLIGHT`.
- **Comma decimal separator finding:** Phase 1 audit identified 464,803 lines containing locale-formatted comma decimal separators (e.g. `0066,4`). Under the strict plain decimal contract, comma-separated decimals fail the plain decimal regex and are quarantined under `E08`, `E10`, and `E11`.

### 3.4 Two-Digit Year Interpretation
- All filenames and dates in logs have `YY` in `['16', '17']`.
- Calendar dates are mapped unambiguously via `2000 + YY` (`16` $\to$ 2016, `17` $\to$ 2017).
- Dates must be strictly validated as real Gregorian calendar dates (e.g. month 1–12, valid days per month, leap year rules).

---

## 4. Participant Metadata Specification

Across all 646 files in `data/raw/participant_metadata/User_*.txt`:
- **Format:** Plain text, UTF-8 encoded, CRLF line endings.
- **Fields:** Exactly 12 lines per file, structured as `Key: Value`.
- **Field Inventory:**
  1. `BirthYear`
  2. `Gender`
  3. `Parkinsons`
  4. `Tremors`
  5. `DiagnosisYear`
  6. `Sided`
  7. `UPDRS`
  8. `Impact`
  9. `Levodopa`
  10. `DA`
  11. `MAOB`
  12. `Other`
- **Duplicate Fields:** Exactly 0 files contain duplicate keys.
- **Label Mapping:**
  - `Parkinsons: True` (case-insensitive) $\to$ `label = 1` (`VALID_LABEL`)
  - `Parkinsons: False` (case-insensitive) $\to$ `label = 0` (`VALID_LABEL`)
  - `Parkinsons: ` (empty/missing) $\to$ `label = None` (`MISSING_LABEL`)
  - Any other value $\to$ `INVALID_LABEL`
- **Label Distribution:**
  - Total metadata files: 646 (455 `True`, 188 `False`, 3 blank)
  - Joint with logs: 609 participants (438 `True`, 168 `False`, 3 blank)
  - **Joint Keys with Valid Diagnosis Labels:** 606 participants present in both logs and metadata with valid labels ($438 + 168 = 606$).
  - **Eligible Cohort:** 597 participants with valid labels and at least one usable accepted or accepted-with-warning record ($432 + 165 = 597$).
  - **No Usable Records:** 9 otherwise validly labeled joint participants were excluded (6 `True`, 3 `False`).
  - The 3 participants in both sources with missing labels are:
    `FXD2HCNIPJ`, `PVADT62JUM`, `QIHPY6RIK3`.

---

## 5. Ten Representative Anomaly Examples

The following 10 real records from `data/raw/keystroke_logs/` illustrate the corruption patterns and corresponding error codes:

| No. | Source File | Line | Raw Record Content (Repr) | Failure Reason & Code |
|---|---|---|---|---|
| 1 | `9JN39MOLWT_1709.txt` | 4426 | `'9JN39MOLWT\t170928\t20:25:18.828\tR\t0070.3\tRR\t0179.7\t\t'` | Missing flight time field (7 tab tokens). `E01_FIELD_COUNT` |
| 2 | `2JTCBKUP8T_1705.txt` | 28871 | `'\t170528\t20:54:30.523\tL\t0078.1\tLL\t0312.5\t0218.8\t'` | Leading tab creates empty UserKey `""`. `E03_BAD_USERKEY`, `E04_USERKEY_MISMATCH` |
| 3 | `0EA27ICBLF_1607.txt` | 2651 | `'0EA27ICB0EA27ICBLF\t160723\t17:35:20.516\tL\t0199.2\tLL\t0453.1\t0281.3\t'` | Duplicated/concatenated UserKey (18 chars). `E03_BAD_USERKEY`, `E04_USERKEY_MISMATCH` |
| 4 | `0EA27ICBLF_1607.txt` | 14482 | `'0EA27ICBLF\t160729\t22:17:34.477\tL\t0093.8\tSL\t160725\t23:51:23.453\t'` | Shifted timestamp in flight field. `E11_BAD_FLIGHT` |
| 5 | `0EA27ICBLF_1607.txt` | 35769 | `'0EA27ICBLF\t160731\t13:25:17.6680EA27ICBLF\t160725\t23:51:23.453\tL\tLL\t11\t'` | Spliced timestamp and key, shifted fields. `E06_BAD_TIME`, `E07_BAD_HAND`, `E08_BAD_HOLD` |
| 6 | `0EA27ICBLF_1608.txt` | 7707 | `'0EA27ICBLF\t160813\t01:36:0EA27ICBLF\t160809\t22:50:04.422\tS\tRS\tAS\t'` | Mid-record concatenation; invalid time, hand, timing. `E06_BAD_TIME`, `E07_BAD_HAND`, `E10_BAD_LATENCY` |
| 7 | `0EA27ICBLF_1608.txt` | 9253 | `'0EA27ICBLF\t160810EA27ICBLF\t160819\t23:17:06.555\tL\t0156.3\t51\t0453.1\t'` | Embedded key in date field; direction has digits `51`. `E05_BAD_DATE`, `E09_BAD_DIRECTION` |
| 8 | `OPACZPPV1Q_1711.txt` | 1 | `'OPACZPPV1Q\t171101\t09:13:52.629\tR\t0066,4\tRR\t0496,1\t0418,0\t'` | Locale comma decimal separator. `E08_BAD_HOLD`, `E10_BAD_LATENCY`, `E11_BAD_FLIGHT` |
| 9 | `2JTCBKUP8T_1612.txt` | 42 | `'2JTCBKUP8T\t161201\t11:03:00.000\t\x00\x00\x00\x00\x00\x00\x00\x00...\t'` | Embedded NUL bytes replacing hand and timing. `E07_BAD_HAND`, `E08_BAD_HOLD` |
| 10 | `0EA27ICBLF_1607.txt` | 94426 | `'00EA27ICBLF\t160725\t23:51:23.453\tL\t0101.6\tLL\t0320.3\t0171.9\t'` | Leading zero prefixed to UserKey (11 chars). `E03_BAD_USERKEY`, `E04_USERKEY_MISMATCH` |

---

## 6. Physical Line & Byte-Level Streaming Contract

1. Files must be read in binary mode (`"rb"`).
2. Physical lines are separated by `\r\n` (CRLF). Unterminated trailing records must be handled without producing phantom rows.
3. Physical line numbers are 1-based and strictly sequential.
4. Each physical line must produce exactly one mutually exclusive outcome:
   `accepted`, `accepted_with_warning`, `quarantined`, or `blank`.
5. For quarantined records, `raw_line_bytes` preserves the exact original byte sequence without loss.
