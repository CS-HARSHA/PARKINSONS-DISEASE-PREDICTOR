# Dataset Diagnosis — Phase 1 Sign-off

**Project:** Parkinson's Disease Prediction from Typing Characteristics

**Status:** Ready for researcher review

## 1. Audit coverage

- Monthly keystroke log files: 2,165
- Metadata files: 646
- Raw files modified: No
- Source records retain their original filenames and physical line locations in diagnostic reports.

## 2. Reconciled structural audit

- Legacy whitespace-based malformed rows: 7
- Tab-delimited records with a field count other than eight: 4
- Legacy participant-ID mismatches: 384
- Tab-based participant-ID mismatches: 387

The count discrepancy is resolved. Four records are flagged as malformed by both methods. Three further records begin with a leading tab: whitespace splitting discards the empty first field, while tab splitting preserves the empty participant-ID field. The legacy audit also skipped ID checks on malformed rows, explaining the difference between 384 and 387 mismatches.

The production parser must validate both field count and a nonempty, filename-consistent participant ID.

## 3. Other diagnostic findings

- Files containing NUL bytes: 34
- Records containing NUL bytes: 92
- NUL bytes counted: 36040
- Candidate concatenation/shift records: 2702
- Invalid dates found during the date-consistency scan: 291
- Valid timestamps checked for chronology: 40328541
- Backward timestamp transitions: 44748
- Backward transitions exceeding one second: 541

The structural diagnostic reported 483295 records with at least one flag under its current validation rules. This is a diagnostic flag count, not a count of confirmed corrupted records.

## 4. Cleaning policy

1. Preserve original files under `data/raw/`; do not edit or overwrite them.
2. Use literal tab delimiters and a documented rule for handling the expected trailing tab.
3. Require exactly eight normalized fields and a nonempty participant ID matching the filename.
4. Preserve source filename and physical line number for accepted and quarantined records.
5. Flag structurally invalid records and quarantine them with explicit reasons; do not silently repair them.
6. Treat backward timestamps as anomalies to review, not automatic reasons for record deletion.
7. Do not automatically reconstruct concatenated records without a validated, documented reconstruction rule.
8. Preserve nonpositive hold-time values pending confirmation of their meaning in the dataset.

## 5. Limitations and next phase

The diagnosis identifies risk patterns but does not prove every flagged record is unusable.
Chronology endpoint checks do not establish that timestamps are scientifically correct.
The parsing and cleaning pipeline still needs to implement and test the rules above.

## 6. Sign-off decision

**Recommended decision:** Accept Phase 1 as complete for purposes of moving into implementation, provided the researcher reviews this summary and the recorded cleaning policy.

The next phase is Phase 2: reusable parsing, validation, and cleaning tests.

**Raw-data integrity:** Original source data was not modified during this diagnosis.
