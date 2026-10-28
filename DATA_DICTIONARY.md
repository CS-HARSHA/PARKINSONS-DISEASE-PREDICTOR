# Data dictionary

## Processed session table

- `subject_id`: PhysioNet patient identifier; keep private in any public demo.
- `cohort`: `MIT-CS1PD` or `MIT-CS2PD`.
- `session_id`: Stable local identifier for one session.
- `label`: `1` for PD, `0` for control.
- `hold_times`: Ordered cleaned hold durations in seconds; no text content is stored.

## Feature tables

- `n_keystrokes`: Number of cleaned hold-time observations in a session.
- `n_sessions`: Number of sessions aggregated for a subject.
- `macd`: Mean absolute consecutive difference in the 60th–100th percentile corridor.
