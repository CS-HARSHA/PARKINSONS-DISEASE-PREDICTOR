# Exploratory data analysis

The processed dataset contains **85 subjects** and **116 valid sessions**.

## Why MACD is a reasonable feature

The scientific motivation is dynamic rather than purely distributional: Parkinsonian motor impairment may appear as less consistent changes from one key hold time to the next. Therefore, this EDA compares the absolute consecutive change in hold times between controls and PD subjects, then compares the corridor-restricted MACD feature used by the paper.

All group summaries below are calculated at the subject level, so a participant with many keystrokes does not count as thousands of independent observations.

## Cohort summary

| Cohort | Subjects | PD | Controls | Sessions | Keystrokes min / median / max |
|---|---:|---:|---:|---:|---:|
| MIT-CS1PD | 31 | 18 | 13 | 62 | 443 / 1188 / 3322 |
| MIT-CS2PD | 54 | 24 | 30 | 54 | 299 / 1503 / 3181 |

## Subject-level group comparison

| Group | Subjects | Mean hold | Hold-time SD | Mean absolute consecutive change | Mean MACD | Median MACD |
|---|---:|---:|---:|---:|---:|---:|
| MIT-CS1PD_Control | 13 | 0.1065 | 0.0298 | 0.0313 | 0.0242 | 0.0222 |
| MIT-CS1PD_PD | 18 | 0.1427 | 0.0529 | 0.0511 | 0.0469 | 0.0390 |
| MIT-CS2PD_Control | 30 | 0.1049 | 0.0294 | 0.0266 | 0.0221 | 0.0215 |
| MIT-CS2PD_PD | 24 | 0.1299 | 0.0436 | 0.0429 | 0.0368 | 0.0337 |

## Figures

- `hold_time_by_label.png`: subject-level average hold time.
- `consecutive_change_by_label.png`: subject-level average absolute change between consecutive hold times.
- `macd_by_label.png`: subject-level corridor-restricted MACD.

These plots support the feature motivation if the PD and control distributions differ, but they are descriptive only. They do not prove causality and they are not used to choose a test-set threshold or claim clinical validity.

Session-level MACD ranges from **0.008651** to **0.109542**, with median **0.028386**.
