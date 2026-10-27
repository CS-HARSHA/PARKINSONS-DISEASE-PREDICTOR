from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data_loading import Session
from .features import macd, stdev


def _subject_groups(sessions: list[Session]) -> dict[tuple[str, str, int], list[Session]]:
    groups: dict[tuple[str, str, int], list[Session]] = defaultdict(list)
    for session in sessions:
        groups[(session.cohort, session.subject_id, session.label)].append(session)
    return groups


def subject_summary_rows(sessions: list[Session]) -> list[dict]:
    rows = []
    for (cohort, subject_id, label), group in sorted(_subject_groups(sessions).items()):
        holds = np.concatenate([np.asarray(s.hold_times, dtype=float) for s in group])
        session_consecutive = [np.abs(np.diff(np.asarray(s.hold_times, dtype=float))) for s in group]
        consecutive = np.concatenate([values for values in session_consecutive if len(values)]) if any(len(values) for values in session_consecutive) else np.asarray([], dtype=float)
        session_macds = [macd(np.asarray(s.hold_times, dtype=float)) for s in group]
        rows.append({
            "cohort": cohort,
            "subject_id": subject_id,
            "label": label,
            "label_name": "PD" if label else "Control",
            "mean_hold": float(np.mean(holds)),
            "median_hold": float(np.median(holds)),
            "stdev_hold": float(np.std(holds)),
            "mean_abs_consecutive_change": float(np.mean(consecutive)) if len(consecutive) else float("nan"),
            "macd": float(np.nanmean(session_macds)),
            "n_keystrokes": int(len(holds)),
        })
    return rows


def summarize_sessions(sessions: list[Session]) -> dict:
    by_cohort = {}
    for cohort in sorted({s.cohort for s in sessions}):
        rows = [s for s in sessions if s.cohort == cohort]
        by_cohort[cohort] = {
            "sessions": len(rows),
            "subjects": len({s.subject_id for s in rows}),
            "pd_subjects": len({s.subject_id for s in rows if s.label == 1}),
            "control_subjects": len({s.subject_id for s in rows if s.label == 0}),
            "keystrokes_min": int(min(len(s.hold_times) for s in rows)),
            "keystrokes_median": float(np.median([len(s.hold_times) for s in rows])),
            "keystrokes_max": int(max(len(s.hold_times) for s in rows)),
        }
    subject_rows = subject_summary_rows(sessions)
    group_summary = {}
    for cohort in sorted({r["cohort"] for r in subject_rows}):
        for label, label_name in [(0, "Control"), (1, "PD")]:
            rows = [r for r in subject_rows if r["cohort"] == cohort and r["label"] == label]
            if not rows:
                continue
            group_summary[f"{cohort}_{label_name}"] = {
                "n_subjects": len(rows),
                "mean_hold_subject_mean": float(np.mean([r["mean_hold"] for r in rows])),
                "mean_hold_subject_median": float(np.median([r["mean_hold"] for r in rows])),
                "mean_stdev_hold": float(np.mean([r["stdev_hold"] for r in rows])),
                "mean_abs_consecutive_change": float(np.mean([r["mean_abs_consecutive_change"] for r in rows])),
                "mean_macd": float(np.mean([r["macd"] for r in rows])),
                "median_macd": float(np.median([r["macd"] for r in rows])),
            }
    session_macd = [macd(np.asarray(s.hold_times)) for s in sessions]
    return {
        "total_sessions": len(sessions),
        "total_subjects": len({s.subject_id for s in sessions}),
        "by_cohort": by_cohort,
        "macd_min": float(np.nanmin(session_macd)),
        "macd_median": float(np.nanmedian(session_macd)),
        "macd_max": float(np.nanmax(session_macd)),
        "subject_level_group_summary": group_summary,
    }


def write_subject_summary(sessions: list[Session], path: Path) -> None:
    write_csv(subject_summary_rows(sessions), path)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
