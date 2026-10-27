from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np

from .data_loading import Session
from .features import macd


def session_feature_rows(sessions: Iterable[Session], feature_fn=macd) -> list[dict]:
    rows = []
    for session in sessions:
        value = float(feature_fn(np.asarray(session.hold_times)))
        rows.append({"subject_id": session.subject_id, "cohort": session.cohort, "session_id": session.session_id, "label": session.label, "n_keystrokes": len(session.hold_times), "macd": value})
    return rows


def subject_feature_rows(sessions: Iterable[Session], feature_fn=macd) -> list[dict]:
    grouped: dict[tuple[str, str], list[Session]] = {}
    for session in sessions:
        grouped.setdefault((session.cohort, session.subject_id), []).append(session)
    rows = []
    for (cohort, subject_id), group in sorted(grouped.items()):
        values = [float(feature_fn(np.asarray(s.hold_times))) for s in group]
        values = [v for v in values if np.isfinite(v)]
        if values:
            rows.append({"subject_id": subject_id, "cohort": cohort, "label": group[0].label, "n_sessions": len(group), "macd": float(np.mean(values))})
    return rows
