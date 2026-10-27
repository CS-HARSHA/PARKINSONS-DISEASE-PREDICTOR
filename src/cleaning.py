from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CleaningSummary:
    raw_rows: int
    parsed_rows: int
    removed_invalid: int
    removed_time_reversal: int
    removed_mouse: int
    removed_long_modifier: int
    removed_backspace: int
    final_rows: int


def clean_session_csv(path: Path) -> tuple[np.ndarray, CleaningSummary]:
    """Load one PhysioNet CSV and return paper-compatible cleaned hold times."""
    raw_rows = 0
    rows: list[tuple[str, float, float, float]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            raw_rows += 1
            if len(row) < 4:
                continue
            try:
                rows.append((row[0], float(row[1]), float(row[3]), float(row[2])))
            except (TypeError, ValueError):
                continue
    if not rows:
        return np.asarray([], dtype=float), CleaningSummary(raw_rows, 0, raw_rows, 0, 0, 0, 0, 0)
    keys = np.asarray([r[0] for r in rows], dtype=str)
    hold = np.asarray([r[1] for r in rows], dtype=float)
    press = np.asarray([r[2] for r in rows], dtype=float)
    release = np.asarray([r[3] for r in rows], dtype=float)
    valid = np.isfinite(hold) & np.isfinite(press) & np.isfinite(release)
    valid &= (press > 0) & (release > 0) & (hold >= 0) & (hold < 5)
    removed_invalid = int((~valid).sum())
    reversal = np.zeros(len(press), dtype=bool)
    for i in range(1, len(press)):
        if press[i] < press[i - 1]:
            reversal[i] = True
    valid &= ~reversal
    valid_keys = keys[valid]
    excluded_mouse = np.array([re.search(r"mouse", k, re.IGNORECASE) is not None for k in valid_keys])
    excluded_modifier = np.array([re.match(r"^(Shift|Alt|Control)", k, re.IGNORECASE) is not None for k in valid_keys])
    excluded_backspace = np.array([re.match(r"^BackSpace$", k, re.IGNORECASE) is not None for k in valid_keys])
    keep_key = ~(excluded_mouse | excluded_modifier | excluded_backspace)
    cleaned = hold[valid][keep_key]
    return cleaned, CleaningSummary(raw_rows, len(rows), removed_invalid, int(reversal.sum()), int(excluded_mouse.sum()), int(excluded_modifier.sum()), int(excluded_backspace.sum()), len(cleaned))
