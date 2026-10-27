from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config
from .cleaning import clean_session_csv


@dataclass
class Session:
    subject_id: str
    cohort: str
    session_id: str
    label: int
    hold_times: list[float]


def extract_archive(raw_dir: Path = config.RAW_DIR) -> Path:
    candidates = sorted(raw_dir.glob("*.zip"))
    if not candidates:
        raise FileNotFoundError(f"No PhysioNet zip found in {raw_dir}")
    marker = raw_dir / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(candidates[0]) as zf:
            zf.extractall(raw_dir)
        marker.touch()
    return raw_dir


def _find_cohort_dir(raw_dir: Path, cohort: str) -> Path:
    matches = [p for p in raw_dir.rglob(cohort) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"Could not find cohort directory {cohort!r}")
    return matches[0]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _find_ground_truth(cohort_dir: Path, cohort: str) -> Path:
    matches = list(cohort_dir.rglob(f"GT_DataPD_{cohort}.csv"))
    if not matches:
        raise FileNotFoundError(f"Ground-truth CSV not found below {cohort_dir}")
    return matches[0]


def _parse_file_names(row: dict[str, str]) -> list[str]:
    names = [value.strip() for key, value in row.items() if key.lower().startswith("file") and value and value.strip()]
    if not names:
        raise ValueError(f"No file_N column found for subject row: {row}")
    return names


def load_sessions(raw_dir: Path = config.RAW_DIR) -> list[Session]:
    extract_archive(raw_dir)
    sessions: list[Session] = []
    cleaning_rows: list[dict] = []
    for cohort in config.COHORTS.values():
        cohort_dir = _find_cohort_dir(raw_dir, cohort)
        gt_path = _find_ground_truth(cohort_dir, cohort)
        data_dir = cohort_dir / f"data_{cohort}"
        if not data_dir.exists():
            data_dir = cohort_dir
        for row in _read_rows(gt_path):
            subject_id = str(row["pID"]).strip()
            label = int(str(row["gt"]).strip().lower() == "true")
            for index, file_name in enumerate(_parse_file_names(row), start=1):
                path = data_dir / Path(file_name).name
                hold_times, cleaning = clean_session_csv(path)
                cleaning_rows.append({"subject_id": subject_id, "cohort": cohort, "session_id": f"{subject_id}_{index}", **asdict(cleaning)})
                if len(hold_times) < config.MIN_SESSION_KEYSTROKES:
                    continue
                sessions.append(Session(subject_id, cohort, f"{subject_id}_{index}", label, hold_times.tolist()))
    with (config.PROCESSED_DIR / "cleaning_summary.json").open("w", encoding="utf-8") as f:
        json.dump(cleaning_rows, f, indent=2)
    return sessions


def save_sessions(sessions, output_dir: Path = config.PROCESSED_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sessions.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in sessions], f)


def load_processed(path: Path = config.PROCESSED_DIR / "sessions.json") -> list[Session]:
    with path.open(encoding="utf-8") as f:
        return [Session(**row) for row in json.load(f)]
