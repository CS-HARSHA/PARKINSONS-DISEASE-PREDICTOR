"""Keystroke record classification, cleaning, and Parquet output generation (Task T6)."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_pipeline.config import PipelineConfig
from src.data_pipeline.parser import RawRecord, parse_log_file
from src.data_pipeline.validation import FileValidationState, validate_record

ACCEPTED_SCHEMA = pa.schema([
    ("source_file", pa.string()),
    ("line_no", pa.int64()),
    ("user_key", pa.string()),
    ("event_time", pa.timestamp("ms")),
    ("hand", pa.string()),
    ("hold_ms", pa.float64()),
    ("direction", pa.string()),
    ("latency_ms", pa.float64()),
    ("flight_ms", pa.float64()),
    ("warning_codes", pa.string()),
])

QUARANTINE_SCHEMA = pa.schema([
    ("source_file", pa.string()),
    ("line_no", pa.int64()),
    ("raw_line_bytes", pa.binary()),
    ("raw_line_display", pa.string()),
    ("reason_codes", pa.string()),
    ("detail", pa.string()),
])


@dataclass
class FileCleaningSummary:
    """Summary of cleaning outcomes and code statistics for a single log file."""
    source_file: str
    user_key: str
    yymm: str
    n_lines: int = 0
    n_accepted: int = 0
    n_warned: int = 0
    n_quarantined: int = 0
    n_blank: int = 0
    is_empty: bool = False
    error_counts: Counter = field(default_factory=Counter)
    warning_counts_accepted: Counter = field(default_factory=Counter)
    warning_counts_quarantined: Counter = field(default_factory=Counter)
    affected_lines_per_code: Counter = field(default_factory=Counter)
    backward_deltas_ms: List[float] = field(default_factory=list)


def clean_log_file(
    file_path: Path | str,
    interim_dir: Path | str,
    config: PipelineConfig,
) -> FileCleaningSummary:
    """Clean and classify all lines in a keystroke log file, writing Parquet outputs.

    Parameters
    ----------
    file_path : Path | str
        Path to source keystroke log file.
    interim_dir : Path | str
        Target root for interim Parquet directories (accepted/ and quarantine/).
    config : PipelineConfig
        Active pipeline configuration.

    Returns
    -------
    FileCleaningSummary
        Per-file line outcome counts and anomaly code metrics.
    """
    path = Path(file_path)
    out_dir = Path(interim_dir)

    match = re.match(r"^([A-Z0-9]{10})_(\d{4})\.txt$", path.name)
    expected_user_key = match.group(1) if match else ""
    expected_yymm = match.group(2) if match else ""

    summary = FileCleaningSummary(
        source_file=path.name,
        user_key=expected_user_key,
        yymm=expected_yymm,
    )

    records, is_empty = parse_log_file(path, config)
    if is_empty:
        summary.is_empty = True
        return summary

    state = FileValidationState(
        filename=path.name,
        expected_user_key=expected_user_key,
        expected_yymm=expected_yymm,
    )

    accepted_rows: List[dict] = []
    quarantine_rows: List[dict] = []

    for raw in records:
        summary.n_lines += 1

        if raw.is_blank:
            summary.n_blank += 1
            continue

        if not raw.is_utf8:
            summary.n_quarantined += 1
            summary.error_counts["E02_BAD_ENCODING"] += 1
            summary.affected_lines_per_code["E02_BAD_ENCODING"] += 1
            quarantine_rows.append({
                "source_file": path.name,
                "line_no": raw.line_no,
                "raw_line_bytes": raw.raw_line_bytes,
                "raw_line_display": raw.raw_line_display,
                "reason_codes": "E02_BAD_ENCODING",
                "detail": "Invalid UTF-8 byte sequence",
            })
            continue

        errors, warnings, event_dt = validate_record(raw.fields, state, config)

        if errors:
            summary.n_quarantined += 1
            for err in errors:
                summary.error_counts[err] += 1
                summary.affected_lines_per_code[err] += 1
            for w in warnings:
                summary.warning_counts_quarantined[w] += 1
                summary.affected_lines_per_code[w] += 1

            quarantine_rows.append({
                "source_file": path.name,
                "line_no": raw.line_no,
                "raw_line_bytes": raw.raw_line_bytes,
                "raw_line_display": raw.raw_line_display,
                "reason_codes": ",".join(errors),
                "detail": "; ".join(errors),
            })
        else:
            # Valid record without errors
            # Non-negotiable: convert numbers with float() only after regex check passed
            hold_ms = float(raw.fields[4])
            direction = raw.fields[5]
            latency_ms = float(raw.fields[6])
            flight_ms = float(raw.fields[7])

            if warnings:
                summary.n_warned += 1
                for w in warnings:
                    summary.warning_counts_accepted[w] += 1
                    summary.affected_lines_per_code[w] += 1
                w_str = ",".join(warnings)
            else:
                summary.n_accepted += 1
                w_str = ""

            accepted_rows.append({
                "source_file": path.name,
                "line_no": raw.line_no,
                "user_key": raw.fields[0],
                "event_time": event_dt,
                "hand": raw.fields[3],
                "hold_ms": hold_ms,
                "direction": direction,
                "latency_ms": latency_ms,
                "flight_ms": flight_ms,
                "warning_codes": w_str,
            })

    # Invariant enforcement: physical_lines = accepted + accepted_with_warning + quarantined + blank
    assert summary.n_lines == (
        summary.n_accepted + summary.n_warned + summary.n_quarantined + summary.n_blank
    ), f"Invariant violated for file {path.name}: {summary}"

    summary.backward_deltas_ms = state.backward_deltas_ms

    # Write accepted parquet
    stem = path.stem
    accepted_dir = out_dir / "accepted"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    if accepted_rows:
        table_acc = pa.Table.from_pylist(accepted_rows, schema=ACCEPTED_SCHEMA)
        pq.write_table(table_acc, accepted_dir / f"{stem}.parquet")

    # Write quarantine parquet (only for files with quarantined records)
    if quarantine_rows:
        quarantine_dir = out_dir / "quarantine"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        table_quar = pa.Table.from_pylist(quarantine_rows, schema=QUARANTINE_SCHEMA)
        pq.write_table(table_quar, quarantine_dir / f"{stem}.parquet")

    return summary
