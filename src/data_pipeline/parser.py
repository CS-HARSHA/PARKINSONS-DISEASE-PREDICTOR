"""Byte-level streaming parser for keystroke logs and participant metadata (Task T4)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from src.data_pipeline.config import PipelineConfig

LINE_TERMINATOR_PATTERN = re.compile(rb"\r\n|\r|\n")
UTF8_BOM = b"\xef\xbb\xbf"


@dataclass(frozen=True)
class RawRecord:
    """Represents a parsed physical line from a keystroke log file."""
    source_file: str
    line_no: int
    raw_line_bytes: bytes
    terminator_bytes: bytes
    raw_line_display: str
    has_bom: bool
    is_blank: bool
    is_utf8: bool
    field_count: int
    fields: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MetadataRecord:
    """Represents a participant's parsed metadata record."""
    user_key: str
    source_file: str
    parkinsons_raw: str
    label: Optional[int]
    label_status: str
    covariates: dict[str, str] = field(default_factory=dict)


def iter_physical_lines(data: bytes) -> Iterator[Tuple[int, bytes, bytes]]:
    """Iterate through raw bytes yielding physical lines.

    Yields
    ------
    tuple
        (line_no, line_bytes_without_terminator, terminator_bytes)
    """
    pos = 0
    line_no = 1
    for match in LINE_TERMINATOR_PATTERN.finditer(data):
        start, end = match.span()
        yield (line_no, data[pos:start], data[start:end])
        line_no += 1
        pos = end

    if pos < len(data):
        yield (line_no, data[pos:], b"")


def parse_log_file(file_path: Path | str, config: PipelineConfig) -> Tuple[List[RawRecord], bool]:
    """Parse a single keystroke log file into RawRecord objects.

    Parameters
    ----------
    file_path : Path | str
        Path to the log file.
    config : PipelineConfig
        Active pipeline configuration.

    Returns
    -------
    tuple
        (records: list[RawRecord], is_empty_file: bool)
    """
    path = Path(file_path)
    with path.open("rb") as f:
        data = f.read()

    if len(data) == 0:
        return ([], True)

    records: List[RawRecord] = []
    source_file = path.name

    for line_no, line_bytes, term_bytes in iter_physical_lines(data):
        has_bom = False
        bytes_for_parsing = line_bytes

        if line_no == 1 and bytes_for_parsing.startswith(UTF8_BOM):
            has_bom = True
            bytes_for_parsing = bytes_for_parsing[len(UTF8_BOM):]

        if len(bytes_for_parsing) == 0:
            records.append(
                RawRecord(
                    source_file=source_file,
                    line_no=line_no,
                    raw_line_bytes=line_bytes,
                    terminator_bytes=term_bytes,
                    raw_line_display="",
                    has_bom=has_bom,
                    is_blank=True,
                    is_utf8=True,
                    field_count=0,
                    fields=[],
                )
            )
            continue

        try:
            line_str = bytes_for_parsing.decode("utf-8")
            is_utf8 = True
            raw_display = line_str
        except UnicodeDecodeError:
            is_utf8 = False
            raw_display = bytes_for_parsing.decode("utf-8", errors="replace")
            records.append(
                RawRecord(
                    source_file=source_file,
                    line_no=line_no,
                    raw_line_bytes=line_bytes,
                    terminator_bytes=term_bytes,
                    raw_line_display=raw_display,
                    has_bom=has_bom,
                    is_blank=False,
                    is_utf8=False,
                    field_count=0,
                    fields=[],
                )
            )
            continue

        delim = config.delimiter
        if config.has_trailing_delimiter and line_str.endswith(delim):
            data_str = line_str[:-len(delim)]
        else:
            data_str = line_str

        tokens = data_str.split(delim)

        records.append(
            RawRecord(
                source_file=source_file,
                line_no=line_no,
                raw_line_bytes=line_bytes,
                terminator_bytes=term_bytes,
                raw_line_display=raw_display,
                has_bom=has_bom,
                is_blank=False,
                is_utf8=True,
                field_count=len(tokens),
                fields=tokens,
            )
        )

    return (records, False)


def parse_metadata_file(file_path: Path | str) -> MetadataRecord:
    """Parse a single participant metadata file.

    Parameters
    ----------
    file_path : Path | str
        Path to participant metadata file (e.g. `User_0EA27ICBLF.txt`).

    Returns
    -------
    MetadataRecord
        Parsed participant metadata and label.
    """
    path = Path(file_path)
    match = re.match(r"^User_([A-Z0-9]+)\.txt$", path.name)
    user_key = match.group(1) if match else path.stem.replace("User_", "")

    with path.open("rb") as f:
        raw_bytes = f.read()

    if raw_bytes.startswith(UTF8_BOM):
        raw_bytes = raw_bytes[len(UTF8_BOM):]

    text = raw_bytes.decode("utf-8", errors="replace")

    covariates: dict[str, str] = {}
    seen_keys: set[str] = set()
    has_duplicate_key = False
    parkinsons_raw = ""

    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue

        key, val = line.split(":", 1)
        k_clean = key.strip()
        v_clean = val.strip()

        k_lower = k_clean.lower()
        if k_lower in seen_keys:
            has_duplicate_key = True
        seen_keys.add(k_lower)

        covariates[k_clean] = v_clean
        if k_lower == "parkinsons":
            parkinsons_raw = v_clean

    if has_duplicate_key:
        return MetadataRecord(
            user_key=user_key,
            source_file=path.name,
            parkinsons_raw=parkinsons_raw,
            label=None,
            label_status="INVALID_DUPLICATE_FIELD",
            covariates=covariates,
        )

    p_norm = parkinsons_raw.lower()
    if p_norm == "true":
        label = 1
        label_status = "VALID_LABEL"
    elif p_norm == "false":
        label = 0
        label_status = "VALID_LABEL"
    elif not parkinsons_raw:
        label = None
        label_status = "MISSING_LABEL"
    else:
        label = None
        label_status = "INVALID_LABEL"

    return MetadataRecord(
        user_key=user_key,
        source_file=path.name,
        parkinsons_raw=parkinsons_raw,
        label=label,
        label_status=label_status,
        covariates=covariates,
    )
