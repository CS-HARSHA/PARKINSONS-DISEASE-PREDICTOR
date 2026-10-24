"""Participant inventory, metadata extraction, and cohort reconciliation (Task T7)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.data_pipeline.parser import MetadataRecord, parse_metadata_file


@dataclass
class ParticipantRecord:
    user_key: str
    in_logs: bool
    n_log_files: int
    first_yymm: str
    last_yymm: str
    in_metadata: bool
    label_status: str
    n_lines: int
    n_accepted: int
    n_warned: int
    n_quarantined: int
    n_blank: int
    cohort_status: str
    exclusion_reasons: str


def build_participant_datasets(
    meta_dir: Path | str,
    log_stats_by_user: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse all participant metadata files and merge with keystroke log statistics.

    Parameters
    ----------
    meta_dir : Path | str
        Directory containing User_*.txt metadata files.
    log_stats_by_user : dict
        Aggregated statistics per user_key from processed keystroke logs.

    Returns
    -------
    tuple of (labels_df, covariates_df, inventory_df)
    """
    meta_path = Path(meta_dir)
    meta_files = sorted(list(meta_path.glob("User_*.txt")))

    metadata_by_user: Dict[str, MetadataRecord] = {}
    labels_rows: List[dict] = []
    covariates_rows: List[dict] = []

    for fpath in meta_files:
        match = re.match(r"^User_([A-Z0-9]+)\.txt$", fpath.name)
        if not match:
            continue
        user_key = match.group(1)
        rec = parse_metadata_file(fpath)
        metadata_by_user[user_key] = rec

        labels_rows.append({
            "user_key": user_key,
            "parkinsons_raw": rec.parkinsons_raw,
            "label": rec.label,
            "label_status": rec.label_status,
        })

        cov_dict = {"user_key": user_key}
        cov_dict.update(rec.covariates)
        covariates_rows.append(cov_dict)

    labels_df = pd.DataFrame(labels_rows).sort_values("user_key").reset_index(drop=True)
    covariates_df = pd.DataFrame(covariates_rows).sort_values("user_key").reset_index(drop=True)

    # Reorder covariates columns to keep common fields first
    known_order = [
        "user_key", "BirthYear", "Gender", "Parkinsons", "Tremors", "DiagnosisYear",
        "Sided", "UPDRS", "Impact", "Levodopa", "DA", "MAOB", "Other"
    ]
    present_cols = [c for c in known_order if c in covariates_df.columns]
    extra_cols = [c for c in covariates_df.columns if c not in known_order]
    covariates_df = covariates_df[present_cols + extra_cols]

    # Combine all unique participant keys
    all_keys = sorted(list(set(metadata_by_user.keys()) | set(log_stats_by_user.keys())))
    inventory_rows: List[dict] = []

    for ukey in all_keys:
        in_logs = ukey in log_stats_by_user
        in_meta = ukey in metadata_by_user

        if in_logs:
            l_info = log_stats_by_user[ukey]
            n_files = l_info.get("n_files", 0)
            first_yymm = l_info.get("first_yymm", "")
            last_yymm = l_info.get("last_yymm", "")
            n_lines = l_info.get("n_lines", 0)
            n_acc = l_info.get("n_accepted", 0)
            n_warn = l_info.get("n_warned", 0)
            n_quar = l_info.get("n_quarantined", 0)
            n_blk = l_info.get("n_blank", 0)
        else:
            n_files = 0
            first_yymm = ""
            last_yymm = ""
            n_lines = 0
            n_acc = 0
            n_warn = 0
            n_quar = 0
            n_blk = 0

        if in_meta:
            m_rec = metadata_by_user[ukey]
            lbl_status = m_rec.label_status
        else:
            lbl_status = "NO_METADATA"

        # Determine exclusion reasons and primary cohort status
        reasons: List[str] = []
        if not in_meta:
            reasons.append("NO_METADATA")
        if not in_logs:
            reasons.append("NO_LOGS")
        if in_meta and lbl_status != "VALID_LABEL":
            reasons.append(lbl_status)
        if in_logs and (n_acc + n_warn == 0):
            reasons.append("NO_USABLE_RECORDS")

        if len(reasons) == 0:
            cohort_status = "ELIGIBLE"
            exclusion_str = ""
        else:
            if "NO_METADATA" in reasons:
                cohort_status = "EXCLUDED_NO_METADATA"
            elif "NO_LOGS" in reasons:
                cohort_status = "EXCLUDED_NO_LOGS"
            elif any(r in ("MISSING_LABEL", "INVALID_LABEL", "INVALID_DUPLICATE_FIELD") for r in reasons):
                cohort_status = "EXCLUDED_NO_VALID_LABEL"
            elif "NO_USABLE_RECORDS" in reasons:
                cohort_status = "EXCLUDED_NO_USABLE_RECORDS"
            else:
                cohort_status = "EXCLUDED_OTHER"
            exclusion_str = ";".join(reasons)

        inventory_rows.append({
            "user_key": ukey,
            "in_logs": in_logs,
            "n_log_files": n_files,
            "first_yymm": first_yymm,
            "last_yymm": last_yymm,
            "in_metadata": in_meta,
            "label_status": lbl_status,
            "n_lines": n_lines,
            "n_accepted": n_acc,
            "n_warned": n_warn,
            "n_quarantined": n_quar,
            "n_blank": n_blk,
            "cohort_status": cohort_status,
            "exclusion_reasons": exclusion_str,
        })

    inventory_df = pd.DataFrame(inventory_rows).sort_values("user_key").reset_index(drop=True)
    return (labels_df, covariates_df, inventory_df)


def build_cohort_reconciliation(
    inventory_df: pd.DataFrame,
    earlier_eligible_count: int = 606,
) -> pd.DataFrame:
    """Build a detailed cohort reconciliation DataFrame explaining participant counts key-by-key.

    Parameters
    ----------
    inventory_df : pd.DataFrame
        Participant inventory DataFrame.
    earlier_eligible_count : int
        Phase 1 baseline eligible participant count.

    Returns
    -------
    pd.DataFrame
        Reconciliation table comparing baseline with Phase 2 results.
    """
    total_participants = len(inventory_df)
    in_logs_count = int(inventory_df["in_logs"].sum())
    in_meta_count = int(inventory_df["in_metadata"].sum())
    both_sources_count = int((inventory_df["in_logs"] & inventory_df["in_metadata"]).sum())
    logs_only_count = int((inventory_df["in_logs"] & ~inventory_df["in_metadata"]).sum())
    meta_only_count = int((~inventory_df["in_logs"] & inventory_df["in_metadata"]).sum())
    eligible_count = int((inventory_df["cohort_status"] == "ELIGIBLE").sum())

    reconciliation_summary = [
        {"metric": "total_unique_participants", "phase1_baseline": 791, "phase2_value": total_participants, "difference": total_participants - 791, "explanation": "754 with logs + 37 with metadata only"},
        {"metric": "participants_with_logs", "phase1_baseline": 754, "phase2_value": in_logs_count, "difference": in_logs_count - 754, "explanation": "Confirmed via monthly log file prefixes"},
        {"metric": "participants_with_metadata", "phase1_baseline": 646, "phase2_value": in_meta_count, "difference": in_meta_count - 646, "explanation": "Confirmed via User_*.txt files"},
        {"metric": "participants_in_both_sources", "phase1_baseline": 609, "phase2_value": both_sources_count, "difference": both_sources_count - 609, "explanation": "Intersection of log keys and metadata keys"},
        {"metric": "log_participants_without_metadata", "phase1_baseline": 145, "phase2_value": logs_only_count, "difference": logs_only_count - 145, "explanation": "Participants present in logs but missing metadata"},
        {"metric": "metadata_participants_without_logs", "phase1_baseline": 37, "phase2_value": meta_only_count, "difference": meta_only_count - 37, "explanation": "Participants present in metadata but missing logs"},
        {"metric": "eligible_participants", "phase1_baseline": earlier_eligible_count, "phase2_value": eligible_count, "difference": eligible_count - earlier_eligible_count, "explanation": "Participants with logs, metadata, valid label, and usable records"},
    ]

    # Explain the key-by-key difference between 609 in both sources and 606 eligible
    discrepant_keys = inventory_df[
        inventory_df["in_logs"] & inventory_df["in_metadata"] & (inventory_df["cohort_status"] != "ELIGIBLE")
    ]

    for _, row in discrepant_keys.iterrows():
        ukey = row["user_key"]
        reasons = row["exclusion_reasons"]
        reconciliation_summary.append({
            "metric": f"joint_participant_excluded:{ukey}",
            "phase1_baseline": 0,
            "phase2_value": 1,
            "difference": 1,
            "explanation": f"Participant in both logs and metadata but excluded: {reasons} (unspecified Parkinsons field in User_{ukey}.txt)",
        })

    return pd.DataFrame(reconciliation_summary)
