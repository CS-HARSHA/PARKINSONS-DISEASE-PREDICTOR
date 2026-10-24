"""Execution entry point, staging pipeline, and atomic publishing (Task T9)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from src.data_pipeline.cleaning import clean_log_file, FileCleaningSummary
from src.data_pipeline.config import PipelineConfig, load_config
from src.data_pipeline.participants import build_participant_datasets
from src.data_pipeline.reporting import generate_reports


def hash_file(file_path: Path | str) -> str:
    """Compute SHA-256 hash of a file reading in binary chunks."""
    h = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def hash_directory_files(base_dir: Path | str, pattern: Optional[str] = None) -> Dict[str, str]:
    """Compute SHA-256 hashes for all files in a directory, returning relative_path -> sha256."""
    dir_path = Path(base_dir)
    results: Dict[str, str] = {}
    if not dir_path.exists():
        return results

    regex = re.compile(pattern) if pattern else None
    for entry in sorted(dir_path.iterdir()):
        if entry.is_file():
            if regex and not regex.match(entry.name):
                continue
            rel = str(entry.relative_to(dir_path.parent))
            results[rel] = hash_file(entry)
    return results


def get_git_info(repo_root: Path) -> Dict[str, Any]:
    """Retrieve current Git commit and dirty status safely."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status) > 0
        return {"commit": commit, "is_dirty": is_dirty}
    except Exception:
        return {"commit": "unknown", "is_dirty": False}


def publish_staging_directory(staging_dir: Path, target_dir: Path) -> None:
    """Atomically publish staging directory to target directory using a safe swap strategy."""
    staging_path = Path(staging_dir).resolve()
    target_path = Path(target_dir).resolve()

    if not staging_path.exists():
        raise FileNotFoundError(f"Staging directory does not exist: {staging_path}")

    parent_dir = target_path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    if not target_path.exists():
        staging_path.rename(target_path)
        return

    # Target exists: use backup swap
    backup_path = parent_dir / f"{target_path.name}_backup_{uuid.uuid4().hex[:8]}"
    try:
        target_path.rename(backup_path)
        staging_path.rename(target_path)
        shutil.rmtree(backup_path, ignore_errors=True)
    except Exception as err:
        # Rollback if staging move failed
        if backup_path.exists() and not target_path.exists():
            backup_path.rename(target_path)
        raise RuntimeError(f"Atomic publish failed from {staging_path} to {target_path}: {err}") from err


def run_pipeline(
    raw_logs_dir: Optional[Path | str] = None,
    raw_metadata_dir: Optional[Path | str] = None,
    interim_dir: Optional[Path | str] = None,
    reports_dir: Optional[Path | str] = None,
    config_path: str = "configs/pipeline.toml",
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the complete Phase 2 parsing, validation, cleaning, and reporting pipeline.

    Parameters
    ----------
    raw_logs_dir : Path | str, optional
        Override path to raw keystroke logs directory.
    raw_metadata_dir : Path | str, optional
        Override path to raw participant metadata directory.
    interim_dir : Path | str, optional
        Override path to interim outputs directory.
    reports_dir : Path | str, optional
        Override path to reports directory.
    config_path : str
        Path to pipeline.toml configuration.
    max_files : int, optional
        Limit number of log files processed (for testing / verification).

    Returns
    -------
    dict
        Execution summary and exit criteria status.
    """
    config = load_config(config_path)

    logs_dir = Path(raw_logs_dir) if raw_logs_dir else config.raw_logs_dir
    meta_dir = Path(raw_metadata_dir) if raw_metadata_dir else config.raw_metadata_dir
    target_interim = Path(interim_dir) if interim_dir else config.interim_dir
    target_reports = Path(reports_dir) if reports_dir else config.reports_dir

    file_limit = max_files if max_files is not None else config.max_files

    # 1. Compute pre-run SHA-256 hashes of all raw files (Non-negotiable Rule 1)
    logs_hashes_pre = hash_directory_files(logs_dir)
    meta_hashes_pre = hash_directory_files(meta_dir)
    all_pre_hashes = {**logs_hashes_pre, **meta_hashes_pre}

    # 2. Setup staging directories on the same parent filesystem
    stage_id = uuid.uuid4().hex[:8]
    staging_interim = target_interim.parent / f"staging_interim_{stage_id}"
    staging_reports = target_reports.parent / f"staging_reports_{stage_id}"

    staging_interim.mkdir(parents=True, exist_ok=True)
    staging_reports.mkdir(parents=True, exist_ok=True)

    try:
        # 3. Discover and sort matching log files
        log_regex = re.compile(config.log_filename_pattern)
        all_log_files = sorted([
            f for f in logs_dir.iterdir()
            if f.is_file() and log_regex.match(f.name)
        ])

        if file_limit > 0:
            active_log_files = all_log_files[:file_limit]
        else:
            active_log_files = all_log_files

        file_summaries: List[FileCleaningSummary] = []
        user_log_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "n_files": 0,
            "first_yymm": "9999",
            "last_yymm": "0000",
            "n_lines": 0,
            "n_accepted": 0,
            "n_warned": 0,
            "n_quarantined": 0,
            "n_blank": 0,
        })

        for idx, log_path in enumerate(active_log_files, start=1):
            summary = clean_log_file(log_path, staging_interim, config)
            file_summaries.append(summary)

            ukey = summary.user_key
            yymm = summary.yymm
            if ukey:
                st = user_log_stats[ukey]
                st["n_files"] += 1
                if yymm:
                    if yymm < st["first_yymm"]:
                        st["first_yymm"] = yymm
                    if yymm > st["last_yymm"]:
                        st["last_yymm"] = yymm
                st["n_lines"] += summary.n_lines
                st["n_accepted"] += summary.n_accepted
                st["n_warned"] += summary.n_warned
                st["n_quarantined"] += summary.n_quarantined
                st["n_blank"] += summary.n_blank

            if idx % 500 == 0 or idx == len(active_log_files):
                print(f"Processed {idx}/{len(active_log_files)} logs...")

        # Normalize any uninitialized year-month bounds
        for st in user_log_stats.values():
            if st["first_yymm"] == "9999":
                st["first_yymm"] = ""
            if st["last_yymm"] == "0000":
                st["last_yymm"] = ""

        # 4. Build participant metadata datasets & inventory
        labels_df, covar_df, inv_df = build_participant_datasets(meta_dir, user_log_stats)

        labels_df.to_csv(staging_interim / "participant_labels.csv", index=False)
        covar_df.to_csv(staging_interim / "participant_covariates.csv", index=False)
        inv_df.to_csv(staging_interim / "participant_inventory.csv", index=False)

        # 5. Compute post-run SHA-256 hashes of all raw files & verify
        logs_hashes_post = hash_directory_files(logs_dir)
        meta_hashes_post = hash_directory_files(meta_dir)
        all_post_hashes = {**logs_hashes_post, **meta_hashes_post}

        hash_comparison_rows: List[dict] = []
        all_identical = True

        for rel_path, h_pre in all_pre_hashes.items():
            h_post = all_post_hashes.get(rel_path, "")
            is_match = (h_pre == h_post)
            if not is_match:
                all_identical = False
            hash_comparison_rows.append({
                "relative_path": rel_path,
                "sha256_before": h_pre,
                "sha256_after": h_post,
                "is_identical": is_match,
            })

        raw_hashes_df = pd.DataFrame(hash_comparison_rows).sort_values("relative_path").reset_index(drop=True)

        if not all_identical:
            raise RuntimeError("CRITICAL ERROR: Raw data modification detected! Aborting publishing.")

        # 6. Build execution manifest
        git_info = get_git_info(Path.cwd())
        config_hash = hash_file(config_path)

        manifest_data = {
            "git_commit": git_info["commit"],
            "git_dirty": git_info["is_dirty"],
            "python_version": sys.version.split()[0],
            "os_system": platform.system(),
            "package_versions": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
            },
            "config_hash_sha256": config_hash,
            "raw_hashes_verified": all_identical,
            "total_raw_files_verified": len(all_pre_hashes),
            "files_processed": len(active_log_files),
        }

        with (staging_interim / "run_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # 7. Generate all reports in staging_reports
        summary_res = generate_reports(
            file_summaries=file_summaries,
            inventory_df=inv_df,
            raw_hashes_df=raw_hashes_df,
            reports_dir=staging_reports,
            manifest_data=manifest_data,
        )

        # 8. Atomically publish staging directories to targets
        publish_staging_directory(staging_interim, target_interim)
        publish_staging_directory(staging_reports, target_reports)

        print("Phase 2 pipeline published successfully!")
        return {
            **manifest_data,
            **summary_res,
        }

    except Exception:
        # Cleanup temporary staging directories on failure
        shutil.rmtree(staging_interim, ignore_errors=True)
        shutil.rmtree(staging_reports, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 parsing, validation, and cleaning pipeline.")
    parser.add_argument("--config", default="configs/pipeline.toml", help="Path to pipeline configuration TOML")
    parser.add_argument("--raw-logs-dir", default=None, help="Path to raw keystroke logs directory")
    parser.add_argument("--raw-metadata-dir", default=None, help="Path to raw participant metadata directory")
    parser.add_argument("--interim-dir", default=None, help="Path to interim output directory")
    parser.add_argument("--reports-dir", default=None, help="Path to reports directory")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit on files processed")

    args = parser.parse_args()
    run_pipeline(
        raw_logs_dir=args.raw_logs_dir,
        raw_metadata_dir=args.raw_metadata_dir,
        interim_dir=args.interim_dir,
        reports_dir=args.reports_dir,
        config_path=args.config,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
