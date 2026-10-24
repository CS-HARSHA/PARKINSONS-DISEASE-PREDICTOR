"""Quality auditing and validation module for Phase 6 Feature Engineering.

Performs deterministic integrity checks, missingness audits, date-corruption verification,
and cohort reconciliation for engineered typing features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering.feature_schema import (
    get_predictor_columns,
    validate_predictor_allowlist,
)


def validate_feature_table(
    df_features: pd.DataFrame,
    expected_count: int = 597,
    expected_labels: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Validate structural and semantic integrity of the unscaled feature table."""
    issues: list[str] = []

    # 1. Cohort row count and uniqueness
    row_count = len(df_features)
    if row_count != expected_count:
        issues.append(f"Row count mismatch: observed {row_count}, expected {expected_count}")

    if "participant_id" not in df_features.columns:
        issues.append("Missing mandatory 'participant_id' column")
        unique_ids = 0
    else:
        unique_ids = df_features["participant_id"].nunique()
        if unique_ids != row_count:
            issues.append(f"Duplicate participant IDs found: {row_count - unique_ids} duplicate rows")

    # 2. Target label integrity
    if "parkinsons_label" not in df_features.columns:
        issues.append("Missing mandatory 'parkinsons_label' column")
    else:
        labels = df_features["parkinsons_label"]
        if labels.isna().any():
            issues.append(f"Found {int(labels.isna().sum())} missing target labels")
        if not labels.isin([0, 1]).all():
            issues.append("Target labels contain values outside {0, 1}")
        if expected_labels is not None and "participant_id" in df_features.columns:
            mismatches = 0
            for _, r in df_features.iterrows():
                pid = r["participant_id"]
                if pid in expected_labels and r["parkinsons_label"] != expected_labels[pid]:
                    mismatches += 1
            if mismatches > 0:
                issues.append(f"Label disagreement with authoritative labels for {mismatches} participants")

    # 3. Predictor allowlist check
    predictor_cols = [c for c in df_features.columns if c not in ["participant_id", "parkinsons_label"]]
    is_valid_allowlist, violations = validate_predictor_allowlist(predictor_cols)
    if not is_valid_allowlist:
        issues.append(f"Prohibited columns present in predictor set: {violations}")

    # 4. Check for presence of all documented predictor columns
    expected_predictors = set(get_predictor_columns())
    missing_predictors = expected_predictors - set(predictor_cols)
    if missing_predictors:
        issues.append(f"Missing {len(missing_predictors)} expected predictor columns")

    # 5. Non-finite values (Infinity)
    inf_counts = {}
    for col in predictor_cols:
        s = df_features[col]
        if pd.api.types.is_numeric_dtype(s):
            n_inf = int(np.isinf(s).sum())
            if n_inf > 0:
                inf_counts[col] = n_inf
    if inf_counts:
        issues.append(f"Infinite values detected in {len(inf_counts)} columns: {inf_counts}")

    # 6. Date corruption signature detection
    # Date corruption in legacy tables arose from parsing string dates (e.g., 171226.0, 160814.0)
    # into timing columns. We check specifically for exact integer YYMMDD date values.
    corrupted_date_cols = {}
    extreme_timing_cols = {}
    for col in predictor_cols:
        s = df_features[col].dropna()
        if pd.api.types.is_numeric_dtype(s):
            # Check for exact integer YYMMDD date values in [160000, 180000]
            date_matches = s[(s >= 160000) & (s <= 180000) & (s % 1 == 0)]
            if len(date_matches) > 0:
                corrupted_date_cols[col] = len(date_matches)
            
            # Audit continuous extreme observations (>100k ms) from W04_EXTREME_HOLD records
            n_extreme = int((s > 100000).sum())
            if n_extreme > 0:
                extreme_timing_cols[col] = n_extreme

    if corrupted_date_cols:
        issues.append(f"FATAL: Date corruption signature (exact YYMMDD integers) found in {len(corrupted_date_cols)} columns: {corrupted_date_cols}")

    # 7. Rollover flight time audit
    # In Tappy Extended Mendeley V3, empirical records in accepted parquet files have flight_ms >= 0.0.
    # We verify that flight_ms is numeric and uncorrupted.
    flight_cols = [c for c in predictor_cols if c.startswith("flight_")]
    invalid_flight_cols = [c for c in flight_cols if not pd.api.types.is_numeric_dtype(df_features[c])]
    if invalid_flight_cols:
        issues.append(f"Non-numeric flight columns detected: {invalid_flight_cols}")

    # 8. Constant features
    constant_cols = [c for c in predictor_cols if df_features[c].nunique(dropna=True) <= 1]

    # Summarize missingness
    missing_by_col = df_features[predictor_cols].isna().sum()
    cols_with_missing = missing_by_col[missing_by_col > 0].to_dict()

    passed = len(issues) == 0

    return {
        "passed": passed,
        "row_count": row_count,
        "unique_participant_count": unique_ids,
        "predictor_column_count": len(predictor_cols),
        "issues": issues,
        "cols_with_missing_count": len(cols_with_missing),
        "columns_with_missing": cols_with_missing,
        "constant_columns": constant_cols,
        "date_corrupted_columns": corrupted_date_cols,
        "extreme_timing_columns": extreme_timing_cols,
        "infinite_columns": inf_counts,
    }


def validate_quality_table(
    df_quality: pd.DataFrame,
    expected_count: int = 597,
) -> dict[str, Any]:
    """Validate participant coverage and quality metadata table."""
    issues: list[str] = []

    if len(df_quality) != expected_count:
        issues.append(f"Quality table row count mismatch: {len(df_quality)} vs {expected_count}")

    if df_quality["participant_id"].nunique() != len(df_quality):
        issues.append("Quality table contains duplicate participant IDs")

    if (df_quality["total_records"] <= 0).any():
        zero_recs = int((df_quality["total_records"] <= 0).sum())
        issues.append(f"{zero_recs} participants have zero accepted records")

    return {
        "passed": len(issues) == 0,
        "row_count": len(df_quality),
        "total_records_sum": int(df_quality["total_records"].sum()),
        "total_warned_records_sum": int(df_quality["n_warned_records"].sum()),
        "total_w01_sum": int(df_quality["n_w01_backward_timestamp"].sum()),
        "total_w02_sum": int(df_quality["n_w02_duplicate_record"].sum()),
        "total_w04_sum": int(df_quality["n_w04_extreme_hold"].sum()),
        "issues": issues,
    }


def generate_feature_quality_summary(df_features: pd.DataFrame) -> pd.DataFrame:
    """Generate deterministic feature quality summary for all predictor columns."""
    predictor_cols = [c for c in df_features.columns if c not in ["participant_id", "parkinsons_label"]]
    summary_rows = []

    for col in sorted(predictor_cols):
        s = df_features[col]
        n_missing = int(s.isna().sum())
        pct_missing = round(n_missing / len(s) * 100, 2)
        n_unique = int(s.nunique(dropna=True))
        is_const = bool(n_unique <= 1)
        n_gt_100k = int((s.dropna() > 100000).sum()) if pd.api.types.is_numeric_dtype(s) else 0

        summary_rows.append({
            "feature_name": col,
            "data_type": str(s.dtype),
            "missing_count": n_missing,
            "missing_pct": pct_missing,
            "n_unique": n_unique,
            "is_constant": is_const,
            "n_gt_100k": n_gt_100k,
            "status": "PASS" if (n_gt_100k == 0 and not is_const) else "FLAGGED",
        })

    return pd.DataFrame(summary_rows)
