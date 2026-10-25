"""Main execution script for Phase 6 Feature Engineering.

Runs feature extraction across all 597 eligible cohort participants,
validates feature integrity, exports parquet data tables and reports,
and generates the authoritative feature manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.feature_engineering.config import FeatureEngineeringConfig
from src.feature_engineering.feature_extraction import aggregate_cohort_features
from src.feature_engineering.feature_schema import get_feature_definitions, get_predictor_columns
from src.feature_engineering.quality import (
    generate_feature_quality_summary,
    validate_feature_table,
    validate_quality_table,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase6_runner")


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_feature_dictionary_markdown(output_path: Path):
    """Generate docs/feature_dictionary.md documenting all 153 features."""
    defs = get_feature_definitions()
    
    lines = [
        "# Typing Feature Dictionary",
        "",
        "**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  ",
        "**Primary Dataset:** Tappy Extended Keystroke Dataset, Mendeley Data Version 3 (DOI: `10.17632/z39mhdsynx.3`)  ",
        "**Phase:** Phase 6 — Feature Engineering & Preprocessing  ",
        f"**Total Features Cataloged:** {len(defs)}  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        "",
        "---",
        "",
        "## 1. Feature Taxonomy Overview",
        "",
        "Typing features are derived deterministically from validated event-level keystroke records under the Phase 2 specification:",
        "- **Hold Time (HT):** Duration of individual keypress ($T_{release} - T_{press}$) in milliseconds.",
        "- **Latency (L):** Interval between consecutive key-press events ($T_{press, n+1} - T_{press, n}$) in milliseconds.",
        "- **Flight Time (FT):** Interval between key-release and consecutive key-press ($T_{press, n+1}- T_{release, n}$) in milliseconds. The Phase 2 validator permits negative values; the accepted-cohort audit observed none, so rollover preservation was not empirically demonstrated.",
        "- **Hand Dynamics:** Relative proportions of keys struck by left hand (`L`), right hand (`R`), or spacebar (`S`).",
        "- **Transition Dynamics:** Relative proportions and directional timing statistics across 9 transitions: `LL`, `LR`, `RL`, `RR`, `LS`, `RS`, `SL`, `SR`, `SS`.",
        "",
        "---",
        "",
        "## 2. Complete Feature Inventory",
        "",
        "| Feature Name | Family | Documented Unit | Domain Constraint | Formula / Calculation Method | Predictor Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for d in defs:
        dom_min = str(d.domain_min) if d.domain_min is not None else "Unbounded"
        dom_max = str(d.domain_max) if d.domain_max is not None else "Unbounded"
        domain = f"[{dom_min}, {dom_max}]" if (d.domain_min is not None or d.domain_max is not None) else "Real (R)"
        lines.append(f"| `{d.name}` | {d.family} | {d.unit} | {domain} | {d.description} | {'APPROVED_PREDICTOR' if d.is_predictor else 'METADATA_ONLY'} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Governance and Non-Predictor Allowlist",
        "",
        "The following columns are strictly excluded from the predictor matrix:",
        "- `participant_id`: 10-character anonymized participant identifier (traceability key only).",
        "- `parkinsons_label`: Binary ground truth target (0 = Control, 1 = Parkinson's Disease). Joined only for model evaluation within CV loops.",
        "- Clinical & Demographic covariates: `Age`, `Gender`, `UPDRS`, `Tremor`, `DiagnosisYear`, `Levodopa`, etc., are maintained in `data/interim/participant_covariates.csv` and are **prohibited** from entering the primary typing-only feature matrix.",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved feature dictionary to %s", output_path)


def generate_phase6_decisions_markdown(output_path: Path):
    """Generate docs/phase6_decisions.md recording engineering decisions."""
    lines = [
        "# Phase 6 Feature Engineering & Preprocessing Decision Log",
        "",
        "**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  ",
        "**Date:** 24 October 2026  ",
        "**Working Branch:** `praneeth`  ",
        "",
        "---",
        "",
        "## Decision D01: Re-Extraction from Clean Phase 2 Parquet Records (Case 2 Resolution)",
        "",
        "- **Context:** Phase 4 data readiness gate and Phase 5 EDA identified that the legacy `features_clean.csv` was invalid: it omitted 102 participants (495 vs 597 eligible), had 28 columns corrupted with shifted YYMMDD date values (>100,000), and clamped negative flight times to NaN.",
        "- **Decision:** Under Section 4 Case 2, re-extract participant-level features directly from the 2,143 accepted Parquet records in `data/phase2_recovery/interim/accepted/` covering the 597 eligible cohort participants.",
        "- **Impact:** Restores the 597-participant cohort and removes legacy table date-corruption artifacts. The accepted-cohort audit observed no negative raw flight_ms values, so rollover preservation cannot be empirically demonstrated in this cohort.",
        "",
        "---",
        "",
        "## Decision D02: Negative Flight-Time Validation Policy and Cohort Observation",
        "",
        "- **Context:** The Phase 2 validator permits signed plain-decimal flight_ms values. The accepted-cohort audit found zero negative values among 40,059,679 accepted flight_ms observations from 2,143 Parquet files.",
        "- **Decision:** Negative flight times remain valid under the validator and are not clamped by that rule. Because no negative event-level values were observed in the accepted cohort, this dataset cannot empirically demonstrate negative-flight preservation.",
        "- **Impact:** Keeps the processing policy compatible with negative values if they occur in another valid dataset; no negative-flight biomarker is empirically demonstrated by this cohort.",
        "",
        "---",
        "",
        "## Decision D03: Sample Size Thresholds for Higher-Order Distribution Statistics",
        "",
        "- **Context:** Sample skewness and sample kurtosis suffer severe finite-sample bias when estimated from very few keystroke events.",
        "- **Decision:** Enforce minimum sample size thresholds:",
        "  - Skewness: minimum N >= 20 valid observations; returns NaN otherwise.",
        "  - Kurtosis: minimum N >= 30 valid observations; returns NaN otherwise.",
        "  - Coefficient of Variation (CV = std / mean): requires N >= 10 and strictly positive mean (mean > 1e-6); returns NaN otherwise.",
        "- **Impact:** Prevents unstable outliers and extreme values caused by sample variance.",
        "",
        "---",
        "",
        "## Decision D04: Strict Separation of Feature Extraction from Model Preprocessing",
        "",
        "- **Context:** Imputing or scaling across the entire participant cohort prior to cross-validation causes subtle data leakage and over-optimistic performance estimates.",
        "- **Decision:** Pipeline A (Feature Extraction) produces unscaled, un-imputed participant features. Pipeline B (Preprocessing) packages reusable, **unfitted** scikit-learn transformers (`SimpleImputer`, `StandardScaler`, `RobustScaler`) that are fitted strictly inside cross-validation folds during Phase 7.",
        "- **Impact:** Eliminates data leakage and guarantees strict methodological compliance with the Phase 3 protocol.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved Phase 6 decisions to %s", output_path)


def generate_preprocessing_decisions_markdown(output_path: Path):
    """Generate reports/feature_engineering/preprocessing_decisions.md."""
    lines = [
        "# Preprocessing Pipeline Design & Component Decisions",
        "",
        "**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  ",
        "**Phase:** Phase 6 — Feature Engineering and Preprocessing  ",
        "**Date:** 24 October 2026  ",
        "",
        "---",
        "",
        "## 1. Summary of Implemented Preprocessing Transformers",
        "",
        "In accordance with Section 9 and Section 10 of the Phase 6 specification, preprocessing components are modularized into reusable scikit-learn transformers via `src.feature_engineering.preprocessing.build_preprocessing_pipeline`.",
        "",
        "| Component | Default Strategy | Supported Options | Downstream Model Alignment | Rationale |",
        "| :--- | :--- | :--- | :--- | :--- |",
        "| **Missing Value Imputer** | `SimpleImputer(strategy='median')` | `'median'`, `'mean'`, optional `add_indicator=True` | All models | Robust to right-skewed timing distributions; preserves feature medians without distributional shift. |",
        "| **Standard Scaler** | `StandardScaler()` | `StandardScaler()`, `RobustScaler()`, `None` | Logistic Regression, SVM, KNN | Normalizes feature scales (mu=0, sigma=1) required for distance and gradient-based estimators. |",
        "| **Robust Scaler** | Configurable alternative | `RobustScaler()` | Linear models under heavy tails | Centers by median and scales by IQR; immune to extreme typing outliers. |",
        "| **Passthrough Scaler** | `'none'` | No scaling | Random Forest, ExtraTrees, XGBoost | Tree-based models are invariant to monotonic scaling; unscaled features preserve clinical interpretability. |",
        "| **Power Transformer** | `PowerTransformer(method='yeo-johnson')` | Optional flag | Gaussian-assuming linear models | Symmetrizes moderately skewed timing distributions; supports negative values (flight times). |",
        "",
        "---",
        "",
        "## 2. Train-Fold-Safety & Leakage Prevention",
        "",
        "1. **Zero Global Fitting:** Preprocessing transformers in Phase 6 remain strictly **unfitted**.",
        "2. **CV Integration:** In Phase 7, the pipeline will be combined with estimators inside `sklearn.pipeline.Pipeline` or `cross_validate`, ensuring that statistics (imputation medians, scaling means/standard deviations) are estimated exclusively on training folds.",
        "3. **Holdout Test Isolation:** Test partition participants ($N=120$) will never contribute to learned transformer parameters.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved preprocessing decisions report to %s", output_path)


def generate_feature_quality_report_markdown(
    output_path: Path,
    feature_validation: dict[str, Any],
    quality_validation: dict[str, Any],
    quality_summary_df: pd.DataFrame,
    flight_time_audit: dict[str, Any],
):
    """Generate reports/feature_engineering/feature_quality_report.md."""
    lines = [
        "# Feature Engineering Quality & Integrity Report",
        "",
        "**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  ",
        "**Dataset:** Tappy Extended Keystroke Dataset, Mendeley Data Version 3 (DOI: `10.17632/z39mhdsynx.3`)  ",
        "**Date:** 24 October 2026  ",
        f"**Validation Verdict:** {'PASSED' if feature_validation['passed'] and quality_validation['passed'] else 'FAILED'}  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Eligible Cohort Size:** {feature_validation['row_count']} participants (100% reconciled with Phase 4 split).",
        f"- **Unique Participant Identifiers:** {feature_validation['unique_participant_count']} (0 duplicates).",
        f"- **Approved Predictor Columns:** {feature_validation['predictor_column_count']} typing-derived features.",
        f"- **Total Keystroke Events Aggregated:** {quality_validation['total_records_sum']:,} events.",
        f"- **Warning-Tagged Events Retained:** {quality_validation['total_warned_records_sum']:,} events.",
        f"- **Date-Corruption Signature Columns:** {len(feature_validation['date_corrupted_columns'])}.",
        f"- **Predictor Columns Containing Infinite Values:** {len(feature_validation['infinite_columns'])}.",
        f"- **Participants with Negative Flight-Time Minimum:** {flight_time_audit['participants_with_negative_minimum']} of {feature_validation['row_count']}.",
        f"- **Minimum Observed Participant Flight Time:** {flight_time_audit['minimum_observed_flight_min_ms']} ms (`flight_min`).",
        "",
        "---",
        "",
        "## 2. Cohort Re-Extraction Resolution (Comparison to Legacy Table)",
        "",
        "| Metric | Legacy `features_clean.csv` | Phase 6 Re-Engineered Parquet | Integrity Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Cohort Coverage** | 495 participants (102 omitted) | **{feature_validation['row_count']} participants** | **RESOLVED (+102 participants)** |",
        f"| **Date Corruption (>100k)** | 28 corrupted columns | **0 corrupted columns** | **RESOLVED (100% clean)** |",
        f"| **Negative Flight Times** | Legacy-table issue requiring review | {flight_time_audit['participants_with_negative_minimum']} of {feature_validation['row_count']} participants have `flight_min < 0` | Empirical preservation is not demonstrated when no negative minima are present |",
        "| **Predictor Separation** | IDs & Labels mixed with features | **Strict allowlist enforced** | **RESOLVED (No leakage)** |",
        "",
        "---",
        "",
        "## 3. Missingness & Zero-Variance Audit",
        "",
        f"- **Features with 0% Missingness:** {len(quality_summary_df[quality_summary_df['missing_count'] == 0])} features.",
        f"- **Features with Missing Values:** {feature_validation['cols_with_missing_count']} features (directional transitions with zero event occurrences).",
        f"- **Constant / Zero-Variance Features:** {len(feature_validation['constant_columns'])} features.",
        "",
        "### Top Features by Missingness",
        "",
        "| Feature Name | Missing Count (out of 597) | Missing Percentage | Interpretation |",
        "| :--- | :--- | :--- | :--- |",
    ]

    top_missing = quality_summary_df[quality_summary_df["missing_count"] > 0].sort_values(by="missing_count", ascending=False).head(15)
    for _, r in top_missing.iterrows():
        lines.append(f"| `{r['feature_name']}` | {r['missing_count']} | {r['missing_pct']}% | Zero transition occurrences in participant sessions |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Verification Check Details",
        "",
    ])

    if feature_validation["issues"]:
        lines.append("### Flagged Issues:")
        for iss in feature_validation["issues"]:
            lines.append(f"- [WARNING] {iss}")
    else:
        lines.append("- [x] Row count matches eligible cohort (597).")
        lines.append("- [x] Participant IDs unique and disjoint.")
        lines.append("- [x] Target labels complete, valid, and consistent with Phase 2 metadata.")
        lines.append("- [x] Predictor allowlist strictly excludes identifiers and clinical covariates.")
        lines.append("- [x] Date corruption (>100k) completely absent.")
        lines.append(f"- Negative-flight observation: {flight_time_audit['participants_with_negative_minimum']} participants have a negative participant-level `flight_min`; this does not prove negative event-level intervals were observed or preserved.")
        lines.append("- [x] Zero infinite values across all predictors.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved feature quality report to %s", output_path)


def run_pipeline(config_path: str | Path | None = None):
    """Execute Phase 6 feature extraction and validation pipeline."""
    t0 = time.time()
    config = FeatureEngineeringConfig.load(
        config_path=config_path,
        repo_root=repo_root,
    )

    logger.info("Starting Phase 6 Feature Engineering & Preprocessing Pipeline")
    logger.info("Repository Root: %s", config.repo_root)

    # 1. Load and validate train/test IDs before constructing the cohort.
    train_df = pd.read_csv(config.train_ids_path)
    test_df = pd.read_csv(config.test_ids_path)

    def validated_split_ids(split_df: pd.DataFrame, split_name: str) -> list[str]:
        if "participant_id" not in split_df.columns:
            raise ValueError(f"{split_name} split is missing participant_id")
        if split_df["participant_id"].isna().any():
            raise ValueError(f"{split_name} split contains missing participant IDs")
        values = split_df["participant_id"].astype(str).tolist()
        if any(not value or value.strip() != value for value in values):
            raise ValueError(f"{split_name} split contains blank or padded participant IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"{split_name} split contains duplicate participant IDs")
        return values

    train_ids = validated_split_ids(train_df, "Training")
    test_ids = validated_split_ids(test_df, "Test")
    train_id_set, test_id_set = set(train_ids), set(test_ids)

    if not train_id_set.isdisjoint(test_id_set):
        overlap = sorted(train_id_set & test_id_set)
        raise ValueError(f"Training and test splits overlap; example IDs: {overlap[:10]}")

    split_union = train_id_set | test_id_set
    if len(split_union) != config.expected_eligible_count:
        raise ValueError(
            f"Eligible cohort mismatch: expected {config.expected_eligible_count}, "
            f"observed {len(split_union)} unique split participants"
        )
    eligible_ids = sorted(split_union)

    # 2. Require exactly one valid authoritative label per eligible participant.
    labels_df = pd.read_csv(config.participant_labels_path)
    if "user_key" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError("Authoritative participant labels must contain user_key and label")
    labels_df = labels_df.copy()
    labels_df["user_key"] = labels_df["user_key"].astype(str)
    eligible_labels = (
        labels_df[labels_df["user_key"].isin(split_union)]
        .dropna(subset=["label"])
        .copy()
    )
    if eligible_labels["user_key"].duplicated().any():
        raise ValueError("Authoritative labels contain duplicate eligible participant keys")
    if set(eligible_labels["user_key"]) != split_union:
        missing_labels = sorted(split_union - set(eligible_labels["user_key"]))
        raise ValueError(f"Authoritative labels do not cover every eligible participant: {missing_labels[:10]}")
    if not eligible_labels["label"].isin([0, 1, True, False]).all():
        raise ValueError("Authoritative labels contain values other than binary 0/1")
    label_map = dict(
        zip(eligible_labels["user_key"], eligible_labels["label"].astype(int))
    )

    # 3. Aggregate cohort features
    df_features, df_quality = aggregate_cohort_features(
        accepted_dir=config.interim_accepted_dir,
        eligible_ids=eligible_ids,
        labels_df=labels_df,
        config=config,
    )

    # Independently verify feature/quality IDs and labels against the authoritative cohort.
    required_metadata = {"participant_id", "parkinsons_label"}
    if not required_metadata.issubset(df_features.columns):
        raise ValueError(
            f"Feature table lacks required metadata columns: {sorted(required_metadata - set(df_features.columns))}"
        )
    if "participant_id" not in df_quality.columns:
        raise ValueError("Quality table is missing participant_id")
    if df_features["participant_id"].isna().any() or df_quality["participant_id"].isna().any():
        raise ValueError("Feature/quality table contains missing participant IDs")

    feature_id_values = df_features["participant_id"].astype(str).tolist()
    quality_id_values = df_quality["participant_id"].astype(str).tolist()
    feature_id_set, quality_id_set = set(feature_id_values), set(quality_id_values)

    if len(feature_id_values) != len(feature_id_set):
        raise ValueError("Feature table contains duplicate participant IDs")
    if len(quality_id_values) != len(quality_id_set):
        raise ValueError("Quality table contains duplicate participant IDs")
    if feature_id_set != split_union:
        raise ValueError(
            "Feature table IDs do not match the split union: "
            f"missing={len(split_union - feature_id_set)}, "
            f"unexpected={len(feature_id_set - split_union)}"
        )
    if quality_id_set != feature_id_set:
        raise ValueError("Quality-table participant IDs do not match feature-table IDs")

    label_consistency = all(
        pd.notna(value)
        and str(participant_id) in label_map
        and int(value) == int(label_map[str(participant_id)])
        for participant_id, value in zip(
            df_features["participant_id"], df_features["parkinsons_label"]
        )
    )
    if not label_consistency:
        raise ValueError("Feature labels do not match authoritative participant labels")

    # 4. Validate feature table
    feature_val = validate_feature_table(
        df_features=df_features,
        expected_count=config.expected_eligible_count,
        expected_labels=label_map,
    )

    # 5. Validate quality table
    quality_val = validate_quality_table(
        df_quality=df_quality,
        expected_count=config.expected_eligible_count,
    )

    logger.info("Feature Validation Status: %s", "PASSED" if feature_val["passed"] else "FAILED")
    logger.info("Quality Validation Status: %s", "PASSED" if quality_val["passed"] else "FAILED")

    if not feature_val["passed"]:
        logger.error("Feature validation issues: %s", feature_val["issues"])
        raise RuntimeError(f"Feature table validation failed: {feature_val['issues']}")

    if not quality_val["passed"]:
        logger.error(
            "Quality validation failed: %s",
            quality_val.get("issues", []),
        )
        raise RuntimeError(
            f"Quality table validation failed: {quality_val}"
        )

    # 6. Save Parquet outputs
    config.features_output_dir.mkdir(parents=True, exist_ok=True)
    raw_features_path = config.features_output_dir / config.raw_features_file
    quality_features_path = config.features_output_dir / config.quality_features_file

    # Compute manifest integrity checks from the actual tables and splits.
    approved_predictors = set(get_predictor_columns())
    actual_predictor_columns = [
        c for c in df_features.columns
        if c not in {"participant_id", "parkinsons_label"}
    ]
    predictor_schema_match = (
        len(actual_predictor_columns) == len(approved_predictors)
        and set(actual_predictor_columns) == approved_predictors
    )

    numeric_predictors = df_features[actual_predictor_columns].select_dtypes(include=[np.number])
    numeric_values = numeric_predictors.to_numpy(dtype=np.float64, na_value=np.nan)
    infinite_predictor_values = int(np.isinf(numeric_values).sum())
    predictor_values_gt_100000 = int((numeric_values > 100000).sum())
    finite_values = numeric_values[np.isfinite(numeric_values)]
    exact_date_signature_values = int((
        (finite_values >= 160000)
        & (finite_values <= 180000)
        & (finite_values == np.floor(finite_values))
    ).sum())

    flight_min_values = (
        df_features["flight_min"].dropna().astype(float)
        if "flight_min" in df_features.columns
        else pd.Series(dtype="float64")
    )
    participants_with_negative_flight_minimum = int((flight_min_values < 0).sum())
    minimum_observed_flight_min_ms = (
        float(flight_min_values.min()) if not flight_min_values.empty else None
    )

    integrity_verifications = {
        "expected_cohort_size_matched": bool(
            len(df_features) == config.expected_eligible_count
            and len(df_quality) == config.expected_eligible_count
        ),
        "feature_ids_match_split_union": bool(
            len(feature_id_values) == config.expected_eligible_count
            and feature_id_set == split_union
        ),
        "train_test_splits_disjoint": bool(train_id_set.isdisjoint(test_id_set)),
        "feature_quality_ids_aligned": bool(
            len(quality_id_values) == config.expected_eligible_count
            and quality_id_set == feature_id_set
        ),
        "authoritative_labels_match": bool(label_consistency),
        "approved_predictor_schema_enforced": bool(predictor_schema_match),
        "zero_predictor_values_gt_100000": bool(predictor_values_gt_100000 == 0),
        "zero_exact_yymmdd_date_signatures": bool(exact_date_signature_values == 0),
        "zero_infinite_predictor_values": bool(infinite_predictor_values == 0),
    }
    failed_checks = [
        name for name, passed in integrity_verifications.items() if not passed
    ]
    if failed_checks:
        raise RuntimeError(
            "Computed Phase 6 integrity checks failed: " + ", ".join(failed_checks)
        )

    flight_time_observation_audit = {
        "observation_scope": "participant-level feature table",
        "feature_name": "flight_min",
        "participants_with_negative_minimum": participants_with_negative_flight_minimum,
        "minimum_observed_flight_min_ms": minimum_observed_flight_min_ms,
        "negative_minimum_observed": bool(participants_with_negative_flight_minimum > 0),
        "interpretation": (
            "The Phase 2 validator permits negative raw flight_ms values, but the accepted-cohort "
            "audit observed none. Participant-level minima alone do not prove negative event-level "
            "intervals were present or preserved."
        ),
    }

    df_features.to_parquet(raw_features_path, index=False)
    df_quality.to_parquet(quality_features_path, index=False)
    logger.info("Saved %s (shape: %s)", raw_features_path, df_features.shape)
    logger.info("Saved %s (shape: %s)", quality_features_path, df_quality.shape)

    # 7. Compute file hashes
    raw_sha = compute_file_sha256(raw_features_path)
    qual_sha = compute_file_sha256(quality_features_path)

    # 8. Save Feature Quality Summary CSV
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    summary_df = generate_feature_quality_summary(df_features)
    quality_summary_path = config.reports_dir / "feature_quality_summary.csv"
    summary_df.to_csv(quality_summary_path, index=False)

    coverage_summary_path = config.reports_dir / "participant_feature_coverage.csv"
    df_quality.to_csv(coverage_summary_path, index=False)

    # 9. Generate Documentation & Markdown Reports
    generate_feature_dictionary_markdown(config.docs_dir / "feature_dictionary.md")
    generate_phase6_decisions_markdown(config.docs_dir / "phase6_decisions.md")
    generate_preprocessing_decisions_markdown(config.reports_dir / "preprocessing_decisions.md")
    generate_feature_quality_report_markdown(
        config.reports_dir / "feature_quality_report.md",
        feature_val,
        quality_val,
        summary_df,
        flight_time_observation_audit,
    )

    # 10. Generate Authoritative Feature Manifest
    manifest = {
        "manifest_version": "1.0",
        "phase": "Phase 6 — Feature Engineering and Preprocessing",
        "dataset_name": "Tappy Extended Keystroke Dataset, Mendeley Data Version 3",
        "dataset_doi": "10.17632/z39mhdsynx.3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_time_seconds": round(time.time() - t0, 2),
        "cohort": {
            "eligible_participant_count": len(df_features),
            "parkinsons_count": int((df_features["parkinsons_label"] == 1).sum()),
            "control_count": int((df_features["parkinsons_label"] == 0).sum()),
            "prevalence_pct": round(float((df_features["parkinsons_label"] == 1).mean()) * 100, 2),
        },
        "feature_counts": {
            "total_columns": df_features.shape[1],
            "participant_id_col": 1,
            "target_label_col": 1,
            "predictor_feature_count": df_features.shape[1] - 2,
        },
        "artifacts": {
            "raw_features_parquet": {
                "path": str(raw_features_path.relative_to(repo_root)),
                "sha256": raw_sha,
                "rows": df_features.shape[0],
                "columns": df_features.shape[1],
            },
            "quality_features_parquet": {
                "path": str(quality_features_path.relative_to(repo_root)),
                "sha256": qual_sha,
                "rows": df_quality.shape[0],
                "columns": df_quality.shape[1],
            },
        },
        "integrity_verifications": integrity_verifications,
        "integrity_metrics": {
            "predictor_values_gt_100000": predictor_values_gt_100000,
            "exact_yymmdd_integer_date_signatures": exact_date_signature_values,
            "infinite_predictor_values": infinite_predictor_values,
        },
        "flight_time_observation_audit": flight_time_observation_audit,
        "split_integrity": {
            "train_count": len(train_id_set),
            "test_count": len(test_id_set),
            "train_test_intersection_count": len(train_id_set & test_id_set),
            "split_union_count": len(split_union),
            "feature_id_count": len(feature_id_set),
            "quality_id_count": len(quality_id_set),
        },
        "preprocessing_state": {
            "imputation_fitted": False,
            "scaler_fitted": False,
            "policy": "Fit preprocessing inside training folds in Phase 7; Phase 6 exports unscaled features.",
        },
    }

    manifest_path = config.features_output_dir / config.manifest_file
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved feature manifest to %s", manifest_path)

    logger.info("Phase 6 pipeline completed successfully in %.2f seconds", time.time() - t0)


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 6 feature engineering and preprocessing."
    )
    parser.add_argument(
        "--config",
        default="configs/feature_engineering.toml",
        help="Path to the feature-engineering TOML configuration.",
    )
    args = parser.parse_args()
    run_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
