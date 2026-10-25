"""Phase 5 Exploratory Data Analysis Engine.

Performs systematic exploratory data analysis on the training cohort of the
Tappy Extended Keystroke Dataset:
- Validates Phase 4 readiness gate with fail-closed enforcement.
- Inventories feature families and attributes dynamically.
- Audits missingness and non-finite values.
- Computes comprehensive univariate descriptive statistics dynamically.
- Screens for outliers, semantic violations, and date-code corruption.
- Evaluates pairwise correlations and collinearity.
- Conducts exploratory class comparisons on the training set.
- Exports aggregate tables, figures, and question briefs for Phase 6.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROHIBITED_COVARIATE_TERMS = {
    "birthyear", "gender", "updrs", "tremor", "tremors", "diagnosisyear",
    "medication", "medications", "age", "sided", "impact", "levodopa",
    "da", "maob", "other",
}


def evaluate_readiness_gate(
    features_path: Path | str | None = None,
    train_ids_path: Path | str | None = None,
    test_ids_path: Path | str | None = None,
    modelling_dataset_path: Path | str | None = None,
    eligible_count: int = 597,
    expected_train_count: int = 477,
    expected_test_count: int = 120,
    expected_feature_count: int | None = 153,
    allow_legacy_fallback: bool = False,
) -> dict[str, Any]:
    """Evaluate mandatory Phase 4 data-readiness gate criteria with strict fail-closed contract.

    Supports reading Parquet or CSV. Rejects untrustworthy legacy tables unless explicitly
    allowed for diagnostic auditing.

    Args:
        features_path: Path to canonical feature table (Parquet or CSV).
        train_ids_path: Path to authoritative training participant IDs.
        test_ids_path: Path to authoritative test participant IDs.
        modelling_dataset_path: Optional path for backward-compatibility checks.
        eligible_count: Expected total eligible participant count (default: 597).
        expected_train_count: Expected training partition participant count (default: 477).
        expected_test_count: Expected test partition participant count (default: 120).
        expected_feature_count: Expected predictor feature count (default: 153).
        allow_legacy_fallback: If False, explicitly rejects legacy features_clean.csv.

    Returns:
        dict[str, Any]: Comprehensive dictionary containing gate_passed flag,
            individual boolean checks, actionable blocker explanations, and metrics.
    """
    # Handle backward-compatibility where modelling_dataset_path might be passed as first argument
    if features_path is not None and train_ids_path is not None and test_ids_path is not None:
        p_feat = Path(features_path)
        p_tr = Path(train_ids_path)
        p_te = Path(test_ids_path)
    elif modelling_dataset_path is not None and features_path is not None and train_ids_path is not None:
        # Caller used legacy positional signature: (modelling_dataset_path, features_path, train_ids_path, test_ids_path)
        p_feat = Path(features_path)
        p_tr = Path(train_ids_path)
        p_te = Path(test_ids_path) if test_ids_path is not None else Path(modelling_dataset_path)
    else:
        p_feat = Path(features_path) if features_path else Path("missing_features")
        p_tr = Path(train_ids_path) if train_ids_path else Path("missing_train_ids")
        p_te = Path(test_ids_path) if test_ids_path else Path("missing_test_ids")

    blockers: list[str] = []

    # Check 1: Feature table existence and legacy rejection
    check1_feature_file_exists = p_feat.is_file()
    check1_legacy_rejected = True
    if not check1_feature_file_exists:
        blockers.append(f"Canonical feature table does not exist at '{p_feat}'.")
    else:
        # Detect legacy corrupted table
        if "features_clean.csv" in p_feat.name.lower() and not allow_legacy_fallback:
            check1_legacy_rejected = False
            blockers.append(
                f"Feature table '{p_feat.name}' is the rejected legacy candidate "
                "(known truncation to 495 participants and 28 date-corrupted columns). "
                "Silent fallback is prohibited."
            )

    # Check 2: Splits existence
    check2_splits_exist = p_tr.is_file() and p_te.is_file()
    if not p_tr.is_file():
        blockers.append(f"Training split file does not exist at '{p_tr}'.")
    if not p_te.is_file():
        blockers.append(f"Test split file does not exist at '{p_te}'.")

    # Read feature table if available and valid
    df_feat: pd.DataFrame | None = None
    feat_rows = 0
    predictor_cols: list[str] = []
    corrupted_date_cols = 0
    infinite_values_found = False

    check3_one_row_per_participant = False
    check4_labels_valid = False
    check5_cohort_coverage = False
    check6_splits_integrity = False
    check7_schema_valid = False
    check8_data_unvitiated = False

    if check1_feature_file_exists and check1_legacy_rejected:
        try:
            if p_feat.suffix.lower() == ".parquet":
                df_feat = pd.read_parquet(p_feat)
            else:
                df_feat = pd.read_csv(p_feat)
            feat_rows = len(df_feat)
        except Exception as e:
            blockers.append(f"Failed to load feature table '{p_feat}': {e}")
            df_feat = None

    if df_feat is not None:
        # Check 3: Unique participant ID
        if "participant_id" in df_feat.columns:
            n_unique = df_feat["participant_id"].nunique(dropna=True)
            check3_one_row_per_participant = bool(n_unique == feat_rows and df_feat["participant_id"].notna().all())
            if not check3_one_row_per_participant:
                blockers.append(
                    f"Participant IDs in feature table are not unique or contain nulls "
                    f"({n_unique} unique IDs for {feat_rows} rows)."
                )
        else:
            blockers.append("Feature table lacks required 'participant_id' column.")

        # Check 4: Valid target labels
        if "parkinsons_label" in df_feat.columns:
            s_label = df_feat["parkinsons_label"]
            check4_labels_valid = bool(s_label.notna().all() and s_label.isin([0, 1]).all())
            if not check4_labels_valid:
                blockers.append(
                    f"Target label 'parkinsons_label' contains invalid values or NaNs: "
                    f"observed values: {s_label.unique().tolist()}."
                )
        else:
            blockers.append("Feature table lacks required 'parkinsons_label' column.")

        # Check 5: Eligible cohort coverage
        check5_cohort_coverage = bool(feat_rows == eligible_count)
        if not check5_cohort_coverage:
            blockers.append(
                f"Feature table row count ({feat_rows}) does not match verified "
                f"eligible cohort count ({eligible_count})."
            )

        # Check 7: Feature schema & predictor count & prohibited covariates
        predictor_cols = [c for c in df_feat.columns if c not in ["participant_id", "parkinsons_label"]]
        schema_issues: list[str] = []

        if expected_feature_count is not None and len(predictor_cols) != expected_feature_count:
            schema_issues.append(
                f"Predictor feature count mismatch: expected {expected_feature_count}, "
                f"observed {len(predictor_cols)}."
            )

        # Prohibited covariates check
        found_prohibited = [
            c for c in predictor_cols
            if any(term in c.lower() for term in PROHIBITED_COVARIATE_TERMS)
        ]
        if found_prohibited:
            schema_issues.append(
                f"Prohibited demographic/clinical covariates found in predictor columns: {found_prohibited}."
            )

        check7_schema_valid = (len(schema_issues) == 0)
        blockers.extend(schema_issues)

        # Check 8: Data unvitiated (zero date corruption, zero inf)
        num_cols = df_feat.select_dtypes(include=np.number).columns
        for c in num_cols:
            if c in predictor_cols:
                s = df_feat[c]
                if (s > 100000).any():
                    corrupted_date_cols += 1
                if np.isinf(s).any():
                    infinite_values_found = True

        check8_data_unvitiated = (corrupted_date_cols == 0 and not infinite_values_found)
        if corrupted_date_cols > 0:
            blockers.append(
                f"{corrupted_date_cols} predictor columns contain values > 100,000 "
                "(date-code corruption artifact)."
            )
        if infinite_values_found:
            blockers.append("Infinite values detected in numerical predictor columns.")

    # Check 6: Split partition integrity (train & test)
    if check2_splits_exist:
        try:
            tr_df = pd.read_csv(p_tr)
            te_df = pd.read_csv(p_te)
            tr_ids = set(tr_df["participant_id"])
            te_ids = set(te_df["participant_id"])

            disjoint = (len(tr_ids.intersection(te_ids)) == 0)
            correct_counts = (len(tr_ids) == expected_train_count and len(te_ids) == expected_test_count)

            union_matches = True
            if df_feat is not None and "participant_id" in df_feat.columns:
                union_matches = (tr_ids.union(te_ids) == set(df_feat["participant_id"]))

            check6_splits_integrity = (disjoint and correct_counts and union_matches)
            if not disjoint:
                blockers.append("Leakage detected: training and test split partitions overlap.")
            if not correct_counts:
                blockers.append(
                    f"Split partition counts mismatch: expected train={expected_train_count}, "
                    f"test={expected_test_count}; observed train={len(tr_ids)}, test={len(te_ids)}."
                )
            if not union_matches:
                blockers.append("Split participant IDs union does not match feature table IDs.")
        except Exception as e:
            blockers.append(f"Failed to validate split files: {e}")

    # Overall gate status
    gate_passed = bool(
        check1_feature_file_exists
        and check1_legacy_rejected
        and check2_splits_exist
        and check3_one_row_per_participant
        and check4_labels_valid
        and check5_cohort_coverage
        and check6_splits_integrity
        and check7_schema_valid
        and check8_data_unvitiated
    )

    return {
        "gate_passed": gate_passed,
        "verdict": "GATE_PASSED" if gate_passed else "GATE_FAILED_BLOCKED",
        "checks": {
            "check1_feature_file_exists": check1_feature_file_exists,
            "check1_legacy_rejected": check1_legacy_rejected,
            "check2_splits_exist": check2_splits_exist,
            "check3_one_row_per_participant": check3_one_row_per_participant,
            "check4_labels_valid": check4_labels_valid,
            "check5_cohort_coverage": check5_cohort_coverage,
            "check6_splits_integrity": check6_splits_integrity,
            "check7_schema_valid": check7_schema_valid,
            "check8_data_unvitiated": check8_data_unvitiated,
        },
        "blockers": blockers,
        "metrics": {
            "feature_table_path": str(p_feat),
            "feature_table_row_count": feat_rows,
            "eligible_cohort_target": eligible_count,
            "predictor_feature_count": len(predictor_cols),
            "expected_feature_count": expected_feature_count,
            "corrupted_date_columns_count": corrupted_date_cols,
            "infinite_values_found": infinite_values_found,
            "train_ids_path": str(p_tr),
            "test_ids_path": str(p_te),
        },
    }


def enforce_readiness_gate(gate_results: dict[str, Any]) -> None:
    """Enforce fail-closed gate behavior.

    Raises RuntimeError with actionable blocker messages if gate_passed is False.
    """
    if not gate_results.get("gate_passed", False):
        blockers = gate_results.get("blockers", ["Phase 4 Readiness Gate failed."])
        msg = (
            "[FAIL-CLOSED READINESS GATE] Analysis execution stopped. "
            "The following prerequisites are unmet:\n" +
            "\n".join(f"  - {b}" for b in blockers)
        )
        raise RuntimeError(msg)


def build_feature_inventory(df_train: pd.DataFrame) -> pd.DataFrame:
    """Build a comprehensive feature inventory table for all candidate predictor columns."""
    predictor_cols = [c for c in df_train.columns if c not in ["participant_id", "parkinsons_label"]]

    inventory = []
    for col in sorted(predictor_cols):
        s = df_train[col]
        dtype = str(s.dtype)
        n_null = int(s.isna().sum())
        null_pct = round(n_null / len(s) * 100, 2)
        n_unique = int(s.nunique(dropna=True))
        is_const = bool(n_unique <= 1)

        # Dynamic family classification
        col_lower = col.lower()
        if col.startswith("hold_"):
            family = "Hold Time (HT)"
            unit = "Milliseconds (ms)"
        elif col.startswith("latency_"):
            family = "Latency (L)"
            unit = "Milliseconds (ms)"
        elif col.startswith("flight_"):
            family = "Flight Time (FT)"
            unit = "Milliseconds (ms)"
        elif col.startswith("hand_") or col.startswith("prop_"):
            family = "Hand Dynamics"
            unit = "Proportion [0, 1]"
        elif any(d in col for d in ["_LL", "_LR", "_RL", "_RR", "_LS", "_RS", "_SL", "_SR", "_SS"]) or col.startswith("trans_"):
            family = "Transition Dynamics"
            unit = "Milliseconds / Proportion"
        else:
            family = "Typing Predictor"
            unit = "Unspecified"

        # Integrity status
        issues = []
        if pd.api.types.is_numeric_dtype(s) and (s > 100000).any():
            issues.append("DATE_CORRUPTED")
        if n_null > 0:
            issues.append("CONTAINS_NULLS")
        status = ";".join(issues) if issues else "VERIFIED_NUMERIC"

        inventory.append({
            "feature_name": col,
            "family": family,
            "dtype": dtype,
            "unit": unit,
            "missing_count": n_null,
            "missing_pct": null_pct,
            "n_unique": n_unique,
            "is_constant": is_const,
            "integrity_status": status,
        })

    return pd.DataFrame(inventory)


def compute_missingness_summary(df_train: pd.DataFrame) -> pd.DataFrame:
    """Compute missingness summary for training predictors dynamically."""
    predictor_cols = [c for c in df_train.columns if c not in ["participant_id", "parkinsons_label"]]
    missing_data = []

    for col in predictor_cols:
        s = df_train[col]
        n_missing = int(s.isna().sum())
        pct_missing = round(n_missing / len(s) * 100, 3)
        n_inf = int(np.isinf(s).sum()) if pd.api.types.is_numeric_dtype(s) else 0

        missing_data.append({
            "feature_name": col,
            "missing_count": n_missing,
            "missing_percentage": pct_missing,
            "infinite_count": n_inf,
            "valid_count": len(s) - n_missing,
            "total_observations": len(s),
        })

    missing_df = pd.DataFrame(missing_data)
    if not missing_df.empty:
        return missing_df.sort_values(by="missing_count", ascending=False).reset_index(drop=True)
    return missing_df


def compute_descriptive_statistics(df_train: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive descriptive statistics across all numerical predictors in training set."""
    predictor_cols = [c for c in df_train.columns if c not in ["participant_id", "parkinsons_label"]]
    stats_rows = []

    for col in sorted(predictor_cols):
        s = df_train[col].dropna()
        if not pd.api.types.is_numeric_dtype(s) or len(s) == 0:
            continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_extreme = int(((s < lower_bound) | (s > upper_bound)).sum())

        stats_rows.append({
            "feature_name": col,
            "count": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4) if len(s) > 1 else 0.0,
            "min": round(float(s.min()), 4),
            "q25": round(q1, 4),
            "median": round(float(s.median()), 4),
            "q75": round(q3, 4),
            "max": round(float(s.max()), 4),
            "iqr": round(iqr, 4),
            "skewness": round(float(s.skew()), 4) if len(s) >= 3 else 0.0,
            "kurtosis": round(float(s.kurtosis()), 4) if len(s) >= 4 else 0.0,
            "n_unique": int(s.nunique()),
            "n_extreme_iqr": n_extreme,
            "pct_extreme_iqr": round(n_extreme / len(s) * 100, 2),
            "n_gt_100k": int((s > 100000).sum()),
        })

    return pd.DataFrame(stats_rows)


def screen_outliers_and_anomalies(df_train: pd.DataFrame) -> pd.DataFrame:
    """Screen statistical extremes and date-corruption artifacts."""
    predictor_cols = [c for c in df_train.columns if c not in ["participant_id", "parkinsons_label"]]
    screening_rows = []

    for col in sorted(predictor_cols):
        s = df_train[col].dropna()
        if not pd.api.types.is_numeric_dtype(s) or len(s) == 0:
            continue

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        n_gt_100k = int((s > 100000).sum())
        n_extreme_iqr = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

        if n_gt_100k > 0:
            anomaly_class = "KNOWN_DATE_CORRUPTION_ARTIFACT"
            severity = "FATAL_BLOCKER"
            threshold = "value > 100,000"
            rec = "Must re-aggregate from Phase 2 interim Parquet in Phase 6."
            n_flagged = n_gt_100k
        elif n_extreme_iqr > 0:
            anomaly_class = "STATISTICAL_EXTREME_VALUE"
            severity = "LOW_EXPLORATORY"
            threshold = f"outside [{round(q1 - 1.5 * iqr, 1)}, {round(q3 + 1.5 * iqr, 1)}]"
            rec = "Evaluate robust scaling or tree models in Phase 6."
            n_flagged = n_extreme_iqr
        else:
            continue

        screening_rows.append({
            "feature_name": col,
            "anomaly_classification": anomaly_class,
            "severity_level": severity,
            "flagged_observations_count": n_flagged,
            "screening_threshold": threshold,
            "observed_min": round(float(s.min()), 2),
            "observed_max": round(float(s.max()), 2),
            "recommended_phase6_action": rec,
        })

    return pd.DataFrame(screening_rows)


def analyze_feature_correlations(df_train: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Analyze feature correlations on uncorrupted training features to identify collinear pairs."""
    clean_cols = [
        c for c in df_train.columns
        if c not in ["participant_id", "parkinsons_label"]
        and pd.api.types.is_numeric_dtype(df_train[c])
        and (df_train[c] <= 100000).all()
        and df_train[c].notna().sum() > 10
    ]

    if len(clean_cols) < 2:
        return pd.DataFrame(columns=["feature_1", "feature_2", "spearman_rho", "absolute_rho", "relationship_interpretation"])

    corr_matrix = df_train[clean_cols].corr(method="spearman")

    pairs = []
    seen = set()
    for i in range(len(clean_cols)):
        for j in range(i + 1, len(clean_cols)):
            col1 = clean_cols[i]
            col2 = clean_cols[j]
            r = corr_matrix.loc[col1, col2]
            if abs(r) >= threshold:
                pair_key = tuple(sorted([col1, col2]))
                if pair_key not in seen:
                    seen.add(pair_key)
                    pairs.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "spearman_rho": round(float(r), 4),
                        "absolute_rho": round(abs(float(r)), 4),
                        "relationship_interpretation": "Collinear central tendencies / scale metrics"
                        if col1.split("_")[0] == col2.split("_")[0]
                        else "Cross-feature correlation",
                    })

    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(by="absolute_rho", ascending=False).reset_index(drop=True)
    return pairs_df


def compute_training_class_summary(df_train: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution comparisons between PD and Control on representative training features."""
    rep_features = [
        "hold_mean", "hold_median", "hold_std", "hold_iqr", "hold_skew", "hold_cv",
        "latency_mean", "latency_median", "latency_std", "latency_iqr",
        "flight_mean", "flight_median", "flight_std",
    ]
    rep_features = [f for f in rep_features if f in df_train.columns]

    class_rows = []
    for f in rep_features:
        s0 = df_train[df_train["parkinsons_label"] == 0][f].dropna()
        s1 = df_train[df_train["parkinsons_label"] == 1][f].dropna()

        c0_mean = round(float(s0.mean()), 3) if len(s0) > 0 else 0.0
        c0_med = round(float(s0.median()), 3) if len(s0) > 0 else 0.0
        c0_iqr = round(float(s0.quantile(0.75) - s0.quantile(0.25)), 3) if len(s0) > 0 else 0.0

        c1_mean = round(float(s1.mean()), 3) if len(s1) > 0 else 0.0
        c1_med = round(float(s1.median()), 3) if len(s1) > 0 else 0.0
        c1_iqr = round(float(s1.quantile(0.75) - s1.quantile(0.25)), 3) if len(s1) > 0 else 0.0

        class_rows.append({
            "feature_name": f,
            "control_count": len(s0),
            "control_mean": c0_mean,
            "control_median": c0_med,
            "control_iqr": c0_iqr,
            "pd_count": len(s1),
            "pd_mean": c1_mean,
            "pd_median": c1_med,
            "pd_iqr": c1_iqr,
            "median_diff_pd_minus_ctrl": round(c1_med - c0_med, 3),
        })

    return pd.DataFrame(class_rows)


def generate_eda_figures(df_train: pd.DataFrame, output_dir: Path | str) -> list[str]:
    """Generate and save readable diagnostic EDA figures dynamically using training participant counts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_figures = []
    n_train = len(df_train)

    # Palette setup
    sns.set_theme(style="whitegrid", font_scale=1.0)
    palette = {"Control": "#2b5c8f", "Parkinson": "#d95f02"}

    # 1. Missingness Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    pred_cols = [c for c in df_train.columns if c not in ["participant_id", "parkinsons_label"]]
    null_counts = df_train[pred_cols].isna().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False).head(20)

    if not null_counts.empty:
        sns.barplot(x=null_counts.values, y=null_counts.index, color="#4575b4", ax=ax)
        ax.set_title("Top Training Predictors with Missing Values", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"Number of Missing Participants (out of N={n_train})", fontsize=10)
        ax.set_ylabel("Feature Name", fontsize=10)
    else:
        ax.text(0.5, 0.5, "Zero Missing Values across Evaluated Predictors", ha="center", va="center")
    plt.tight_layout()
    fig1_path = out_dir / "fig1_missingness_by_feature.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    saved_figures.append(str(fig1_path))

    # 2. Hold Time Distributions
    df_plot = df_train.copy()
    df_plot["Group"] = df_plot["parkinsons_label"].map({0: "Control", 1: "Parkinson"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if "hold_mean" in df_plot.columns:
        sns.histplot(data=df_plot, x="hold_mean", hue="Group", kde=True, palette=palette, ax=axes[0], bins=25)
        axes[0].set_title(f"Key Hold Time (Mean) Distribution (N={n_train})", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Hold Time Mean (ms)", fontsize=10)
        axes[0].set_ylabel("Participant Count", fontsize=10)

    if "hold_std" in df_plot.columns:
        sns.histplot(data=df_plot, x="hold_std", hue="Group", kde=True, palette=palette, ax=axes[1], bins=25)
        axes[1].set_title(f"Key Hold Time Variability (Std Dev) (N={n_train})", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Hold Time Std Dev (ms)", fontsize=10)
        axes[1].set_ylabel("Participant Count", fontsize=10)
    plt.tight_layout()
    fig2_path = out_dir / "fig2_hold_time_distributions.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    saved_figures.append(str(fig2_path))

    # 3. Date Corruption Screening / Extreme Value Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if "latency_max" in df_plot.columns and "hold_mean" in df_plot.columns:
        sns.scatterplot(data=df_plot, x="hold_mean", y="latency_max", hue="Group", palette=palette, alpha=0.8, ax=axes[0])
        axes[0].axhline(100000, color="red", linestyle="--", label="Date Threshold (100k)")
        axes[0].set_title("Maximum Latency vs Mean Hold Time", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Hold Time Mean (ms)", fontsize=10)
        axes[0].set_ylabel("Maximum Latency (ms)", fontsize=10)
        axes[0].legend(loc="upper left")

    if "flight_max" in df_plot.columns and "hold_mean" in df_plot.columns:
        sns.scatterplot(data=df_plot, x="hold_mean", y="flight_max", hue="Group", palette=palette, alpha=0.8, ax=axes[1])
        axes[1].axhline(100000, color="red", linestyle="--", label="Date Threshold (100k)")
        axes[1].set_title("Maximum Flight Time vs Mean Hold Time", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Hold Time Mean (ms)", fontsize=10)
        axes[1].set_ylabel("Maximum Flight Time (ms)", fontsize=10)
        axes[1].legend(loc="upper left")
    plt.tight_layout()
    fig3_path = out_dir / "fig3_date_corruption_evidence.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    saved_figures.append(str(fig3_path))

    # 4. Participant-Level Minimum Flight Time Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    if "flight_min" in df_plot.columns:
        flight_min_values = df_plot["flight_min"].dropna()
        if not flight_min_values.empty:
            sns.histplot(flight_min_values, bins=30, color="#4575b4", ax=ax)
            ax.set_title(
                f"Participant-Level Minimum Flight Time (n={len(flight_min_values)})",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xlabel("Minimum Flight Time (ms)", fontsize=10)
            ax.set_ylabel("Participant Count", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No non-missing flight_min values", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "flight_min not present in training features", ha="center", va="center")
    plt.tight_layout()
    fig4_path = out_dir / "fig4_flight_time_minimum_distribution.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    saved_figures.append(str(fig4_path))

    # 5. Correlation Heatmap on Representative Core Features
    core_features = [
        "hold_mean", "hold_std", "hold_median", "hold_iqr", "hold_cv", "hold_skew",
        "latency_mean", "latency_median", "latency_std", "latency_iqr",
    ]
    core_features = [f for f in core_features if f in df_plot.columns]
    fig, ax = plt.subplots(figsize=(10, 8))
    if len(core_features) >= 2:
        corr_mat = df_plot[core_features].corr(method="spearman")
        sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title(f"Spearman Correlation Matrix: Core Dynamics (Train N={n_train})", fontsize=12, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Insufficient core features for correlation heatmap", ha="center", va="center")
    plt.tight_layout()
    fig5_path = out_dir / "fig5_correlation_heatmap.png"
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    saved_figures.append(str(fig5_path))

    # 6. Label Comparison Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    if "hold_mean" in df_plot.columns:
        sns.boxplot(data=df_plot, x="Group", y="hold_mean", hue="Group", palette=palette, legend=False, ax=axes[0], showmeans=True)
        axes[0].set_title("Key Hold Time Mean by Label", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("Hold Time Mean (ms)", fontsize=10)

    if "hold_std" in df_plot.columns:
        sns.boxplot(data=df_plot, x="Group", y="hold_std", hue="Group", palette=palette, legend=False, ax=axes[1], showmeans=True)
        axes[1].set_title("Hold Time Std Dev by Label", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Hold Time Std Dev (ms)", fontsize=10)

    if "hold_cv" in df_plot.columns:
        sns.boxplot(data=df_plot, x="Group", y="hold_cv", hue="Group", palette=palette, legend=False, ax=axes[2], showmeans=True)
        axes[2].set_title("Hold Time CV by Label", fontsize=11, fontweight="bold")
        axes[2].set_ylabel("Coefficient of Variation (CV)", fontsize=10)
    plt.tight_layout()
    fig6_path = out_dir / "fig6_label_comparison_boxplots.png"
    plt.savefig(fig6_path, dpi=200)
    plt.close()
    saved_figures.append(str(fig6_path))

    return saved_figures
