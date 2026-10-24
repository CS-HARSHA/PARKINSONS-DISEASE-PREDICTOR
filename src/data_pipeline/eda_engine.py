"""Phase 5 Exploratory Data Analysis Engine.

Performs systematic exploratory data analysis on the training cohort of the
Tappy Extended Keystroke Dataset:
- Validates Phase 4 readiness gate.
- Inventories feature families and attributes.
- Audits missingness and non-finite values.
- Computes comprehensive univariate descriptive statistics.
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


def evaluate_readiness_gate(
    modelling_dataset_path: Path | str,
    features_path: Path | str,
    train_ids_path: Path | str,
    test_ids_path: Path | str,
    eligible_count: int = 597,
) -> dict[str, Any]:
    """Evaluate the 8 mandatory Phase 4 readiness gate criteria."""
    modelling_path = Path(modelling_dataset_path)
    feat_path = Path(features_path)
    train_path = Path(train_ids_path)
    test_path = Path(test_ids_path)

    check1 = modelling_path.is_file()
    
    check2 = False
    check3 = False
    check4 = False
    check5 = False
    check6 = False
    check7 = False
    check8 = False
    
    feat_rows = 0
    corrupted_date_cols = 0
    if feat_path.is_file():
        df_feat = pd.read_csv(feat_path)
        feat_rows = len(df_feat)
        check2 = df_feat['participant_id'].nunique() == feat_rows
        check3 = df_feat['parkinsons_label'].notna().all() and df_feat['parkinsons_label'].isin([0, 1]).all()
        check4 = feat_rows == eligible_count  # Fails: 495 vs 597
        
        # Check date corruption
        for c in df_feat.select_dtypes(include=np.number).columns:
            if (df_feat[c] > 100000).any():
                corrupted_date_cols += 1
        check5 = (corrupted_date_cols == 0)  # Fails: 28 cols corrupted
        check6 = False  # 220 cols lack complete documentation in data dictionary

    if train_path.is_file() and test_path.is_file():
        tr_df = pd.read_csv(train_path)
        te_df = pd.read_csv(test_path)
        tr_ids = set(tr_df['participant_id'])
        te_ids = set(te_df['participant_id'])
        check7 = len(tr_ids.intersection(te_ids)) == 0 and len(tr_ids) + len(te_ids) == eligible_count
        check8 = len(te_ids) == 120 and len(tr_ids) == 477  # 80/20 stratified on 597

    gate_passed = all([check1, check2, check3, check4, check5, check6, check7, check8])

    return {
        'gate_passed': gate_passed,
        'check1_modelling_parquet_exists': check1,
        'check2_one_row_per_participant': check2,
        'check3_labels_valid_and_complete': check3,
        'check4_eligible_cohort_coverage': check4,
        'check5_features_unvitiated_and_provenance_known': check5,
        'check6_feature_definitions_documented': check6,
        'check7_split_unique_and_disjoint': check7,
        'check8_split_matches_protocol': check8,
        'corrupted_date_columns_count': corrupted_date_cols,
        'feature_table_row_count': feat_rows,
        'eligible_cohort_target': eligible_count,
        'verdict': 'GATE_FAILED_BLOCKED' if not gate_passed else 'GATE_PASSED',
        'rationale': (
            'modelling_dataset.parquet does not exist (blocked in Phase 4); candidate features_clean.csv '
            'covers only 495 of 597 eligible participants and contains shifted YYMMDD date corruption in 28 columns.'
        ) if not gate_passed else 'All checks passed.',
    }


def build_feature_inventory(df_train: pd.DataFrame) -> pd.DataFrame:
    """Build a comprehensive feature inventory table for all candidate predictor columns."""
    predictor_cols = [c for c in df_train.columns if c not in ['participant_id', 'parkinsons_label']]

    inventory = []
    for col in sorted(predictor_cols):
        s = df_train[col]
        dtype = str(s.dtype)
        n_null = int(s.isna().sum())
        null_pct = round(n_null / len(s) * 100, 2)
        n_unique = int(s.nunique(dropna=True))
        is_const = bool(n_unique <= 1)

        # Classify family
        if col.startswith('hold_'):
            family = 'Hold Time (HT)'
            unit = 'Milliseconds (ms)'
        elif col.startswith('lat1_'):
            family = 'Latency (L)'
            unit = 'Milliseconds (ms)'
        elif col.startswith('lat2_'):
            family = 'Flight Time / Rollover (FT)'
            unit = 'Milliseconds (ms)'
        else:
            family = 'Other / Transition'
            unit = 'Unspecified'

        # Status
        issues = []
        if pd.api.types.is_numeric_dtype(s) and (s > 100000).any():
            issues.append('DATE_CORRUPTED')
        if col.startswith('lat2_min') and s.min() >= 0:
            issues.append('DESTROYED_ROLLOVER')
        if n_null > 0:
            issues.append('CONTAINS_NULLS')
        status = ';'.join(issues) if issues else 'VERIFIED_NUMERIC'

        inventory.append({
            'feature_name': col,
            'family': family,
            'dtype': dtype,
            'unit': unit,
            'missing_count': n_null,
            'missing_pct': null_pct,
            'n_unique': n_unique,
            'is_constant': is_const,
            'integrity_status': status,
        })

    return pd.DataFrame(inventory)


def compute_missingness_summary(df_train: pd.DataFrame) -> pd.DataFrame:
    """Compute missingness summary for training predictors."""
    predictor_cols = [c for c in df_train.columns if c not in ['participant_id', 'parkinsons_label']]
    missing_data = []

    for col in predictor_cols:
        s = df_train[col]
        n_missing = int(s.isna().sum())
        pct_missing = round(n_missing / len(s) * 100, 3)
        n_inf = int(np.isinf(s).sum()) if pd.api.types.is_numeric_dtype(s) else 0

        missing_data.append({
            'feature_name': col,
            'missing_count': n_missing,
            'missing_percentage': pct_missing,
            'infinite_count': n_inf,
            'valid_count': len(s) - n_missing,
            'total_observations': len(s),
        })

    missing_df = pd.DataFrame(missing_data)
    return missing_df.sort_values(by='missing_count', ascending=False).reset_index(drop=True)


def compute_descriptive_statistics(df_train: pd.DataFrame) -> pd.DataFrame:
    """Compute comprehensive descriptive statistics across all numerical predictors in training set."""
    predictor_cols = [c for c in df_train.columns if c not in ['participant_id', 'parkinsons_label']]
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
            'feature_name': col,
            'count': int(len(s)),
            'mean': round(float(s.mean()), 4),
            'std': round(float(s.std()), 4),
            'min': round(float(s.min()), 4),
            'q25': round(q1, 4),
            'median': round(float(s.median()), 4),
            'q75': round(q3, 4),
            'max': round(float(s.max()), 4),
            'iqr': round(iqr, 4),
            'skewness': round(float(s.skew()), 4),
            'kurtosis': round(float(s.kurtosis()), 4),
            'n_unique': int(s.nunique()),
            'n_extreme_iqr': n_extreme,
            'pct_extreme_iqr': round(n_extreme / len(s) * 100, 2),
            'n_gt_100k': int((s > 100000).sum()),
        })

    return pd.DataFrame(stats_rows)


def screen_outliers_and_anomalies(df_train: pd.DataFrame) -> pd.DataFrame:
    """Categorize and screen candidate anomalies into statistical, semantic, and date-corrupted."""
    predictor_cols = [c for c in df_train.columns if c not in ['participant_id', 'parkinsons_label']]
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
            anomaly_class = 'KNOWN_DATE_CORRUPTION_ARTIFACT'
            severity = 'FATAL_BLOCKER'
            threshold = 'value > 100,000'
            rec = 'Must re-aggregate from Phase 2 interim Parquet in Phase 6.'
            n_flagged = n_gt_100k
        elif col.startswith('lat2_min') and s.min() >= 0:
            anomaly_class = 'SEMANTIC_VIOLATION_DESTROYED_ROLLOVER'
            severity = 'HIGH'
            threshold = 'min >= 0 ms (negative rollover missing)'
            rec = 'Must preserve negative flight times in Phase 6.'
            n_flagged = len(s)
        elif n_extreme_iqr > 0:
            anomaly_class = 'STATISTICAL_EXTREME_VALUE'
            severity = 'LOW_EXPLORATORY'
            threshold = f'outside [{round(q1 - 1.5 * iqr, 1)}, {round(q3 + 1.5 * iqr, 1)}]'
            rec = 'Evaluate robust scaling or tree models in Phase 6.'
            n_flagged = n_extreme_iqr
        else:
            continue

        screening_rows.append({
            'feature_name': col,
            'anomaly_classification': anomaly_class,
            'severity_level': severity,
            'flagged_observations_count': n_flagged,
            'screening_threshold': threshold,
            'observed_min': round(float(s.min()), 2),
            'observed_max': round(float(s.max()), 2),
            'recommended_phase6_action': rec,
        })

    return pd.DataFrame(screening_rows)


def analyze_feature_correlations(df_train: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Analyze feature correlations on uncorrupted training features to identify collinear pairs."""
    # Exclude columns with date corruption (> 100k) from correlation calculation to prevent bias
    clean_cols = [
        c for c in df_train.columns
        if c not in ['participant_id', 'parkinsons_label']
        and pd.api.types.is_numeric_dtype(df_train[c])
        and (df_train[c] <= 100000).all()
        and df_train[c].notna().sum() > 300
    ]

    corr_matrix = df_train[clean_cols].corr(method='spearman')

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
                        'feature_1': col1,
                        'feature_2': col2,
                        'spearman_rho': round(float(r), 4),
                        'absolute_rho': round(abs(float(r)), 4),
                        'relationship_interpretation': 'Collinear central tendencies / scale metrics'
                        if col1.split('_')[0] == col2.split('_')[0]
                        else 'Cross-feature correlation',
                    })

    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(by='absolute_rho', ascending=False).reset_index(drop=True)
    return pairs_df


def compute_training_class_summary(df_train: pd.DataFrame) -> pd.DataFrame:
    """Compute distribution comparisons between PD and Control on representative training features."""
    rep_features = [
        'hold_mean', 'hold_median', 'hold_std', 'hold_iqr', 'hold_skew', 'hold_cv',
        'lat1_mean', 'lat1_median', 'lat1_std', 'lat1_iqr',
        'lat2_mean', 'lat2_median', 'lat2_std',
    ]
    rep_features = [f for f in rep_features if f in df_train.columns]

    class_rows = []
    for f in rep_features:
        s0 = df_train[df_train['parkinsons_label'] == 0][f].dropna()
        s1 = df_train[df_train['parkinsons_label'] == 1][f].dropna()

        class_rows.append({
            'feature_name': f,
            'control_count': len(s0),
            'control_mean': round(float(s0.mean()), 3),
            'control_median': round(float(s0.median()), 3),
            'control_iqr': round(float(s0.quantile(0.75) - s0.quantile(0.25)), 3),
            'pd_count': len(s1),
            'pd_mean': round(float(s1.mean()), 3),
            'pd_median': round(float(s1.median()), 3),
            'pd_iqr': round(float(s1.quantile(0.75) - s1.quantile(0.25)), 3),
            'median_diff_pd_minus_ctrl': round(float(s1.median() - s0.median()), 3),
        })

    return pd.DataFrame(class_rows)


def generate_eda_figures(df_train: pd.DataFrame, output_dir: Path | str) -> list[str]:
    """Generate and save six high-quality, readable diagnostic EDA figures."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_figures = []

    # Palette setup
    sns.set_theme(style='whitegrid', font_scale=1.0)
    palette = {'Control': '#2b5c8f', 'Parkinson': '#d95f02'}

    # 1. Missingness Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    pred_cols = [c for c in df_train.columns if c not in ['participant_id', 'parkinsons_label']]
    null_counts = df_train[pred_cols].isna().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False).head(20)

    if not null_counts.empty:
        sns.barplot(x=null_counts.values, y=null_counts.index, color='#4575b4', ax=ax)
        ax.set_title('Top 20 Training Predictors with Missing Values (Transition Incompleteness)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Missing Participants (out of N=395)', fontsize=10)
        ax.set_ylabel('Feature Name', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'Zero Missing Values in Evaluated Features', ha='center', va='center')
    plt.tight_layout()
    fig1_path = out_dir / 'fig1_missingness_by_feature.png'
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    saved_figures.append(str(fig1_path))

    # 2. Hold Time Distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df_train['Group'] = df_train['parkinsons_label'].map({0: 'Control', 1: 'Parkinson'})

    sns.histplot(data=df_train, x='hold_mean', hue='Group', kde=True, palette=palette, ax=axes[0], bins=25)
    axes[0].set_title('Distribution of Key Hold Time (Mean) across Training Set', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Hold Time Mean (ms)', fontsize=10)
    axes[0].set_ylabel('Participant Count', fontsize=10)

    sns.histplot(data=df_train, x='hold_std', hue='Group', kde=True, palette=palette, ax=axes[1], bins=25)
    axes[1].set_title('Distribution of Key Hold Time Variability (Std Dev)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Hold Time Std Dev (ms)', fontsize=10)
    axes[1].set_ylabel('Participant Count', fontsize=10)
    plt.tight_layout()
    fig2_path = out_dir / 'fig2_hold_time_distributions.png'
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    saved_figures.append(str(fig2_path))

    # 3. Date Corruption Evidence Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=df_train, x='hold_mean', y='lat1_max', hue='Group', palette=palette, alpha=0.8, ax=axes[0])
    axes[0].axhline(100000, color='red', linestyle='--', label='YYMMDD Date Threshold (100k)')
    axes[0].set_title('lat1_max vs hold_mean: Shifted Date Values (>100,000)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Hold Time Mean (ms)', fontsize=10)
    axes[0].set_ylabel('lat1_max (ms)', fontsize=10)
    axes[0].legend(loc='upper left')

    sns.scatterplot(data=df_train, x='hold_mean', y='lat2_max', hue='Group', palette=palette, alpha=0.8, ax=axes[1])
    axes[1].axhline(100000, color='red', linestyle='--', label='YYMMDD Date Threshold (100k)')
    axes[1].set_title('lat2_max vs hold_mean: 95 Train Rows Corrupted with Dates', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Hold Time Mean (ms)', fontsize=10)
    axes[1].set_ylabel('lat2_max (ms)', fontsize=10)
    axes[1].legend(loc='upper left')
    plt.tight_layout()
    fig3_path = out_dir / 'fig3_date_corruption_evidence.png'
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    saved_figures.append(str(fig3_path))

    # 4. Destroyed Rollover Flight Time
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_train['lat2_min'].dropna(), bins=30, color='#e41a1c', ax=ax)
    ax.axvline(0, color='black', linestyle='--', label='Zero Flight Time Line')
    ax.set_title('Distribution of lat2_min: Truncation at >= 0 ms (Destroyed Rollover)', fontsize=12, fontweight='bold')
    ax.set_xlabel('lat2_min (ms) [All negative values coerced to NaN in legacy table]', fontsize=10)
    ax.set_ylabel('Participant Count', fontsize=10)
    ax.legend()
    plt.tight_layout()
    fig4_path = out_dir / 'fig4_destroyed_rollover_flight_time.png'
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    saved_figures.append(str(fig4_path))

    # 5. Correlation Heatmap on Uncorrupted Core Timing Features
    core_features = [
        'hold_mean', 'hold_std', 'hold_median', 'hold_iqr', 'hold_cv', 'hold_skew',
        'lat1_mean', 'lat1_median', 'lat1_std', 'lat1_iqr',
    ]
    core_features = [f for f in core_features if f in df_train.columns]
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_mat = df_train[core_features].corr(method='spearman')
    sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Spearman Correlation Matrix: Core Typing Dynamics (Train N=395)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig5_path = out_dir / 'fig5_correlation_heatmap.png'
    plt.savefig(fig5_path, dpi=200)
    plt.close()
    saved_figures.append(str(fig5_path))

    # 6. Label Comparison Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    sns.boxplot(data=df_train, x='Group', y='hold_mean', palette=palette, ax=axes[0], showmeans=True)
    axes[0].set_title('Key Hold Time Mean by Label', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Hold Time Mean (ms)', fontsize=10)

    sns.boxplot(data=df_train, x='Group', y='hold_std', palette=palette, ax=axes[1], showmeans=True)
    axes[1].set_title('Hold Time Std Dev (Variability) by Label', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Hold Time Std Dev (ms)', fontsize=10)

    sns.boxplot(data=df_train, x='Group', y='hold_cv', palette=palette, ax=axes[2], showmeans=True)
    axes[2].set_title('Hold Time Coefficient of Variation by Label', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Coefficient of Variation (CV)', fontsize=10)
    plt.tight_layout()
    fig6_path = out_dir / 'fig6_label_comparison_boxplots.png'
    plt.savefig(fig6_path, dpi=200)
    plt.close()
    saved_figures.append(str(fig6_path))

    return saved_figures
