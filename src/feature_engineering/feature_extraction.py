"""Feature extraction module for Parkinson's Disease typing characteristics.

Extracts participant-level typing aggregate features and quality metadata
from accepted, validated keystroke event records.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.feature_engineering.config import ExtractionConfig, FeatureEngineeringConfig
from src.feature_engineering.feature_schema import (
    ALLOWED_DIRECTIONS,
    ALLOWED_HANDS,
    get_predictor_columns,
)

logger = logging.getLogger(__name__)


def compute_timing_statistics(
    series: pd.Series,
    prefix: str,
    config: ExtractionConfig,
) -> dict[str, float]:
    """Compute overall descriptive statistics for a continuous timing variable."""
    clean = series.dropna()
    n = len(clean)

    if n == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_skew": np.nan,
            f"{prefix}_kurt": np.nan,
            f"{prefix}_cv": np.nan,
        }

    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=1)) if n > 1 else 0.0
    med_val = float(clean.median())
    min_val = float(clean.min())
    max_val = float(clean.max())
    q25 = float(clean.quantile(0.25))
    q75 = float(clean.quantile(0.75))
    iqr_val = q75 - q25

    # Skewness with minimum sample size check
    if n >= config.min_obs_skewness:
        skew_val = float(clean.skew())
    else:
        skew_val = np.nan

    # Kurtosis with minimum sample size check
    if n >= config.min_obs_kurtosis:
        kurt_val = float(clean.kurtosis())
    else:
        kurt_val = np.nan

    # Coefficient of variation: only defined meaningfully for strictly positive means
    if n >= config.min_obs_cv and mean_val > config.cv_mean_epsilon:
        cv_val = std_val / mean_val
    else:
        cv_val = np.nan

    return {
        f"{prefix}_mean": round(mean_val, 4),
        f"{prefix}_std": round(std_val, 4),
        f"{prefix}_median": round(med_val, 4),
        f"{prefix}_min": round(min_val, 4),
        f"{prefix}_max": round(max_val, 4),
        f"{prefix}_q25": round(q25, 4),
        f"{prefix}_q75": round(q75, 4),
        f"{prefix}_iqr": round(iqr_val, 4),
        f"{prefix}_skew": round(skew_val, 4) if not np.isnan(skew_val) else np.nan,
        f"{prefix}_kurt": round(kurt_val, 4) if not np.isnan(kurt_val) else np.nan,
        f"{prefix}_cv": round(cv_val, 4) if not np.isnan(cv_val) else np.nan,
    }


def compute_directional_timing(
    clean: pd.Series,
    prefix: str,
    direction: str,
) -> dict[str, float]:
    """Compute central tendency and spread for a direction-filtered timing subset."""
    n = len(clean)
    if n == 0:
        return {
            f"{prefix}_mean_{direction}": np.nan,
            f"{prefix}_std_{direction}": np.nan,
            f"{prefix}_median_{direction}": np.nan,
            f"{prefix}_iqr_{direction}": np.nan,
        }

    mean_val = float(clean.mean())
    std_val = float(clean.std(ddof=1)) if n > 1 else 0.0
    med_val = float(clean.median())
    q25 = float(clean.quantile(0.25))
    q75 = float(clean.quantile(0.75))
    iqr_val = q75 - q25

    return {
        f"{prefix}_mean_{direction}": round(mean_val, 4),
        f"{prefix}_std_{direction}": round(std_val, 4),
        f"{prefix}_median_{direction}": round(med_val, 4),
        f"{prefix}_iqr_{direction}": round(iqr_val, 4),
    }


def extract_participant_features(
    events_df: pd.DataFrame,
    config: ExtractionConfig,
) -> dict[str, Any]:
    """Extract all typing-derived predictor features for a single participant."""
    features: dict[str, Any] = {}

    # 1. Overall Timing Feature Families
    features.update(compute_timing_statistics(events_df["hold_ms"], "hold", config))
    features.update(compute_timing_statistics(events_df["latency_ms"], "latency", config))
    features.update(compute_timing_statistics(events_df["flight_ms"], "flight", config))

    # 2. Hand Relative Frequencies
    total_events = len(events_df)
    if total_events > 0:
        hand_counts = events_df["hand"].value_counts()
        for h in ALLOWED_HANDS:
            features[f"hand_freq_{h}"] = round(float(hand_counts.get(h, 0)) / total_events, 4)
    else:
        for h in ALLOWED_HANDS:
            features[f"hand_freq_{h}"] = np.nan

    # 3. Direction Relative Frequencies
    if total_events > 0:
        dir_counts = events_df["direction"].value_counts()
        for d in ALLOWED_DIRECTIONS:
            features[f"dir_freq_{d}"] = round(float(dir_counts.get(d, 0)) / total_events, 4)
    else:
        for d in ALLOWED_DIRECTIONS:
            features[f"dir_freq_{d}"] = np.nan

    # 4. Direction-Specific Timing Dynamics
    for d in ALLOWED_DIRECTIONS:
        dir_mask = events_df["direction"] == d
        subset = events_df[dir_mask]

        features.update(compute_directional_timing(subset["hold_ms"].dropna(), "hold", d))
        features.update(compute_directional_timing(subset["latency_ms"].dropna(), "latency", d))
        features.update(compute_directional_timing(subset["flight_ms"].dropna(), "flight", d))

    return features


def extract_participant_quality(
    events_df: pd.DataFrame,
    participant_id: str,
    n_files: int,
) -> dict[str, Any]:
    """Extract participant-level quality and temporal coverage metadata."""
    total_records = len(events_df)
    if total_records == 0:
        return {
            "participant_id": participant_id,
            "total_records": 0,
            "n_files": n_files,
            "n_event_dates": 0,
            "n_warned_records": 0,
            "n_w01_backward_timestamp": 0,
            "n_w02_duplicate_record": 0,
            "n_w04_extreme_hold": 0,
            "date_range_start": None,
            "date_range_end": None,
            "data_coverage_days": 0,
        }

    # Temporal metrics
    event_times = pd.to_datetime(events_df["event_time"].dropna())
    if not event_times.empty:
        start_time = str(event_times.min())
        end_time = str(event_times.max())
        n_dates = int(event_times.dt.date.nunique())
        coverage_days = int((event_times.max() - event_times.min()).days) + 1
    else:
        start_time = None
        end_time = None
        n_dates = 0
        coverage_days = 0

    # Warning metrics
    warn_series = events_df["warning_codes"].dropna()
    n_warned = int((warn_series != "").sum())
    n_w01 = int(warn_series.str.contains("W01_BACKWARD_TIMESTAMP", regex=False).sum())
    n_w02 = int(warn_series.str.contains("W02_DUPLICATE_RECORD", regex=False).sum())
    n_w04 = int(warn_series.str.contains("W04_EXTREME_HOLD", regex=False).sum())

    return {
        "participant_id": participant_id,
        "total_records": total_records,
        "n_files": n_files,
        "n_event_dates": n_dates,
        "n_warned_records": n_warned,
        "n_w01_backward_timestamp": n_w01,
        "n_w02_duplicate_record": n_w02,
        "n_w04_extreme_hold": n_w04,
        "date_range_start": start_time,
        "date_range_end": end_time,
        "data_coverage_days": coverage_days,
    }


def aggregate_cohort_features(
    accepted_dir: Path | str,
    eligible_ids: list[str],
    labels_df: pd.DataFrame,
    config: FeatureEngineeringConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate all accepted records for the eligible cohort into participant tables."""
    dir_path = Path(accepted_dir)
    eligible_set = set(eligible_ids)

    # Group files by participant key
    participant_files: dict[str, list[Path]] = defaultdict(list)
    for p_file in dir_path.glob("*.parquet"):
        key = p_file.name.split("_")[0]
        if key in eligible_set:
            participant_files[key].append(p_file)

    logger.info(
        "Aggregating features for %d eligible participants across %d monthly files",
        len(eligible_set),
        sum(len(v) for v in participant_files.values()),
    )

    # Map labels for eligible participants
    eligible_labels = labels_df[labels_df["user_key"].isin(eligible_set)].dropna(subset=["label"])
    label_map = dict(zip(eligible_labels["user_key"], eligible_labels["label"].astype(int)))

    feature_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    columns_to_load = [
        "user_key", "event_time", "hand", "hold_ms",
        "direction", "latency_ms", "flight_ms", "warning_codes"
    ]

    for participant_id in sorted(eligible_set):
        files = participant_files.get(participant_id, [])
        if not files:
            logger.warning("Participant %s has no accepted parquet files", participant_id)
            continue

        # Load and concatenate monthly parquet files for this participant
        events_list = [pd.read_parquet(f, columns=columns_to_load) for f in files]
        df_participant = pd.concat(events_list, ignore_index=True)

        # 1. Extract predictors
        p_features = extract_participant_features(df_participant, config.extraction)
        p_features["participant_id"] = participant_id
        p_features["parkinsons_label"] = label_map[participant_id]
        feature_rows.append(p_features)

        # 2. Extract quality metadata
        p_quality = extract_participant_quality(df_participant, participant_id, len(files))
        quality_rows.append(p_quality)

    df_features = pd.DataFrame(feature_rows)
    df_quality = pd.DataFrame(quality_rows)

    # Reorder columns deterministically: participant_id, parkinsons_label, then sorted predictors
    predictor_cols = get_predictor_columns()
    ordered_cols = ["participant_id", "parkinsons_label"] + predictor_cols
    df_features = df_features[ordered_cols].sort_values(by="participant_id").reset_index(drop=True)
    df_quality = df_quality.sort_values(by="participant_id").reset_index(drop=True)

    return df_features, df_quality
