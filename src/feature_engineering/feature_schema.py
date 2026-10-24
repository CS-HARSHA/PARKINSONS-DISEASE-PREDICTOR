"""Feature schema and definitions for typing characteristics.

Defines the explicit taxonomy, allowlists, units, and documentation metadata
for participant-level typing features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FeatureDefinition:
    """Metadata specification for a single engineered feature."""

    name: str
    family: str
    unit: str
    description: str
    is_predictor: bool
    domain_min: float | None = None
    domain_max: float | None = None


# Core directions and hands per Phase 2 log specification
ALLOWED_HANDS = ("L", "R", "S")
ALLOWED_DIRECTIONS = (
    "LL", "LR", "RL", "RR",
    "LS", "RS", "SL", "SR", "SS"
)

# Statistical metrics calculated for overall timing variables
TIMING_STATS = (
    "mean", "std", "median", "min", "max",
    "q25", "q75", "iqr", "skew", "kurt", "cv"
)

# Direction-specific statistical metrics
DIRECTION_TIMING_STATS = ("mean", "std", "median", "iqr")


def get_feature_definitions() -> list[FeatureDefinition]:
    """Generate the complete, authoritative catalog of all engineered features."""
    definitions: list[FeatureDefinition] = []

    # 1. Overall Key Hold Time (HT)
    for stat in TIMING_STATS:
        unit = "ratio" if stat == "cv" else ("unitless" if stat in ("skew", "kurt") else "ms")
        definitions.append(
            FeatureDefinition(
                name=f"hold_{stat}",
                family="Hold Time (HT)",
                unit=unit,
                description=f"Key depression duration overall {stat} across all keystroke events",
                is_predictor=True,
                domain_min=0.0 if stat not in ("skew", "kurt") else None,
            )
        )

    # 2. Overall Latency Time (L)
    for stat in TIMING_STATS:
        unit = "ratio" if stat == "cv" else ("unitless" if stat in ("skew", "kurt") else "ms")
        definitions.append(
            FeatureDefinition(
                name=f"latency_{stat}",
                family="Latency (L)",
                unit=unit,
                description=f"Inter-key press interval overall {stat} across all keystroke events",
                is_predictor=True,
                domain_min=0.0 if stat not in ("skew", "kurt") else None,
            )
        )

    # 3. Overall Flight Time (FT) - Rollover Dynamics
    for stat in TIMING_STATS:
        unit = "ratio" if stat == "cv" else ("unitless" if stat in ("skew", "kurt") else "ms")
        definitions.append(
            FeatureDefinition(
                name=f"flight_{stat}",
                family="Flight Time (FT)",
                unit=unit,
                description=f"Key release to next key press interval overall {stat} (negative values valid)",
                is_predictor=True,
            )
        )

    # 4. Hand Relative Frequencies
    for hand in ALLOWED_HANDS:
        definitions.append(
            FeatureDefinition(
                name=f"hand_freq_{hand}",
                family="Hand Dynamics",
                unit="proportion [0, 1]",
                description=f"Proportion of keystrokes executed by hand category {hand}",
                is_predictor=True,
                domain_min=0.0,
                domain_max=1.0,
            )
        )

    # 5. Direction Relative Frequencies
    for direction in ALLOWED_DIRECTIONS:
        definitions.append(
            FeatureDefinition(
                name=f"dir_freq_{direction}",
                family="Transition Dynamics",
                unit="proportion [0, 1]",
                description=f"Proportion of transitions matching direction pattern {direction}",
                is_predictor=True,
                domain_min=0.0,
                domain_max=1.0,
            )
        )

    # 6. Direction-Specific Timing Dynamics
    for direction in ALLOWED_DIRECTIONS:
        for stat in DIRECTION_TIMING_STATS:
            # Hold time for direction
            definitions.append(
                FeatureDefinition(
                    name=f"hold_{stat}_{direction}",
                    family="Directional Hold Time",
                    unit="ms",
                    description=f"Key hold duration {stat} for {direction} transitions",
                    is_predictor=True,
                    domain_min=0.0,
                )
            )
            # Latency for direction
            definitions.append(
                FeatureDefinition(
                    name=f"latency_{stat}_{direction}",
                    family="Directional Latency",
                    unit="ms",
                    description=f"Inter-key latency {stat} for {direction} transitions",
                    is_predictor=True,
                    domain_min=0.0,
                )
            )
            # Flight time for direction
            definitions.append(
                FeatureDefinition(
                    name=f"flight_{stat}_{direction}",
                    family="Directional Flight Time",
                    unit="ms",
                    description=f"Flight duration {stat} for {direction} transitions (negative values valid)",
                    is_predictor=True,
                )
            )

    return definitions


def get_predictor_columns() -> list[str]:
    """Return the ordered list of approved predictor feature names."""
    return [f.name for f in get_feature_definitions() if f.is_predictor]


def get_quality_columns() -> list[str]:
    """Return the list of participant-level quality/coverage metadata columns."""
    return [
        "participant_id",
        "total_records",
        "n_files",
        "n_event_dates",
        "n_warned_records",
        "n_w01_backward_timestamp",
        "n_w02_duplicate_record",
        "n_w04_extreme_hold",
        "date_range_start",
        "date_range_end",
        "data_coverage_days",
    ]


def validate_predictor_allowlist(columns: list[str]) -> tuple[bool, list[str]]:
    """Verify that columns strictly exclude IDs, labels, and clinical/demographic covariates."""
    prohibited = {
        "participant_id", "user_key", "id", "parkinsons_label", "label", "parkinsons_raw",
        "age", "gender", "sex", "updrs", "tremor", "impact", "nhy", "years_since_dx",
        "levodopa", "daa", "maob", "other_meds", "file_count", "total_records", "n_files"
    }
    violations = [c for c in columns if c.lower() in prohibited]
    return (len(violations) == 0, violations)
