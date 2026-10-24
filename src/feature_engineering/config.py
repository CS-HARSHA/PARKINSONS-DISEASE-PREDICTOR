"""Configuration module for Phase 6 Feature Engineering and Preprocessing.

Loads and validates settings from TOML configuration or sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore


@dataclass(frozen=True)
class ExtractionConfig:
    """Settings controlling feature extraction safeguards and categories."""

    min_obs_skewness: int = 20
    min_obs_kurtosis: int = 30
    min_obs_cv: int = 10
    cv_mean_epsilon: float = 1e-6
    max_hold_ms: float = 10000.0
    max_latency_ms: float = 100000.0
    allowed_hands: tuple[str, ...] = ("L", "R", "S")
    allowed_directions: tuple[str, ...] = (
        "LL", "LR", "RL", "RR",
        "LS", "RS", "SL", "SR", "SS"
    )


@dataclass(frozen=True)
class PreprocessingConfig:
    """Settings controlling preprocessing transformers for downstream models."""

    default_imputer_strategy: str = "median"
    default_add_indicator: bool = False
    default_scaler: str = "standard"  # "standard", "robust", "none"


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Top-level configuration for Phase 6 feature engineering pipeline."""

    repo_root: Path
    interim_accepted_dir: Path
    participant_labels_path: Path
    participant_inventory_path: Path
    train_ids_path: Path
    test_ids_path: Path
    features_output_dir: Path
    reports_dir: Path
    docs_dir: Path
    expected_eligible_count: int = 597
    expected_train_count: int = 477
    expected_test_count: int = 120
    raw_features_file: str = "participant_features_raw.parquet"
    quality_features_file: str = "participant_feature_quality.parquet"
    manifest_file: str = "feature_manifest.json"
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    @classmethod
    def load(cls, config_path: Path | str | None = None, repo_root: Path | str | None = None) -> FeatureEngineeringConfig:
        """Load configuration from a TOML file or resolve defaults relative to repo_root."""
        if repo_root is None:
            # Assume file is located in src/feature_engineering/config.py
            resolved_root = Path(__file__).resolve().parent.parent.parent
        else:
            resolved_root = Path(repo_root).resolve()

        if config_path is None:
            resolved_config_path = resolved_root / "configs" / "feature_engineering.toml"
        else:
            resolved_config_path = Path(config_path)
            if not resolved_config_path.is_absolute():
                resolved_config_path = resolved_root / resolved_config_path

        data: dict[str, Any] = {}
        if resolved_config_path.is_file():
            with open(resolved_config_path, "rb") as f:
                data = tomllib.load(f)

        paths = data.get("paths", {})
        cohort = data.get("cohort", {})
        ext_data = data.get("extraction", {})
        prep_data = data.get("preprocessing", {})

        extraction = ExtractionConfig(
            min_obs_skewness=ext_data.get("min_obs_skewness", 20),
            min_obs_kurtosis=ext_data.get("min_obs_kurtosis", 30),
            min_obs_cv=ext_data.get("min_obs_cv", 10),
            cv_mean_epsilon=ext_data.get("cv_mean_epsilon", 1e-6),
            max_hold_ms=ext_data.get("max_hold_ms", 10000.0),
            max_latency_ms=ext_data.get("max_latency_ms", 100000.0),
            allowed_hands=tuple(ext_data.get("allowed_hands", ["L", "R", "S"])),
            allowed_directions=tuple(
                ext_data.get("allowed_directions", [
                    "LL", "LR", "RL", "RR",
                    "LS", "RS", "SL", "SR", "SS"
                ])
            ),
        )

        preprocessing = PreprocessingConfig(
            default_imputer_strategy=prep_data.get("default_imputer_strategy", "median"),
            default_add_indicator=prep_data.get("default_add_indicator", False),
            default_scaler=prep_data.get("default_scaler", "standard"),
        )

        return cls(
            repo_root=resolved_root,
            interim_accepted_dir=resolved_root / paths.get("interim_accepted_dir", "data/interim/accepted"),
            participant_labels_path=resolved_root / paths.get("participant_labels_path", "data/interim/participant_labels.csv"),
            participant_inventory_path=resolved_root / paths.get("participant_inventory_path", "data/interim/participant_inventory.csv"),
            train_ids_path=resolved_root / paths.get("train_ids_path", "data/processed/splits/train_ids.csv"),
            test_ids_path=resolved_root / paths.get("test_ids_path", "data/processed/splits/test_ids.csv"),
            features_output_dir=resolved_root / paths.get("features_output_dir", "data/processed/features"),
            reports_dir=resolved_root / paths.get("reports_dir", "reports/feature_engineering"),
            docs_dir=resolved_root / paths.get("docs_dir", "docs"),
            expected_eligible_count=cohort.get("expected_eligible_count", 597),
            expected_train_count=cohort.get("expected_train_count", 477),
            expected_test_count=cohort.get("expected_test_count", 120),
            raw_features_file=paths.get("raw_features_file", "participant_features_raw.parquet"),
            quality_features_file=paths.get("quality_features_file", "participant_feature_quality.parquet"),
            manifest_file=paths.get("manifest_file", "feature_manifest.json"),
            extraction=extraction,
            preprocessing=preprocessing,
        )
