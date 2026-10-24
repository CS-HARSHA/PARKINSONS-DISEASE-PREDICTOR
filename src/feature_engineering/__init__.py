"""Phase 6 Feature Engineering & Preprocessing Package.

Provides deterministic typing-feature extraction, quality validation,
and reusable preprocessing components for the Parkinson's Disease Prediction project.
"""

from src.feature_engineering.config import (
    ExtractionConfig,
    FeatureEngineeringConfig,
    PreprocessingConfig,
)
from src.feature_engineering.feature_extraction import (
    aggregate_cohort_features,
    extract_participant_features,
    extract_participant_quality,
)
from src.feature_engineering.feature_schema import (
    ALLOWED_DIRECTIONS,
    ALLOWED_HANDS,
    FeatureDefinition,
    get_feature_definitions,
    get_predictor_columns,
    get_quality_columns,
    validate_predictor_allowlist,
)
from src.feature_engineering.preprocessing import build_preprocessing_pipeline
from src.feature_engineering.quality import (
    generate_feature_quality_summary,
    validate_feature_table,
    validate_quality_table,
)

__all__ = [
    "FeatureEngineeringConfig",
    "ExtractionConfig",
    "PreprocessingConfig",
    "FeatureDefinition",
    "ALLOWED_HANDS",
    "ALLOWED_DIRECTIONS",
    "get_feature_definitions",
    "get_predictor_columns",
    "get_quality_columns",
    "validate_predictor_allowlist",
    "extract_participant_features",
    "extract_participant_quality",
    "aggregate_cohort_features",
    "validate_feature_table",
    "validate_quality_table",
    "generate_feature_quality_summary",
    "build_preprocessing_pipeline",
]
