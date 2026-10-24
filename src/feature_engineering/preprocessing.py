"""Preprocessing module for Phase 6 Feature Engineering.

Provides reusable, unfitted scikit-learn preprocessing pipelines and transformers
for downstream model training (Phase 7).
"""

from __future__ import annotations

from typing import Literal

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from src.feature_engineering.feature_schema import get_predictor_columns


def build_preprocessing_pipeline(
    scaler: Literal["standard", "robust", "none"] = "standard",
    imputer_strategy: Literal["median", "mean"] = "median",
    add_indicator: bool = False,
    power_transform: bool = False,
    predictor_cols: list[str] | None = None,
) -> Pipeline:
    """Build a reusable, unfitted scikit-learn preprocessing pipeline.

    CRITICAL SAFEGUARD:
    This pipeline must remain UNFITTED in Phase 6. It is designed to be fitted
    strictly within cross-validation training folds during Phase 7 to prevent data leakage.

    Parameters
    ----------
    scaler : {"standard", "robust", "none"}
        Scaling method to apply after imputation.
    imputer_strategy : {"median", "mean"}
        Imputation strategy for missing values.
    add_indicator : bool
        Whether to append MissingIndicator binary columns.
    power_transform : bool
        Whether to apply Yeo-Johnson power transformation.
    predictor_cols : list[str] or None
        Specific subset of columns to transform. If None, uses default predictor allowlist.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted scikit-learn pipeline.
    """
    if predictor_cols is None:
        target_columns = get_predictor_columns()
    else:
        target_columns = list(predictor_cols)

    steps: list[tuple[str, Any]] = []

    # 1. Missing Value Imputation
    imputer = SimpleImputer(strategy=imputer_strategy, add_indicator=add_indicator)
    steps.append(("imputer", imputer))

    # 2. Optional Power Transformation (Yeo-Johnson)
    if power_transform:
        steps.append(("power_transform", PowerTransformer(method="yeo-johnson", standardize=False)))

    # 3. Feature Scaling
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaler == "none":
        pass  # Identity / No scaling (e.g., for Random Forest / Tree ensembles)
    else:
        raise ValueError(f"Unsupported scaler option: '{scaler}'. Choose from 'standard', 'robust', 'none'.")

    pipeline = Pipeline(steps)

    # Wrap in ColumnTransformer if specific columns are targeted
    column_transformer = ColumnTransformer(
        transformers=[("num_typing", pipeline, target_columns)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Enable pandas output if scikit-learn version supports it
    try:
        column_transformer.set_output(transform="pandas")
    except (AttributeError, ValueError):
        pass

    return column_transformer
