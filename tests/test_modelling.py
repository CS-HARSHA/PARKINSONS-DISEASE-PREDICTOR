"""Unit and integration tests for Phase 7 Baseline Modeling pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.modelling.run_baseline_models import (
    make_pipeline,
    get_models,
    positive_scores,
    evaluate_fold,
    sha256_file,
    ROOT,
    OUTPUT_DIR,
)


def test_get_models_structure():
    """Verify that get_models provides the 6 prospective architectures with correct scaling."""
    models = get_models()
    expected_names = [
        "Dummy Prior",
        "Logistic Regression",
        "Linear SVC",
        "RBF SVC",
        "K-Nearest Neighbors",
        "Random Forest",
    ]
    assert list(models.keys()) == expected_names

    # Imputer is present in all pipelines
    for name, pipeline in models.items():
        assert "imputer" in pipeline.named_steps
        assert pipeline.named_steps["imputer"].strategy == "median"

    # Scaler is present for linear and distance models, absent for Dummy and RF
    assert "scaler" in models["Logistic Regression"].named_steps
    assert "scaler" in models["Linear SVC"].named_steps
    assert "scaler" in models["RBF SVC"].named_steps
    assert "scaler" in models["K-Nearest Neighbors"].named_steps
    assert "scaler" not in models["Dummy Prior"].named_steps
    assert "scaler" not in models["Random Forest"].named_steps


def test_evaluate_fold_metrics():
    """Verify exact mathematical computation of primary and secondary classification metrics."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # 4 controls, 6 PD
    y_pred = np.array([0, 0, 1, 0, 1, 1, 1, 1, 0, 1])  # TN=3, FP=1, FN=1, TP=5
    scores = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.8, 0.9, 0.85, 0.4, 0.95])

    metrics = evaluate_fold(y_true, y_pred, scores)

    assert metrics["tn"] == 3
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 5

    # Sensitivity = 5 / 6
    assert np.isclose(metrics["sensitivity"], 5.0 / 6.0)
    # Specificity = 3 / 4
    assert np.isclose(metrics["specificity"], 3.0 / 4.0)
    # Balanced Accuracy = (5/6 + 3/4) / 2 = (10/12 + 9/12) / 4 = 19/24 ≈ 0.791667
    assert np.isclose(metrics["balanced_accuracy"], 0.5 * (5.0 / 6.0 + 3.0 / 4.0))
    # Precision = 5 / 6
    assert np.isclose(metrics["precision"], 5.0 / 6.0)
    # Accuracy = 8 / 10 = 0.8
    assert np.isclose(metrics["accuracy"], 0.8)
    assert not np.isnan(metrics["roc_auc"])
    assert not np.isnan(metrics["average_precision"])


def test_positive_scores_predict_proba():
    """Verify score extraction for estimators implementing predict_proba."""
    X = np.array([[1.0, 2.0], [2.0, 1.0], [0.5, 0.8], [3.0, 2.5]])
    y = np.array([0, 1, 0, 1])
    pipe = make_pipeline(LogisticRegression(), scale=True)
    pipe.fit(X, y)

    scores = positive_scores(pipe, X)
    assert scores is not None
    assert len(scores) == 4
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_positive_scores_decision_function():
    """Verify score extraction for estimators implementing decision_function (e.g. LinearSVC)."""
    X = np.array([[1.0, 2.0], [2.0, 1.0], [0.5, 0.8], [3.0, 2.5]])
    y = np.array([0, 1, 0, 1])
    pipe = make_pipeline(LinearSVC(dual=False, random_state=42), scale=True)
    pipe.fit(X, y)

    scores = positive_scores(pipe, X)
    assert scores is not None
    assert len(scores) == 4


def test_sha256_file(tmp_path: Path):
    """Verify SHA-256 calculation on known content."""
    f = tmp_path / "sample.txt"
    f.write_text("hello world", encoding="utf-8")
    expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    assert sha256_file(f) == expected


def test_baseline_cv_artifacts_integrity():
    """Verify that baseline cross-validation summary and manifest exist and adhere to protocol."""
    summary_path = OUTPUT_DIR / "baseline_cv_summary.csv"
    manifest_path = OUTPUT_DIR / "baseline_cv_manifest.json"

    assert summary_path.is_file(), f"Missing {summary_path}"
    assert manifest_path.is_file(), f"Missing {manifest_path}"

    df = pd.read_csv(summary_path)
    assert len(df) == 6
    assert "model" in df.columns
    assert "balanced_accuracy_mean" in df.columns
    assert "roc_auc_mean" in df.columns

    # Verify top ranked model is Random Forest
    top_model = df.iloc[0]["model"]
    assert top_model == "Random Forest"

    # Verify Dummy Prior baseline floor is exactly 0.50 balanced accuracy
    dummy_row = df[df["model"] == "Dummy Prior"].iloc[0]
    assert np.isclose(dummy_row["balanced_accuracy_mean"], 0.50)
    assert np.isclose(dummy_row["sensitivity_mean"], 1.0)
    assert np.isclose(dummy_row["specificity_mean"], 0.0)
