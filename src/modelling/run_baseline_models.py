import hashlib
import json
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data/processed/features/participant_features_raw.parquet"
TRAIN_IDS_PATH = ROOT / "data/processed/splits/train_ids.csv"
OUTPUT_DIR = ROOT / "reports/modelling"
SEED = 42
N_SPLITS = 5
TARGET = "parkinsons_label"
ID_COLUMN = "participant_id"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_pipeline(classifier, scale):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def get_models():
    return {
        "Dummy Prior": make_pipeline(
            DummyClassifier(strategy="prior"), scale=False
        ),
        "Logistic Regression": make_pipeline(
            LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=1000
            ),
            scale=True,
        ),
        "Linear SVC": make_pipeline(
            LinearSVC(dual=False, max_iter=2000, random_state=SEED),
            scale=True,
        ),
        "RBF SVC": make_pipeline(
            SVC(kernel="rbf", probability=True, random_state=SEED),
            scale=True,
        ),
        "K-Nearest Neighbors": make_pipeline(
            KNeighborsClassifier(metric="euclidean"),
            scale=True,
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=200, random_state=SEED, n_jobs=-1
            ),
            scale=False,
        ),
    }


def positive_scores(fitted_pipeline, X):
    classifier = fitted_pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        probabilities = fitted_pipeline.predict_proba(X)
        classes = list(classifier.classes_)
        return probabilities[:, classes.index(1)]
    if hasattr(classifier, "decision_function"):
        scores = fitted_pipeline.decision_function(X)
        classes = list(classifier.classes_)
        return scores if classes.index(1) == 1 else -scores
    return None


def evaluate_fold(y_true, y_pred, scores):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "specificity": specificity,
        "precision": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": (
            roc_auc_score(y_true, scores)
            if scores is not None and len(np.unique(y_true)) == 2
            else float("nan")
        ),
        "average_precision": (
            average_precision_score(y_true, scores)
            if scores is not None
            else float("nan")
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def main():
    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(f"Feature table not found: {FEATURES_PATH}")
    if not TRAIN_IDS_PATH.is_file():
        raise FileNotFoundError(f"Training ID file not found: {TRAIN_IDS_PATH}")

    features = pd.read_parquet(FEATURES_PATH)
    train_ids = pd.read_csv(TRAIN_IDS_PATH)

    required_columns = {ID_COLUMN, TARGET}
    missing_columns = required_columns - set(features.columns)
    if missing_columns:
        raise ValueError(f"Feature table is missing columns: {sorted(missing_columns)}")
    if ID_COLUMN not in train_ids.columns:
        raise ValueError(f"Training split must contain a '{ID_COLUMN}' column.")

    if features[ID_COLUMN].duplicated().any():
        raise ValueError("Feature table contains duplicate participant IDs.")
    if train_ids[ID_COLUMN].duplicated().any():
        raise ValueError("Training split contains duplicate participant IDs.")

    training_ids = set(train_ids[ID_COLUMN].astype(str))
    features[ID_COLUMN] = features[ID_COLUMN].astype(str)
    train = features.loc[features[ID_COLUMN].isin(training_ids)].copy()

    if len(train) != len(training_ids):
        missing_ids = sorted(training_ids - set(train[ID_COLUMN]))
        raise ValueError(
            f"Some training IDs are missing from the feature table: {missing_ids[:10]}"
        )
    if len(train) != 477:
        raise ValueError(
            f"Expected 477 training participants based on the frozen split; found {len(train)}."
        )

    y = train[TARGET].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"Expected binary labels 0 and 1; found {sorted(y.unique())}")

    X = train.drop(columns=[ID_COLUMN, TARGET])
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric predictors found: {non_numeric}")
    if X.shape[1] != 153:
        raise ValueError(
            f"Expected 153 typing-derived predictors; found {X.shape[1]}."
        )
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError(
            "Predictors contain infinite values. Resolve this before modelling."
        )

    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED
    )
    models = get_models()
    fold_rows = []
    summary_rows = []

    for model_name, estimator in models.items():
        print(f"Evaluating {model_name}...")
        for fold_number, (fit_indices, valid_indices) in enumerate(
            cv.split(X, y), start=1
        ):
            fitted = clone(estimator)
            X_fit, X_valid = X.iloc[fit_indices], X.iloc[valid_indices]
            y_fit, y_valid = y.iloc[fit_indices], y.iloc[valid_indices]

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fitted.fit(X_fit, y_fit)

            y_pred = fitted.predict(X_valid)
            scores = positive_scores(fitted, X_valid)
            metrics = evaluate_fold(y_valid, y_pred, scores)

            fold_rows.append({
                "model": model_name,
                "fold": fold_number,
                "fit_participants": len(fit_indices),
                "validation_participants": len(valid_indices),
                **metrics,
                "warnings": " | ".join(
                    sorted({str(w.message) for w in caught})
                ),
            })

        model_folds = [
            row for row in fold_rows if row["model"] == model_name
        ]
        summary = {
            "model": model_name,
            "n_folds": len(model_folds),
            "predictor_count": X.shape[1],
        }
        for metric in [
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1_macro",
            "f1_weighted",
            "accuracy",
            "roc_auc",
            "average_precision",
        ]:
            values = np.array([row[metric] for row in model_folds], dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(values))
            summary[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
        summary_rows.append(summary)

    fold_results = pd.DataFrame(fold_rows)
    summary_results = pd.DataFrame(summary_rows).sort_values(
        by=["balanced_accuracy_mean", "roc_auc_mean"],
        ascending=False,
        kind="stable",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_path = OUTPUT_DIR / "baseline_cv_fold_results.csv"
    summary_path = OUTPUT_DIR / "baseline_cv_summary.csv"
    manifest_path = OUTPUT_DIR / "baseline_cv_manifest.json"

    fold_results.to_csv(fold_path, index=False)
    summary_results.to_csv(summary_path, index=False)

    manifest = {
        "phase": "Phase 7 — Baseline Model Comparison",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "random_seed": SEED,
        "cross_validation": {
            "method": "StratifiedKFold",
            "folds": N_SPLITS,
            "shuffle": True,
            "random_state": SEED,
            "data_used": "training split only",
            "training_participants": len(train),
            "held_out_test_set_loaded": False,
        },
        "cohort": {
            "parkinsons_count": int((y == 1).sum()),
            "control_count": int((y == 0).sum()),
        },
        "predictors": {
            "count": int(X.shape[1]),
            "columns": X.columns.tolist(),
            "participant_id_excluded": True,
            "target_excluded_from_predictors": True,
        },
        "input_sha256": {
            str(FEATURES_PATH.relative_to(ROOT)): sha256_file(FEATURES_PATH),
            str(TRAIN_IDS_PATH.relative_to(ROOT)): sha256_file(TRAIN_IDS_PATH),
        },
        "models": list(models.keys()),
        "scaling": (
            "StandardScaler inside fold pipeline for Logistic Regression, "
            "Linear SVC, RBF SVC and KNN; no scaler for Dummy Prior and Random Forest."
        ),
        "primary_metric": "balanced_accuracy",
        "ranking_rule": (
            "Sorted by mean 5-fold balanced accuracy, then mean ROC-AUC. "
            "Final protocol tie-break review remains required for differences within 0.005."
        ),
        "outputs": [
            str(fold_path.relative_to(ROOT)),
            str(summary_path.relative_to(ROOT)),
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\nBaseline cross-validation complete.")
    print("Training participants:", len(train))
    print("Predictors:", X.shape[1])
    print("\nSummary (sorted by mean balanced accuracy):")
    print(summary_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved:")
    print(f"  {fold_path.relative_to(ROOT)}")
    print(f"  {summary_path.relative_to(ROOT)}")
    print(f"  {manifest_path.relative_to(ROOT)}")
    print("\nThe held-out test set was not loaded or evaluated.")


if __name__ == "__main__":
    main()


