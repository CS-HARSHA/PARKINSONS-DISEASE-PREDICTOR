from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from . import config
from .data_loading import Session
from .features import macd
from .modeling import fit_logistic, predict_probability


@dataclass
class SubjectExample:
    subject_id: str
    cohort: str
    label: int
    feature: float


def aggregate_subjects(sessions: Iterable[Session], feature_fn=macd) -> list[SubjectExample]:
    grouped: dict[tuple[str, str], list[Session]] = {}
    for session in sessions:
        grouped.setdefault((session.cohort, session.subject_id), []).append(session)
    examples = []
    for (cohort, subject_id), rows in sorted(grouped.items()):
        values = [feature_fn(np.asarray(row.hold_times)) for row in rows]
        values = [v for v in values if np.isfinite(v)]
        if values:
            examples.append(SubjectExample(subject_id, cohort, rows[0].label, float(np.mean(values))))
    return examples


def youden_cutoff(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    index = int(np.argmax(tpr - fpr))
    return float(thresholds[index])


def bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray, reps: int = config.BOOTSTRAP_REPS,
                     seed: int = config.RANDOM_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(y_true)
    for _ in range(reps):
        indices = rng.integers(0, n, size=n)
        y = y_true[indices]
        if np.unique(y).size < 2:
            continue
        values.append(roc_auc_score(y, scores[indices]))
    if not values:
        return (float("nan"), float("nan"))
    return tuple(np.quantile(values, [0.025, 0.975]))


def cross_cohort_evaluation(examples: Iterable[SubjectExample]) -> dict:
    rows = list(examples)
    by_cohort = {name: [r for r in rows if r.cohort == cohort] for name, cohort in config.COHORTS.items()}
    predictions = []
    fold_details = []
    for train_name, test_name in (("early", "de_novo"), ("de_novo", "early")):
        train, test = by_cohort[train_name], by_cohort[test_name]
        train_subjects = {r.subject_id for r in train}
        test_subjects = {r.subject_id for r in test}
        if train_subjects & test_subjects:
            raise AssertionError("Subject leakage detected between train and test")
        # The paper fits on independent sessions and averages multiple session
        # scores per subject. SubjectExample is already the session average in
        # the current processed representation, so fit the stated one-feature
        # model here while retaining the common MACD score for pooled ROC.
        model = fit_logistic(np.array([r.feature for r in train]), np.array([r.label for r in train]))
        train_scores = predict_probability(model, np.array([r.feature for r in train]))
        cutoff = youden_cutoff(np.array([r.label for r in train]), train_scores)
        for row in test:
            model_score = float(predict_probability(model, np.array([row.feature]))[0])
            predictions.append({"subject_id": row.subject_id, "cohort": row.cohort, "label": row.label, "score": float(row.feature), "model_score": model_score, "fold": train_name})
        fold_details.append({"train_cohort": train_name, "test_cohort": test_name, "n_train": len(train), "n_test": len(test), "youden_cutoff_from_train_model_probability": cutoff})
    y = np.array([p["label"] for p in predictions])
    scores = np.array([p["score"] for p in predictions])
    pooled_cutoff = youden_cutoff(y, scores)
    predicted = (scores >= pooled_cutoff).astype(int)
    return {"n_subjects": len(predictions), "auc": float(roc_auc_score(y, scores)), "auc_ci_95_subject_bootstrap": bootstrap_auc_ci(y, scores), "youden_cutoff_pooled_descriptive": pooled_cutoff, "accuracy_at_pooled_descriptive_cutoff": float(accuracy_score(y, predicted)), "confusion_matrix_at_pooled_descriptive_cutoff": confusion_matrix(y, predicted).tolist(), "folds": fold_details, "scoring_note": "Pooled ROC uses the common MACD feature scale; fold-specific logistic probabilities are retained as model_score because their intercepts are fit separately.", "predictions": predictions}
