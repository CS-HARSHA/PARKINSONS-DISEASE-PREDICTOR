from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

from . import config
from .data_loading import Session
from .feature_sets import feature_vector


@dataclass
class SubjectFeatureRow:
    subject_id: str
    cohort: str
    label: int
    values: dict[str, float]


def aggregate_feature_sets(sessions: Iterable[Session], names: tuple[str, ...]) -> list[SubjectFeatureRow]:
    grouped: dict[tuple[str, str], list[Session]] = defaultdict(list)
    for session in sessions:
        grouped[(session.cohort, session.subject_id)].append(session)
    rows = []
    for (cohort, subject_id), group in sorted(grouped.items()):
        session_values = [feature_vector(np.asarray(s.hold_times), names) for s in group]
        values = {name: float(np.mean([v[name] for v in session_values])) for name in names}
        rows.append(SubjectFeatureRow(subject_id, cohort, group[0].label, values))
    return rows


def model_factories() -> dict[str, Callable[[], object]]:
    return {
        "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(random_state=config.RANDOM_SEED)),
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=config.RANDOM_SEED),
        "rbf_svm": lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=False, random_state=config.RANDOM_SEED)),
    }


def _score(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return model.decision_function(x)


def benchmark(rows: Iterable[SubjectFeatureRow], feature_names: tuple[str, ...], model_name: str) -> dict:
    rows = list(rows)
    by_cohort = {cohort: [r for r in rows if r.cohort == cohort] for cohort in config.COHORTS.values()}
    factory = model_factories()[model_name]
    fold_results = []
    pooled = []
    for train_cohort, test_cohort in ((config.COHORTS["early"], config.COHORTS["de_novo"]), (config.COHORTS["de_novo"], config.COHORTS["early"])):
        train, test = by_cohort[train_cohort], by_cohort[test_cohort]
        train_ids = {r.subject_id for r in train}; test_ids = {r.subject_id for r in test}
        if train_ids & test_ids:
            raise AssertionError("Subject leakage detected between train and test")
        model = factory()
        x_train = np.array([[r.values[n] for n in feature_names] for r in train])
        x_test = np.array([[r.values[n] for n in feature_names] for r in test])
        y_train = np.array([r.label for r in train]); y_test = np.array([r.label for r in test])
        model.fit(x_train, y_train)
        scores = _score(model, x_test)
        fold_auc = roc_auc_score(y_test, scores)
        fold_results.append({"train_cohort": train_cohort, "test_cohort": test_cohort, "auc": float(fold_auc), "n_test": len(test)})
        pooled.extend(zip(y_test.tolist(), scores.tolist()))
    pooled_y = np.array([p[0] for p in pooled]); pooled_scores = np.array([p[1] for p in pooled])
    return {"model": model_name, "features": list(feature_names), "folds": fold_results, "mean_fold_auc": float(np.mean([f["auc"] for f in fold_results])), "pooled_auc_descriptive": float(roc_auc_score(pooled_y, pooled_scores)), "note": "Fold AUCs are the primary comparable results. Pooled model outputs are descriptive because independently fitted models may have different score scales."}
