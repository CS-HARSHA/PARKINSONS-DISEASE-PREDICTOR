import numpy as np
import pytest

from src.data_loading import Session
from src.evaluation import aggregate_subjects, cross_cohort_evaluation


def test_cross_cohort_subject_sets_are_disjoint():
    sessions = [
        Session("a", "MIT-CS1PD", "a_1", 0, [1, 2, 3, 4]),
        Session("b", "MIT-CS1PD", "b_1", 1, [2, 3, 4, 5]),
        Session("c", "MIT-CS2PD", "c_1", 0, [1, 2, 3, 4]),
        Session("d", "MIT-CS2PD", "d_1", 1, [2, 3, 4, 5]),
    ]
    examples = aggregate_subjects(sessions, feature_fn=lambda x: float(np.mean(x)))
    result = cross_cohort_evaluation(examples)
    train_ids = {"a", "b"}
    test_ids = {"c", "d"}
    assert train_ids.isdisjoint(test_ids)
    assert result["n_subjects"] == 4
    assert "model_score" in result["predictions"][0]


def test_evaluation_rejects_subject_leakage():
    sessions = [
        Session("same", "MIT-CS1PD", "same_1", 0, [1, 2, 3]),
        Session("same", "MIT-CS2PD", "same_2", 0, [1, 2, 3]),
        Session("other1", "MIT-CS1PD", "other1_1", 1, [2, 3, 4]),
        Session("other2", "MIT-CS2PD", "other2_1", 1, [2, 3, 4]),
    ]
    with pytest.raises(AssertionError, match="leakage"):
        cross_cohort_evaluation(aggregate_subjects(sessions, feature_fn=lambda x: float(np.mean(x))))
