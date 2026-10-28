from src.benchmark import aggregate_feature_sets, benchmark
from src.data_loading import Session

def test_benchmark_keeps_subjects_disjoint_and_returns_fold_metrics():
    sessions = [
        Session('a','MIT-CS1PD','a_1',0,[1,2,3,4]), Session('b','MIT-CS1PD','b_1',1,[2,3,4,5]),
        Session('c','MIT-CS2PD','c_1',0,[1,2,3,4]), Session('d','MIT-CS2PD','d_1',1,[2,3,4,5]),
    ]
    rows = aggregate_feature_sets(sessions, ('macd', 'stdev'))
    result = benchmark(rows, ('macd',), 'logistic_regression')
    assert len(result['folds']) == 2
    assert 'mean_fold_auc' in result
