"""WP8: Shared model comparison and metric-table helpers."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from . import config
from .evaluation import cross_cohort_evaluate,save_result
class FeatureSessionModel:
    def __init__(self,feature_fns,classifier_factory): self.feature_fns=feature_fns; self.classifier_factory=classifier_factory
    def _x(self,sessions): return np.asarray([[fn(np.asarray(s.hold_times_s)) for fn in self.feature_fns] for s in sessions],float)
    def fit(self,train_sessions):
        self.model=self.classifier_factory(); x=self._x(train_sessions); self.model.fit(x,[s.label for s in train_sessions]); return self
    def scores(self,test_sessions):
        x=self._x(test_sessions)
        # For a one-feature monotone baseline, use the common feature scale for
        # pooled cross-cohort ranking; independently fitted intercepts otherwise
        # make pooled probabilities incomparable across directions.
        if len(self.feature_fns)==1: return x[:,0]
        return self.model.decision_function(x) if hasattr(self.model,'decision_function') else self.model.predict_proba(x)[:,1]
def evaluate_feature_set(name,features,sessions,classifier='logistic'):
    from .models import make_classifier
    return cross_cohort_evaluate(lambda:FeatureSessionModel(features,lambda:make_classifier(classifier,config.SEED)),sessions,name=name)
def collect_saved_results():
    out=[]
    for p in sorted(config.METRICS_DIR.glob('*.json')):
        try:
            d=json.loads(p.read_text())
            if 'auc_pooled' in d: out.append(d)
        except Exception: pass
    return out
