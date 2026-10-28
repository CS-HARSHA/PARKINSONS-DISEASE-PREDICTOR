"""WP7: Training-only FRESH feature selection and logistic scoring model."""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from .fresh_features import extract_all,select_top_m
class FreshModel:
    def __init__(self,m=5,cache=None,seed=42): self.m=m; self.cache=cache; self.seed=seed
    def _features(self,sessions): return self.cache if self.cache is not None else extract_all(sessions)
    def fit(self,train_sessions):
        self.train_ids=[s.session_id for s in train_sessions]; full=self._features(train_sessions); train_rows=full.set_index('session_id').loc[self.train_ids].reset_index(); self.cols=select_top_m(train_rows,[s.label for s in train_sessions],self.m); x=train_rows.set_index('session_id').loc[self.train_ids,self.cols].fillna(0).to_numpy(); self.scaler=StandardScaler().fit(x); self.model=LogisticRegression(random_state=self.seed,max_iter=2000).fit(self.scaler.transform(x),[s.label for s in train_sessions]); return self
    def scores(self,test_sessions):
        full=self._features(test_sessions); x=full.set_index('session_id').reindex([s.session_id for s in test_sessions])[self.cols].fillna(0).to_numpy(); return self.model.decision_function(self.scaler.transform(x))
