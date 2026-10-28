"""WP4: Transparent single-feature baselines for mean hold and standard deviation."""
import numpy as np
from .features import mean_hold, stdev
from .models import make_classifier
class SimpleFeatureModel:
    def __init__(self, feature='mean_hold', classifier='logistic', seed=42): self.feature=feature; self.classifier_name=classifier; self.seed=seed
    def _x(self,sessions):
        fn=mean_hold if self.feature=='mean_hold' else stdev
        return np.asarray([[fn(s.hold_times_s)] for s in sessions],float)
    def fit(self,train_sessions): self.model=make_classifier(self.classifier_name,self.seed); self.model.fit(self._x(train_sessions),[s.label for s in train_sessions]); return self
    def scores(self,test_sessions):
        x=self._x(test_sessions); return self.model.decision_function(x) if hasattr(self.model,'decision_function') else self.model.predict_proba(x)[:,1]
class MeanHoldModel(SimpleFeatureModel):
    def __init__(self,**kw): super().__init__('mean_hold',**kw)
class StdHoldModel(SimpleFeatureModel):
    def __init__(self,**kw): super().__init__('std_hold',**kw)
