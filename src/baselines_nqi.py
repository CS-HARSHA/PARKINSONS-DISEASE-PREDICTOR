"""WP5: Approximate nQi-style window features and SVR ensemble baseline."""
import numpy as np
from sklearn.svm import LinearSVR
from . import config

def nqi_window_features(hold_times_s, press_times_s, window_s=90.0, min_keystrokes=30):
    h=np.asarray(hold_times_s,float); p=np.asarray([x if x is not None else np.nan for x in press_times_s],float)
    if len(p)!=len(h) or not np.isfinite(p).any(): p=np.arange(len(h),dtype=float)
    p=p-np.nanmin(p); windows=[]; end=np.nanmax(p) if len(p) else 0
    for start in np.arange(0,end+window_s,window_s):
        idx=(p>=start)&(p<start+window_s)&np.isfinite(h)
        vals=h[idx]
        if len(vals)<min_keystrokes: continue
        q1,q2,q3=np.quantile(vals,[.25,.5,.75]); iqr=q3-q1
        out=float(np.mean((vals<q1-1.5*iqr)|(vals>q3+1.5*iqr))) if iqr else 0.0
        skew=float((q2-q1)/(q3-q1)) if q3>q1 else 0.0
        flights=np.diff(p[idx]); flight=float(np.mean(flights)) if len(flights) else 0.0
        bins=[float(np.mean((vals>=lo)&(vals<hi))) for lo,hi in [(0,.125),(.125,.25),(.25,.375),(.375,.5)]]
        windows.append([out,skew,flight,*bins])
    return np.asarray(windows,float).reshape((-1,7))
class NqiModel:
    def __init__(self,n_members=200,seed=42): self.n_members=n_members; self.seed=seed
    def _windows(self,sessions):
        xs=[]; ys=[]; owners=[]
        for s in sessions:
            w=nqi_window_features(s.hold_times_s,s.press_times_s,config.NQI_WINDOW_SECONDS,config.NQI_MIN_KEYSTROKES)
            if len(w): xs.append(w); ys.extend([s.label]*len(w)); owners.extend([s.session_id]*len(w))
        return np.vstack(xs) if xs else np.empty((0,7)),np.asarray(ys),owners
    def fit(self,train_sessions):
        x,y,_=self._windows(train_sessions); self.models=[]; rng=np.random.default_rng(self.seed)
        if len(x)==0: raise ValueError('No nQi windows available')
        for _ in range(self.n_members):
            idx=rng.integers(0,len(x),len(x)); m=LinearSVR(C=1.0,epsilon=0.0,random_state=self.seed); m.fit(x[idx],y[idx]); self.models.append(m)
        return self
    def scores(self,test_sessions):
        out=[]
        for s in test_sessions:
            w=nqi_window_features(s.hold_times_s,s.press_times_s,config.NQI_WINDOW_SECONDS,config.NQI_MIN_KEYSTROKES)
            if len(w): out.append(float(np.median([m.predict(w).mean() for m in self.models])))
            else: out.append(float('nan'))
        return np.asarray(out)
