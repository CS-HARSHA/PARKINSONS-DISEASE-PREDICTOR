"""WP6: Leakage-safe tsfresh extraction, imputation, and training-only selection."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

def extract_all(sessions):
    try:
        from tsfresh import extract_features
        from tsfresh.feature_extraction import ComprehensiveFCParameters
        rows=[]
        for s in sessions:
            for i,h in enumerate(s.hold_times_s): rows.append({'id':s.session_id,'time':i,'value':h})
        long=pd.DataFrame(rows); out=extract_features(long,column_id='id',column_sort='time',column_value='value',default_fc_parameters=ComprehensiveFCParameters(),disable_progressbar=True)
        out=out.replace([np.inf,-np.inf],np.nan).fillna(0.0); out.index.name='session_id'; return out.reset_index()
    except ImportError:
        # A readable fallback keeps the pipeline runnable if optional tsfresh is absent.
        rows=[]
        for s in sessions:
            x=np.asarray(s.hold_times_s,float); rows.append({'session_id':s.session_id,'value__mean':float(x.mean()),'value__standard_deviation':float(x.std()),'value__median':float(np.median(x))})
        return pd.DataFrame(rows)
def rank_features(train_features,train_labels):
    from sklearn.feature_selection import mutual_info_classif
    x=train_features.drop(columns=['session_id'],errors='ignore').fillna(0); y=np.asarray(train_labels)
    vals=mutual_info_classif(x,y,random_state=42); return list(x.columns[np.argsort(-vals)])
def select_top_m(train_features,train_labels,m): return rank_features(train_features,train_labels)[:m]
