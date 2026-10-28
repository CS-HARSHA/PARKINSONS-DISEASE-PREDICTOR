"""WP2: Convert session records into reproducible session and subject feature tables."""
from __future__ import annotations
from typing import Iterable
import numpy as np
from .data_loading import Session
from .features import macd,macd_with_count,mean_hold,stdev

def session_feature_rows(sessions:Iterable[Session], feature_fn=macd):
    rows=[]
    for s in sessions:
        m,n=macd_with_count(s.hold_times_s)
        rows.append({'session_id':s.session_id,'subject_id':s.subject_id,'cohort':s.cohort,'label':s.label,'n_keystrokes':s.n_keystrokes,'mean_hold':mean_hold(s.hold_times_s),'std_hold':stdev(s.hold_times_s),'macd':float(feature_fn(np.asarray(s.hold_times_s))),'n_corridor_pairs':n})
    return rows
def subject_feature_rows(sessions:Iterable[Session],feature_fn=macd):
    grouped={}
    for s in sessions: grouped.setdefault((s.cohort,s.subject_id),[]).append(s)
    out=[]
    for (cohort,sid),group in sorted(grouped.items()):
        vals=[feature_fn(np.asarray(s.hold_times_s)) for s in group]; vals=[v for v in vals if np.isfinite(v)]
        if vals:
            out.append({'subject_id':sid,'cohort':cohort,'label':group[0].label,'n_sessions':len(group),'mean_hold':float(np.mean([mean_hold(s.hold_times_s) for s in group])),'std_hold':float(np.mean([stdev(s.hold_times_s) for s in group])),'macd':float(np.mean(vals))})
    return out
