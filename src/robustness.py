"""WP9: Test-only browser timing perturbations and keystroke truncation helpers."""
from dataclasses import replace
import numpy as np

def transform_sessions(sessions, kind, value, seed=42):
    rng=np.random.default_rng(seed); out=[]
    for s in sessions:
        h=np.asarray(s.hold_times_s,float)
        if kind=='jitter': h=np.maximum(0.001,h+rng.normal(0,value,len(h)))
        elif kind=='quantize': h=np.round(h/value)*value
        elif kind=='offset': h=np.maximum(0.001,h+value)
        elif kind=='scale': h=h*value
        out.append(replace(s,hold_times_s=h.tolist()))
    return out

def truncate_sessions(sessions,n):
    out=[]; short=0
    for s in sessions:
        k=min(n,len(s.hold_times_s)); short += len(s.hold_times_s)<n
        out.append(replace(s,hold_times_s=s.hold_times_s[:k],press_times_s=s.press_times_s[:k],n_keystrokes=k))
    return out,short
