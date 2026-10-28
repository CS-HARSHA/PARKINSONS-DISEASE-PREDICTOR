"""WP3: Canonical MACD and simple timing feature definitions."""
from __future__ import annotations
import numpy as np

def _finite(x): return np.asarray(x,dtype=float)[np.isfinite(np.asarray(x,dtype=float))]
def macd(hold_times, ql=0.6, qh=1.0):
    x=_finite(hold_times)
    if x.size<2: return float('nan')
    lo,hi=np.quantile(x,[ql,qh],method='linear')
    keep=(x[:-1]>=lo)&(x[:-1]<=hi)&(x[1:]>=lo)&(x[1:]<=hi)
    return float(np.mean(np.abs(x[1:][keep]-x[:-1][keep]))) if np.any(keep) else float('nan')
def macd_with_count(hold_times,ql=0.6,qh=1.0):
    x=_finite(hold_times)
    if x.size<2:return float('nan'),0
    lo,hi=np.quantile(x,[ql,qh],method='linear'); keep=(x[:-1]>=lo)&(x[:-1]<=hi)&(x[1:]>=lo)&(x[1:]<=hi)
    return (float(np.mean(np.abs(np.diff(x)[keep]))),int(keep.sum())) if np.any(keep) else (float('nan'),0)
def mean_hold(x): return float(np.mean(_finite(x)))
def stdev(x): return float(np.std(_finite(x),ddof=0))
def macd_log(x,ql=0.6,qh=1.0): return macd(np.log1p(_finite(x)),ql,qh)
def macd_norm(x,ql=0.6,qh=1.0):
    a=_finite(x); m=np.mean(a); return macd(a/m,ql,qh) if m>0 else float('nan')
