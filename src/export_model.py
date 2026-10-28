"""WP10: Export the transparent MACD logistic model for browser inference."""
from datetime import datetime,timezone
import json,numpy as np
from sklearn.linear_model import LogisticRegression
from . import config
from .features import macd
def export_model(sessions,out_path=config.ROOT/'models'/'macd_model.json'):
    vals=np.asarray([macd(s.hold_times_s) for s in sessions],float); keep=np.isfinite(vals); x=vals[keep].reshape(-1,1); y=np.asarray([s.label for s in sessions])[keep]
    model=LogisticRegression(random_state=config.SEED,max_iter=2000).fit(x,y); stats={k:float(v) for k,v in zip(['min','p1','p5','median','p95','p99','max'],np.percentile(x[:,0],[0,1,5,50,95,99,100]))}
    payload={'model_type':'logistic_regression','feature':'macd','ql':config.CORRIDOR_LOW,'qh':config.CORRIDOR_HIGH,'unit':'seconds','min_corridor_pairs':config.MIN_CORRIDOR_PAIRS,'coef':float(model.coef_[0,0]),'intercept':float(model.intercept_[0]),'coefficient':float(model.coef_[0,0]),'training_stats':stats,'training_feature_min':stats['min'],'training_feature_max':stats['max'],'training_feature_median':stats['median'],'min_keystrokes':config.MIN_SESSION_KEYSTROKES,'trained_on':'all processed sessions, subject labels','created_utc':datetime.now(timezone.utc).isoformat(),'paper_reference':'Milne, Farrahi & Nicolaou, Less is More','disclaimer':'Research screening demo only; not a medical diagnosis.'}
    out_path.parent.mkdir(parents=True,exist_ok=True); out_path.write_text(json.dumps(payload,indent=2)+'\n'); return payload
def export_logistic_model(model,features,output,feature_name='macd'):
    x=np.asarray(features,float); payload={'model_type':'logistic_regression','feature':feature_name,'coefficient':float(model.coef_[0,0]),'intercept':float(model.intercept_[0]),'corridor_low':config.CORRIDOR_LOW,'corridor_high':config.CORRIDOR_HIGH,'training_feature_min':float(np.nanmin(x)),'training_feature_max':float(np.nanmax(x)),'training_feature_median':float(np.nanmedian(x)),'disclaimer':'Research screening demo only; not a medical diagnosis.'}; output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2)+'\n')
