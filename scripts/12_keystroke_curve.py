"""WP9: Measure cross-cohort AUC as sessions are truncated to N keystrokes."""
from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import config
from src.data_loading import load_processed
from src.features import macd
from src.compare import FeatureSessionModel
from src.evaluation import cross_cohort_evaluate
from src.robustness import truncate_sessions

def main():
    base=load_processed(); out=[]
    for n in [50,100,150,200,300,500,1000,None]:
        s,short=truncate_sessions(base,n or 10**9); r=cross_cohort_evaluate(lambda:FeatureSessionModel([macd],lambda:__import__('sklearn.linear_model',fromlist=['LogisticRegression']).LogisticRegression(max_iter=2000,random_state=42)),s,name=f'curve_{n or "full"}'); out.append({'n_keystrokes':n or 'full','sessions_shorter_than_n':short,'auc_pooled':r.auc_pooled,'ci_low':r.ci_low,'ci_high':r.ci_high})
    (config.METRICS_DIR/'keystroke_curve.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
