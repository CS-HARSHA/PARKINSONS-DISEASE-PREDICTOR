"""WP3: Evaluate MACD corridor sensitivity without changing the primary setting."""
from pathlib import Path
import sys,csv,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sklearn.linear_model import LogisticRegression
from src import config
from src.data_loading import load_processed
from src.features import macd
from src.compare import FeatureSessionModel
from src.evaluation import cross_cohort_evaluate
def main():
    sessions=load_processed(); rows=[]
    for q in [0.0,.4,.5,.6,.7,.8]:
        fn=lambda x,q=q:macd(x,q,1.0); r=cross_cohort_evaluate(lambda:FeatureSessionModel([fn],lambda:LogisticRegression(max_iter=2000,random_state=42)),sessions,name=f'corridor_{q}'); rows.append({'ql':q,'auc_pooled':r.auc_pooled,'ci_low':r.ci_low,'ci_high':r.ci_high,**{f'auc_{k}':v for k,v in r.auc_by_direction.items()}})
    config.METRICS_DIR.mkdir(parents=True,exist_ok=True); (config.METRICS_DIR/'corridor_sensitivity.json').write_text(json.dumps(rows,indent=2))
    with (config.METRICS_DIR/'corridor_sensitivity.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
