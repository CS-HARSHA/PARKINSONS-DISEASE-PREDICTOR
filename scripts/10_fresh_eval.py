"""WP7: Evaluate FRESH top-m feature curves with training-only selection."""
from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import config
from src.data_loading import load_processed
from src.fresh_features import extract_all
from src.fresh_eval import FreshModel
from src.evaluation import cross_cohort_evaluate

def main():
    s=load_processed(); cache=extract_all(s); rows=[]
    for m in range(1,11):
        r=cross_cohort_evaluate(lambda m=m:FreshModel(m,cache=cache),s,name=f'fresh_m{m}'); rows.append({'m':m,'auc_pooled':r.auc_pooled,'ci_low':r.ci_low,'ci_high':r.ci_high,'auc_by_direction':r.auc_by_direction})
    config.METRICS_DIR.mkdir(parents=True,exist_ok=True); (config.METRICS_DIR/'fresh_auc_vs_m.json').write_text(json.dumps(rows,indent=2));
    with (config.METRICS_DIR/'fresh_auc_vs_m.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['m','auc_pooled','ci_low','ci_high']); w.writeheader(); w.writerows([{k:r[k] for k in ['m','auc_pooled','ci_low','ci_high']} for r in rows])
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
