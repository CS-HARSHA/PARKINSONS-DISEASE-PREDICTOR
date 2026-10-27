from pathlib import Path
import csv, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.data_loading import load_processed
from src.evaluation import aggregate_subjects, cross_cohort_evaluation
from src.features import macd

def main():
    sessions=load_processed(); rows=[]
    for ql in [0.4,0.5,0.6,0.7,0.8]:
        examples=aggregate_subjects(sessions, feature_fn=lambda x, q=ql: macd(x,q,1.0))
        result=cross_cohort_evaluation(examples)
        rows.append({'corridor_low':ql,'corridor_high':1.0,'auc':result['auc'],'ci_low':result['auc_ci_95_subject_bootstrap'][0],'ci_high':result['auc_ci_95_subject_bootstrap'][1]})
    path=config.METRICS_DIR/'corridor_sensitivity.csv'; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(*rows,sep='\n')
if __name__=='__main__': main()
