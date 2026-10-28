"""WP5: Run the documented nQi-style SVR ensemble approximation."""
from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import config
from src.data_loading import load_processed
from src.baselines_nqi import NqiModel,nqi_window_features
from src.evaluation import cross_cohort_evaluate,save_result
def main():
    s=load_processed(); rows=[]
    for sess in s:
        w=nqi_window_features(sess.hold_times_s,sess.press_times_s)
        for i,row in enumerate(w): rows.append({'session_id':sess.session_id,'window_idx':i,**{f'f{j+1}':float(v) for j,v in enumerate(row)}})
    with (config.PROCESSED_DIR/'nqi_window_features.csv').open('w',newline='') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    try:
        r=cross_cohort_evaluate(lambda:NqiModel(),s,name='baseline_nqi'); save_result(r,'baseline_nqi'); d={'name':'baseline_nqi','auc_pooled':r.auc_pooled,'ci_low':r.ci_low,'ci_high':r.ci_high,'auc_by_direction':r.auc_by_direction}
    except ValueError as e: d={'name':'baseline_nqi','status':'unavailable','reason':str(e)}
    (config.METRICS_DIR/'baseline_nqi.json').write_text(json.dumps(d,indent=2)); print(json.dumps(d,indent=2))
if __name__=='__main__':main()
