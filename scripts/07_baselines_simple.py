"""WP4: Run simple mean-hold and standard-deviation baseline models."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import config
from src.data_loading import load_processed
from src.baselines_simple import MeanHoldModel,StdHoldModel
from src.evaluation import cross_cohort_evaluate,save_result
def main():
    s=load_processed(); out=[]
    for name,cls in [('baseline_mean',MeanHoldModel),('baseline_std',StdHoldModel)]:
        r=cross_cohort_evaluate(lambda cls=cls:cls(),s,name=name); save_result(r,name); out.append({'name':name,'auc_pooled':r.auc_pooled,'ci_low':r.ci_low,'ci_high':r.ci_high,'auc_by_direction':r.auc_by_direction})
    (config.METRICS_DIR/'baselines_simple.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
