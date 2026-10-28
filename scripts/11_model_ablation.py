"""WP9: Compare classifiers on MACD and small transparent feature sets."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import config
from src.data_loading import load_processed
from src.features import macd,stdev,mean_hold
from src.models import make_classifier
from src.compare import FeatureSessionModel
from src.evaluation import cross_cohort_evaluate

def main():
    s=load_processed(); out=[]
    for name in ['logistic','linear_svm','random_forest','knn']:
        r=cross_cohort_evaluate(lambda name=name:FeatureSessionModel([macd],lambda:make_classifier(name,42)),s,name=f'ablation_{name}'); out.append({'model':name,'features':'macd','auc_pooled':r.auc_pooled,'ci_low':r.ci_low,'ci_high':r.ci_high,'auc_by_direction':r.auc_by_direction})
    for fns,n in [([macd,stdev],'macd_std'),([macd,stdev,mean_hold],'macd_std_mean')]:
        r=cross_cohort_evaluate(lambda fns=fns:FeatureSessionModel(fns,lambda:make_classifier('logistic',42)),s,name=f'ablation_{n}'); out.append({'model':'logistic','features':n,'auc_pooled':r.auc_pooled,'ci_low':r.ci_low,'ci_high':r.ci_high,'auc_by_direction':r.auc_by_direction})
    (config.METRICS_DIR/'model_ablation.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
