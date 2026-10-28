"""WP3: Compare MACD timing variants as robustness analyses."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sklearn.linear_model import LogisticRegression
from src import config
from src.data_loading import load_processed
from src.features import macd,macd_log,macd_norm
from src.compare import FeatureSessionModel
from src.evaluation import cross_cohort_evaluate

def main():
    s=load_processed(); out=[]
    for name,fn in [('macd',macd),('macd_log',macd_log),('macd_norm',macd_norm)]:
        r=cross_cohort_evaluate(lambda fn=fn:FeatureSessionModel([fn],lambda:LogisticRegression(max_iter=2000,random_state=42)),s,name=name); out.append({'name':name,'auc_pooled':r.auc_pooled,'ci_low':r.ci_low,'ci_high':r.ci_high,'auc_by_direction':r.auc_by_direction})
    (config.METRICS_DIR/'feature_variants.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
