"""WP8: Run primary MACD plus logistic regression cross-cohort evaluation."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from src import config
from src.data_loading import load_processed
from src.features import macd
from src.compare import FeatureSessionModel
from src.evaluation import cross_cohort_evaluate,save_result
def main():
    sessions=load_processed(); factory=lambda:FeatureSessionModel([macd],lambda:LogisticRegression(max_iter=2000,random_state=config.SEED)); result=cross_cohort_evaluate(factory,sessions,name='macd_cross_cohort'); save_result(result,'macd_cross_cohort'); y=[r['label'] for r in result.subject_table]; s=[r['score'] for r in result.subject_table]; fpr,tpr,_=roc_curve(y,s); config.FIGURES_DIR.mkdir(parents=True,exist_ok=True); plt.figure(figsize=(5,4)); plt.plot(fpr,tpr,label=f'AUC={result.auc_pooled:.3f}'); plt.plot([0,1],[0,1],'--',color='gray'); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.legend(); plt.tight_layout(); plt.savefig(config.FIGURES_DIR/'roc_macd.png',dpi=160); plt.close(); print(json.dumps({'auc':result.auc_pooled,'ci':[result.ci_low,result.ci_high],'by_direction':result.auc_by_direction},indent=2))
if __name__=='__main__':main()
