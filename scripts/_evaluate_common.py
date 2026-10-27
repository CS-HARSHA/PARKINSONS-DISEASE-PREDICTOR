from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay
from src import config
from src.data_loading import load_processed
from src.evaluation import aggregate_subjects, cross_cohort_evaluation

def run():
    examples=aggregate_subjects(load_processed())
    result=cross_cohort_evaluation(examples)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True); config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (config.METRICS_DIR/'macd_cross_cohort.json').write_text(json.dumps(result, indent=2)+'\n')
    with (config.METRICS_DIR/'macd_subject_predictions.csv').open('w', newline='') as f:
        fields=['subject_id','cohort','label','score','model_score','fold']; w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(result['predictions'])
    y=np.array([p['label'] for p in result['predictions']]); scores=np.array([p['score'] for p in result['predictions']])
    RocCurveDisplay.from_predictions(y, scores); plt.title(f"Cross-cohort MACD ROC (AUC={result['auc']:.3f})"); plt.tight_layout(); plt.savefig(config.FIGURES_DIR/'roc_macd.png', dpi=160); plt.close()
    print(f"Cross-cohort AUC: {result['auc']:.4f}")
    print(f"Bootstrap 95% CI: {result['auc_ci_95_subject_bootstrap'][0]:.4f}–{result['auc_ci_95_subject_bootstrap'][1]:.4f}")

if __name__ == '__main__': run()
