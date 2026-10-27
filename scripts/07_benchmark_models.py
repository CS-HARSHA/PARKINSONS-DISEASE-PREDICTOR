from pathlib import Path
import csv, json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.benchmark import aggregate_feature_sets, benchmark, model_factories
from src.data_loading import load_processed

def main():
    feature_sets = {
        'macd': ('macd',),
        'stdev': ('stdev',),
        'mean_hold': ('mean_hold',),
        'macd_stdev': ('macd', 'stdev'),
        'macd_stdev_mean': ('macd', 'stdev', 'mean_hold'),
    }
    rows=[]
    for set_name, names in feature_sets.items():
        subject_rows=aggregate_feature_sets(load_processed(), names)
        for model_name in model_factories():
            result=benchmark(subject_rows, names, model_name)
            rows.append({'feature_set':set_name,'model':model_name,'mean_fold_auc':result['mean_fold_auc'],'pooled_auc_descriptive':result['pooled_auc_descriptive'],'fold_1_auc':result['folds'][0]['auc'],'fold_2_auc':result['folds'][1]['auc']})
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path=config.METRICS_DIR/'model_feature_benchmark.csv'
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (config.METRICS_DIR/'model_feature_benchmark.json').write_text(json.dumps(rows,indent=2)+'\n')
    ranked=sorted(rows,key=lambda r:r['mean_fold_auc'],reverse=True)
    report=['# Model and feature benchmark','', 'This is an exploratory comparison. The MACD replication baseline remains frozen and is not replaced by the best row automatically.', '', '| Feature set | Model | Mean held-out fold AUC | Cohort 1 AUC | Cohort 2 AUC |', '|---|---|---:|---:|---:|']
    for row in ranked:
        report.append(f"| {row['feature_set']} | {row['model']} | {row['mean_fold_auc']:.4f} | {row['fold_1_auc']:.4f} | {row['fold_2_auc']:.4f} |")
    report += ['', '## Interpretation', '', f"The best exploratory row is **{ranked[0]['feature_set']} + {ranked[0]['model']}**, with mean held-out fold AUC **{ranked[0]['mean_fold_auc']:.4f}**. This is not a definitive improvement claim: the dataset has only two cross-cohort folds, and model selection must be confirmed with a predeclared validation plan or an independent dataset.", '', 'The primary comparison is the mean of the two held-out cohort AUCs. Pooled outputs from independently fitted models are reported as descriptive because their score scales may differ.']
    (config.ROOT/'reports'/'model_benchmark.md').write_text('\n'.join(report)+'\n')
    for row in sorted(rows,key=lambda r:r['mean_fold_auc'],reverse=True): print(row)
if __name__=='__main__': main()
