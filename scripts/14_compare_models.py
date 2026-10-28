"""WP8: Build a comparison table from saved model metrics without inventing values."""
from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import config
from src.compare import collect_saved_results
def main():
    rows=[]
    for d in collect_saved_results(): rows.append({'model':d.get('name'),'auc_pooled':d.get('auc_pooled'),'ci_low':d.get('ci_low'),'ci_high':d.get('ci_high'),'auc_by_direction':json.dumps(d.get('auc_by_direction',{}))})
    config.METRICS_DIR.mkdir(parents=True,exist_ok=True)
    with (config.METRICS_DIR/'model_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','auc_pooled','ci_low','ci_high','auc_by_direction']); w.writeheader(); w.writerows(rows)
    (config.METRICS_DIR/'model_comparison.json').write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
