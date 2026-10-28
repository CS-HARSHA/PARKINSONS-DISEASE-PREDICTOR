"""WP2: Generate canonical session and subject feature tables."""
from pathlib import Path
import csv,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import config
from src.data_loading import load_processed
from src.feature_extraction import session_feature_rows,subject_feature_rows
def write(rows,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def main():
    s=load_processed(); sr=session_feature_rows(s); ur=subject_feature_rows(s); write(sr,config.PROCESSED_DIR/'session_features.csv'); write(ur,config.PROCESSED_DIR/'subject_features.csv'); print(f'Wrote {len(sr)} session and {len(ur)} subject feature rows.')
if __name__=='__main__': main()
