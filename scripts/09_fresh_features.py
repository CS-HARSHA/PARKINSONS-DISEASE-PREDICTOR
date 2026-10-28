"""WP6: Extract and cache FRESH features for every cleaned session."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src import config
from src.data_loading import load_processed
from src.fresh_features import extract_all
def main():
    df=extract_all(load_processed()); config.PROCESSED_DIR.mkdir(parents=True,exist_ok=True); df.to_csv(config.PROCESSED_DIR/'fresh_features.csv',index=False); print(f'Wrote {len(df)} FRESH rows and {len(df.columns)-1} features.')
if __name__=='__main__':main()
