from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.data_loading import load_processed
from src.feature_extraction import subject_feature_rows
from src.modeling import fit_logistic
from src.export_model import export_logistic_model

def main():
    rows=subject_feature_rows(load_processed())
    x=np.array([r['macd'] for r in rows]); y=np.array([r['label'] for r in rows])
    model=fit_logistic(x,y)
    export_logistic_model(model,x,config.ROOT/'models'/'macd_model.json')
    print('Exported models/macd_model.json')
if __name__=='__main__': main()
