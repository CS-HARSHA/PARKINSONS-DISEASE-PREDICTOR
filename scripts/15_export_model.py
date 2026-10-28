"""WP10: Retrain MACD logistic regression on all sessions and export browser JSON."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data_loading import load_processed
from src.export_model import export_model
if __name__=='__main__':
    payload=export_model(load_processed()); print(f"Exported {payload['model_type']} to models/macd_model.json")
