from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
RANDOM_SEED = 20261027

COHORTS = {"early": "MIT-CS1PD", "de_novo": "MIT-CS2PD"}
CORRIDOR_LOW = 0.60
CORRIDOR_HIGH = 1.00
MIN_SESSION_KEYSTROKES = 20
BOOTSTRAP_REPS = 2000
