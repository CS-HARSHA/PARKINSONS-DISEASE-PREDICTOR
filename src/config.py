"""WP1: Shared configuration and named constants for the reproducible pipeline."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / 'data' / 'raw'
PROCESSED_DIR = ROOT / 'data' / 'processed'
RESULTS_DIR = ROOT / 'results'
METRICS_DIR = RESULTS_DIR / 'metrics'
FIGURES_DIR = RESULTS_DIR / 'figures'
SEED = 42
RANDOM_SEED = SEED
COHORTS = {'early_pd': 'MIT-CS1PD', 'de_novo_pd': 'MIT-CS2PD'}
CORRIDOR_LOW = 0.60
CORRIDOR_HIGH = 1.00
MIN_SESSION_KEYSTROKES = 20
MIN_CORRIDOR_PAIRS = 1
MAX_HOLD_SECONDS = 5.0
MIN_HOLD_SECONDS = 0.0
BOOTSTRAP_REPS = 2000
NQI_WINDOW_SECONDS = 90.0
NQI_MIN_KEYSTROKES = 30
