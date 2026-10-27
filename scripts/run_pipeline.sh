#!/usr/bin/env bash
set -euo pipefail
python3 scripts/01_prepare_data.py
python3 scripts/02_eda.py
python3 scripts/03_extract_features.py
python3 scripts/04_evaluate_macd.py
python3 scripts/05_corridor_sensitivity.py
python3 scripts/06_export_model.py
python3 scripts/07_benchmark_models.py
pytest -q
