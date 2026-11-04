#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for script in 01_prepare_data.py 02_eda.py 03_extract_features.py 04_evaluate_macd.py 05_corridor_sensitivity.py 06_feature_variants.py 07_baselines_simple.py 08_baselines_nqi.py 09_fresh_features.py 10_fresh_eval.py 11_model_ablation.py 12_keystroke_curve.py 13_noise_injection.py 14_compare_models.py 15_export_model.py; do
 echo "=== $script ==="; python3 "scripts/$script"
done
pytest -q
