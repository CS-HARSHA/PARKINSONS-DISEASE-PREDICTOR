# Pipeline scripts

These files are execution entry points. They import reusable functions from `src/` and save outputs. The scientific implementation should remain in `src/`; scripts should stay thin and easy to run.

## Execution order

Run from the repository root:

```bash
python3 scripts/01_prepare_data.py
python3 scripts/02_eda.py
python3 scripts/03_extract_features.py
python3 scripts/04_evaluate_macd.py
python3 scripts/05_corridor_sensitivity.py
python3 scripts/06_export_model.py
python3 scripts/07_benchmark_models.py
pytest -q
```

Or run the same sequence with:

```bash
./scripts/run_pipeline.sh
```

## Script inventory

- `01_prepare_data.py`: loads the raw PhysioNet archive, applies cleaning through `src.data_loading`, and writes processed sessions plus a cleaning audit.
- `02_eda.py`: loads processed sessions, calls `src.eda.summarize_sessions`, and writes EDA JSON and Markdown.
- `02_reproduce_macd.py`: backwards-compatible wrapper for the MACD evaluation stage.
- `03_extract_features.py`: writes session-level and subject-level MACD tables.
- `04_evaluate_macd.py`: runs the frozen paper replication through the shared evaluation helper.
- `05_corridor_sensitivity.py`: evaluates predeclared MACD corridor lower quantiles.
- `06_export_model.py`: fits the final all-subject baseline and writes `models/macd_model.json`.
- `07_benchmark_models.py`: compares simple feature sets and classical models under the same cross-cohort folds.
- `_evaluate_common.py`: shared orchestration used by the evaluation entry points.
- `run_pipeline.sh`: runs every stage and then the tests.
