# Parkinson's Disease Predictor

This repository is organized for a machine-learning workflow using keystroke
logs and participant metadata.

## Project structure

```text
data/
  raw/
    keystroke_logs/       # Raw per-session typing logs
    participant_metadata/ # Participant-level metadata
  interim/                # Intermediate data products
  processed/              # Model-ready datasets
notebooks/
  01_dataset_audit.ipynb
src/                      # Data processing and model code
models/                   # Saved model artifacts
app/                      # Application code
```

The raw dataset is stored under `data/raw`. The audit notebook provides a
starting point for checking file counts and inspecting the tab-delimited logs
and metadata. Keep generated and processed data out of `data/raw`.
