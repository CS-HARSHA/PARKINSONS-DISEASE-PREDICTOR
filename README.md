# Parkinson's Disease Predictor

This repository implements a reproducible machine-learning workflow to predict Parkinson's disease from natural keystroke dynamics and clinical metadata.

---

## Project Structure

```text
configs/
  pipeline.toml                 # Phase 2 pipeline configuration
data/
  raw/
    keystroke_logs/             # 2,165 raw typing session logs (KEY_YYMM.txt)
    participant_metadata/       # 646 participant clinical metadata files (User_KEY.txt)
  interim/                      # Intermediate data products (Git-ignored)
    accepted/                   # Validated keystroke records (Parquet)
    quarantine/                 # Quarantined records with lossless bytes (Parquet)
    participant_labels.csv      # Participant diagnosis labels
    participant_covariates.csv  # Participant clinical covariates
    participant_inventory.csv   # Master inventory & cohort status
    run_manifest.json           # Execution manifest & environment metadata
  processed/                    # Model-ready datasets (Phase 3+)
docs/
  phase2_recon.md               # Reconnaissance and baseline recovery
  log_format_spec.md            # Empirically verified format specification
  data_dictionary.md            # Schema and data dictionary for interim outputs
  phase2_decisions.md           # Engineering decision log
notebooks/
  01_dataset_audit.ipynb        # Phase 1 exploratory audit
  02_parser_validation_tests.ipynb # Phase 2 verification notebook
reports/
  data_quality/                 # Published data-quality reports
src/
  data_pipeline/                # Pipeline implementation
    config.py                   # Configuration loader and validator
    parser.py                   # Byte-level streaming parser
    validation.py               # Pure record validator (E01-E11, W01-W05)
    cleaning.py                 # Line classifier & Parquet writer
    participants.py             # Metadata and inventory processor
    reporting.py                # Report and reconciliation generator
    run.py                      # CLI entry point, staging, atomic publisher
tests/                          # Automated test suite (pytest)
```

---

## Installation & Environment Setup

The pipeline requires **Python 3.11+** (verified on Python 3.13.7, 64-bit).

```powershell
# Clone repository and switch to branch
git checkout praneeth

# Install required dependencies
pip install -r requirements.txt
```

---

## Running Automated Tests

Run the full automated test suite (covering unit tests for config, parser, validator, cleaner, participants, reports, and atomic publishing):

```powershell
pytest
```

---

## Running the Data Pipeline (Phase 2)

Execute the full production pipeline across all 2,165 logs and 646 participant metadata files:

```powershell
python -m src.data_pipeline.run
```

### CLI Options

You can customize directory paths or limit processed files for quick testing:

```powershell
# Run with a limit of 10 files
python -m src.data_pipeline.run --max-files 10

# Override configuration or output paths
python -m src.data_pipeline.run --config configs/pipeline.toml --interim-dir data/interim
```

---

## Pipeline Guarantees & Verification

1. **Read-Only Raw Data:** All 2,818 raw files are hashed with SHA-256 before and after execution; hashes must be 100% identical.
2. **Safe Staging & Atomic Publishing:** All outputs are written to isolated staging directories and published atomically only after all invariant and hash checks pass.
3. **Mutual Exclusivity & Line Invariant:** Every physical line produces exactly one outcome:
   $$\text{physical\_lines} = \text{accepted} + \text{accepted\_with\_warning} + \text{quarantined} + \text{blank}$$
4. **Lossless Quarantine:** Quarantined rows retain exact source bytes (`raw_line_bytes`).
5. **Deterministic Reruns:** Deterministic outputs with stable sorting across all files and rows.

---

## Phase 2 Reports & Reconciliations

Published reports are located under `reports/data_quality/`:
- `data_quality_report.md`: Executive summary and line outcome breakdown.
- `summary.json`: High-level run metrics and hash verification flag.
- `audit_reconciliation.csv`: Reconciled metrics against Phase 1 baseline audit.
- `cohort_reconciliation.csv`: Cohort reconciliation explaining the 606 labeled vs 597 eligible participants key-by-key.
- `code_counts.csv`: Error and warning code occurrences and affected physical line counts.
- `backward_timestamp_summary.csv`: Backward timestamp transitions and millisecond deltas.
- `raw_files_sha256.csv`: Pre- and post-run hash verification for all 2,818 raw files.
