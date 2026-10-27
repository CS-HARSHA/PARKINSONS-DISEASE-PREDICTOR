# Final project structure and file inventory

Generated from the current repository. Raw downloaded data are listed separately from reusable code and generated outputs.

## Top-level structure

```text
parkinsons-keystroke/
├── README.md
├── PROJECT_STRUCTURE.md
├── TEAM_WORKPLAN.md
├── DATA_DICTIONARY.md
├── docs_GLOSSARY.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── src/
├── scripts/
├── notebooks/
├── reports/
├── results/
├── models/
├── tests/
└── webapp/
```

## Pipeline order

```text
01_prepare_data.py → cleaned sessions + audit
02_eda.py → group-comparison EDA + plots + report
03_extract_features.py → feature tables
04_evaluate_macd.py → baseline AUC/CI/ROC
05_corridor_sensitivity.py → corridor sweep
06_export_model.py → exported model JSON
07_benchmark_models.py → model/feature comparison
run_pipeline.sh → all stages + tests
```

## Root files

- `.gitignore`
- `DATA_DICTIONARY.md`
- `README.md`
- `TEAM_WORKPLAN.md`
- `docs_GLOSSARY.md`
- `requirements.txt`

## src

- `src/__init__.py`
- `src/benchmark.py`
- `src/cleaning.py`
- `src/config.py`
- `src/data_loading.py`
- `src/eda.py`
- `src/evaluation.py`
- `src/export_model.py`
- `src/feature_extraction.py`
- `src/feature_sets.py`
- `src/features.py`
- `src/modeling.py`

## scripts

- `scripts/01_prepare_data.py`
- `scripts/02_eda.py`
- `scripts/02_reproduce_macd.py`
- `scripts/03_extract_features.py`
- `scripts/04_evaluate_macd.py`
- `scripts/05_corridor_sensitivity.py`
- `scripts/06_export_model.py`
- `scripts/07_benchmark_models.py`
- `scripts/README.md`
- `scripts/_evaluate_common.py`
- `scripts/run_pipeline.sh`

## notebooks

- `notebooks/01_explore_data.ipynb`

## reports

- `reports/eda.md`
- `reports/model_benchmark.md`
- `reports/replication_report.md`

## results

- `results/figures/consecutive_change_by_label.png`
- `results/figures/hold_time_by_label.png`
- `results/figures/macd_by_label.png`
- `results/figures/roc_macd.png`
- `results/metrics/corridor_sensitivity.csv`
- `results/metrics/eda_summary.json`
- `results/metrics/macd_cross_cohort.json`
- `results/metrics/macd_subject_predictions.csv`
- `results/metrics/model_feature_benchmark.csv`
- `results/metrics/model_feature_benchmark.json`

## models

- `models/macd_model.json`

## data/processed

- `data/processed/cleaning_summary.json`
- `data/processed/session_features.csv`
- `data/processed/sessions.json`
- `data/processed/subject_eda_summary.csv`
- `data/processed/subject_features.csv`

## tests

- `tests/conftest.py`
- `tests/test_benchmark.py`
- `tests/test_cleaning.py`
- `tests/test_evaluation.py`
- `tests/test_feature_extraction.py`
- `tests/test_features.py`
- `tests/test_js_parity.md`

## webapp

- `webapp/css/style.css`
- `webapp/index.html`
- `webapp/js/app.js`
- `webapp/js/capture.js`
- `webapp/js/macd.js`

## Raw data inventory

The raw directory contains **124 files**, including the PhysioNet archive, extracted cohort data, ground-truth CSVs, the original loader, and paper materials.

- `data/raw/.extracted`
- `data/raw/MIT-CS1PD/GT_DataPD_MIT-CS1PD.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1401114972.068_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1401117235.067_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1401978395.067_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1401980765.068_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1402930351.011_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1402932300.060_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1403706430.011_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1403708258.060_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1403792224.074_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1403795516.072_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1403797570.077_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1404311419.070_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1404313658.081_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1404743687.070_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1404745424.081_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1404747197.074_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1404819923.080_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1404828566.083_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1404830509.082_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1404832986.072_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1404834530.077_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1407163319.084_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1407165254.085_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1407421738.075_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1407423449.076_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1407425493.082_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1407427375.083_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1408029544.097_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1408372533.088_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1408374282.089_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1408376132.093_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1408378153.094_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1408630965.080_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1408633211.097_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1408976968.086_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1408979136.087_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1408980899.085_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1408982860.084_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1409235978.088_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1409238087.089_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1409754834.076_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1409756763.075_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1410272175.092_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1410274781.086_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1410276815.087_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1410446321.071_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1410448218.095_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1410450204.078_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1410451895.079_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1411655869.071_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1411657572.095_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1412086661.092_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1412088758.078_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1412090543.079_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1412692310.094_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1412694503.093_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1418744709.098_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1418746466.099_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1418748996.073_001_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1419866851.098_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1419868806.099_003_014.csv`
- `data/raw/MIT-CS1PD/data_MIT-CS1PD/1419870664.073_003_014.csv`
- `data/raw/MIT-CS2PD/GT_DataPD_MIT-CS2PD.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1424946827.1000_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1426676689.1002_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1427279751.1001_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1429091219.1006_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1429176842.1008_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1429866367.1004_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1430134526.1005_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1435046994.1010_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1436192511.1009_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1436195831.1011_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1437580607.1014_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1437583257.1012_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1437670749.1016_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1440577460.1022_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1441110114.1013_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1441279015.1015_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1441893019.1017_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1442299892.1019_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1442506646.1024_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1443090933.1028_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1443094060.1030_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1443534559.1021_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1444035286.1020_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1444400554.1023_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1444893029.1025_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1445956280.1029_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1446028987.1032_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1446626569.1034_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1449158503.1031_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1449842818.1033_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1451987802.1042_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1452598928.1035_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1453990802.1037_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1454061159.1050_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1456139318.1052_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1456752658.1041_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1457345994.1056_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1457347484.1039_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1458555711.1043_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1458557031.1062_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1458723488.1064_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1459162558.1047_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1459181723.1045_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1460104760.1066_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1460559248.1068_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1461063976.1049_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1461082447.1053_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1461084495.1051_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1461089132.1057_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1461091652.1055_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1461677130.1059_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1463130212.1070_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1463172280.1061_001_014.csv`
- `data/raw/MIT-CS2PD/data_MIT-CS2PD/1463511198.1063_001_014.csv`
- `data/raw/less_is_more.pdf`
- `data/raw/less_is_more.txt`
- `data/raw/neuroQWERTY.zip`
- `data/raw/nqDataLoader.py`
- `data/raw/readme.ipynb`

## Validation status

- Enhanced EDA regenerated successfully.
- 7 automated tests passed.
- Baseline replication AUC: 0.8505.
- Exploratory MACD + random forest mean held-out fold AUC: 0.8650.

## Recommended starting files

- `README.md` for execution.
- `src/eda.py` and `scripts/02_eda.py` for the EDA implementation.
- `notebooks/01_explore_data.ipynb` for interactive analysis.
- `reports/eda.md` for the generated interpretation.
- `TEAM_WORKPLAN.md` for team ownership.
