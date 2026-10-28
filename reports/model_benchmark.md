# Model and feature benchmark

This is an exploratory comparison. The MACD replication baseline remains frozen and is not replaced by the best row automatically.

| Feature set | Model | Mean held-out fold AUC | Cohort 1 AUC | Cohort 2 AUC |
|---|---|---:|---:|---:|
| macd | random_forest | 0.8650 | 0.8646 | 0.8654 |
| macd | logistic_regression | 0.8513 | 0.8222 | 0.8803 |
| macd_stdev | logistic_regression | 0.8451 | 0.8014 | 0.8889 |
| macd_stdev_mean | logistic_regression | 0.8382 | 0.7875 | 0.8889 |
| stdev | logistic_regression | 0.8321 | 0.7625 | 0.9017 |
| macd_stdev | random_forest | 0.8263 | 0.8021 | 0.8504 |
| stdev | rbf_svm | 0.8217 | 0.7417 | 0.9017 |
| macd | rbf_svm | 0.8212 | 0.8090 | 0.8333 |
| macd_stdev | rbf_svm | 0.8062 | 0.7833 | 0.8291 |
| macd_stdev_mean | random_forest | 0.7949 | 0.7479 | 0.8419 |
| macd_stdev_mean | rbf_svm | 0.7806 | 0.7278 | 0.8333 |
| mean_hold | logistic_regression | 0.7481 | 0.7056 | 0.7906 |
| stdev | random_forest | 0.7255 | 0.7181 | 0.7329 |
| mean_hold | rbf_svm | 0.7182 | 0.6500 | 0.7863 |
| mean_hold | random_forest | 0.5839 | 0.5674 | 0.6004 |

## Interpretation

The best exploratory row is **macd + random_forest**, with mean held-out fold AUC **0.8650**. This is not a definitive improvement claim: the dataset has only two cross-cohort folds, and model selection must be confirmed with a predeclared validation plan or an independent dataset.

The primary comparison is the mean of the two held-out cohort AUCs. Pooled outputs from independently fitted models are reported as descriptive because their score scales may differ.
