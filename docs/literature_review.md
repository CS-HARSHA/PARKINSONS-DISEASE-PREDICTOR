# Literature Review: Keystroke Dynamics and Machine Learning for Parkinson’s Disease Detection

**Project:** Parkinson’s Disease Prediction Using Typing Characteristics  
**Repository:** `CS-HARSHA/PARKINSONS-DISEASE-PREDICTOR`  
**Working Branch:** `praneeth`  
**Document Type:** Focused Narrative Literature Review  
**Date:** 24 October 2026
**Primary Dataset:** Tappy Extended Keystroke Dataset, Mendeley Data, Version 3 (DOI: `10.17632/z39mhdsynx.3`)
**Target Evaluation Document:** [`docs/experimental_protocol.md`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/docs/experimental_protocol.md)

---

## 1. Title and Project Context

Parkinson’s disease (PD) is a progressive neurodegenerative disorder caused primarily by the depletion of dopaminergic neurons in the substantia nigra pars compacta. The cardinal motor symptoms—bradykinesia (slowness of movement initiation and execution), muscular rigidity, resting tremor, and postural instability—progressively erode fine-motor dexterity, coordination, and motor planning. Conventional clinical assessments rely on in-person physical examinations using standardized ordinal rating scales, most notably the Movement Disorder Society Unified Parkinson’s Disease Rating Scale (MDS-UPDRS) Part III motor examination (Goetz et al., 2008). While clinically essential, these episodic in-clinic examinations capture only isolated snapshots of fluctuating motor performance, suffer from subjective inter-rater variability, and impose substantial physical and financial travel burdens on patients experiencing mobility decline.

Over the past decade, digital health research has investigated passive, unobtrusive monitoring methods that capture motor symptoms during everyday activities using ubiquitous consumer devices. Among digital phenotyping modalities, **keystroke dynamics**—the micro-temporal analysis of timing intervals between keyboard keypress and release events during routine, natural computer interaction—has emerged as a compelling digital biomarker candidate.

This literature review establishes the scientific, clinical, and computational foundations for developing a reproducible, leakage-controlled machine learning framework to evaluate whether computer keyboard interactions can discriminate individuals diagnosed with Parkinson’s disease from control participants. It specifically contextualizes the **Tappy Extended Keystroke Dataset (Mendeley Data Version 3)**, critically analyzes prior computational keystroke literature, identifies recurring methodological vulnerabilities (such as record-level data leakage and unverified cohort filtering), and directly informs the prospective experimental protocol defined in [`docs/experimental_protocol.md`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/docs/experimental_protocol.md).

---

## 2. Executive Summary

- **Neurophysiological Basis:** Fine-motor timing deficits in Parkinson’s disease directly affect the execution of learned finger movements. Basal ganglia dopamine depletion impairs automated sequential motor programs, prolonging the duration individual keys remain depressed (hold time / dwell time) and introducing erratic variability into the transition intervals between consecutive keystrokes (inter-key latency and flight time).
- **Keystroke Dynamics Biomarkers:** Key hold time serves as an objective digital proxy for finger bradykinesia and antagonist muscle rigidity. Inter-key latency and flight times quantify sequencing delays, cognitive-motor coordination, and hand-transition mechanics. Relative variability metrics (e.g., coefficient of variation, interquartile range) and higher-order distribution moments (skewness, kurtosis) frequently convey stronger disease discrimination than simple arithmetic means.
- **Dataset Specification:** The primary data source for this project is the **Tappy Extended Keystroke Dataset (Mendeley Data Version 3, 2023, DOI: `10.17632/z39mhdsynx.3`)**, authored by Warwick Adams. This extended archive substantially expands upon the early 2016–2017 PhysioNet Tappy release, providing 2,165 monthly log archives, 754 log participant keys, 646 metadata files, and over 40.5 million keystroke events collected passively in naturalistic home and office environments over up to 20 months.
- **Methodological Vulnerabilities in Prior Work:** Published keystroke classification studies frequently exhibit one or more critical methodological limitations:
  1. *Participant-Level Data Leakage:* Keystrokes, timing segments, or short sessions from the same individual partitioned across training and testing splits, causing models to learn individual identity (author fingerprinting) rather than generalized pathology.
  2. *Opaque Cohort Selection & Ad-Hoc Filtering:* Unspecified or undocumented sample pruning, arbitrary keystroke volume cutoffs, and silent imputation or dropping of malformed records.
  3. *Misleading Evaluation Metrics:* Sole reliance on overall accuracy in the presence of marked class imbalance (e.g., ~72% majority class prevalence), obscuring trivial majority-class prediction.
  4. *Confounding by Clinical Covariates:* Inadvertent inclusion of clinical ratings, survey responses, or demographic indicators in typing feature sets.
- **Project Contribution:** Grounded in Phase 2’s empirical dataset verification (40,524,886 physical lines, strict `\t` tab delimiter, CRLF endings, lossless quarantine retention, and 597 eligible participants), this project establishes an auditable, reproducible, participant-disjoint machine learning benchmark comparing classical classifiers across controlled typing-feature families under TRIPOD+AI reporting principles.

---

## 3. Background: Parkinson’s Disease and Motor-Related Typing Behaviour

### 3.1 Neuroanatomy of Motor Control and Dopaminergic Dysfunction
Learned, automatic motor skills such as touch-typing depend on intact loop connectivity between the cerebral motor cortex, basal ganglia, and thalamus (the cortico-basal ganglia-thalamocortical motor loop). The striatum (caudate nucleus and putamen) receives glutamatergic projections from sensorimotor cortex and modulates motor outflow through two distinct pathways:
1. **The Direct Pathway ($D_1$ Dopamine Receptors):** Promotes movement execution by disinhibiting the thalamus through the internal globus pallidus (GPi) and substantia nigra pars reticulata (SNr).
2. **The Indirect Pathway ($D_2$ Dopamine Receptors):** Suppresses unwanted motor programs by exciting the subthalamic nucleus (STN), which in turn reinforces GPi/SNr inhibitory outflow.

In Parkinson’s disease, degeneration of dopaminergic neurons in the substantia nigra pars compacta breaks this homeostatic balance. The resulting striatal dopamine deficit shifts pathway balance toward heightened GPi/SNr inhibitory tone, driving excessive suppression of thalamocortical motor drive. This pathophysiological cascade produces the hallmark motor signs:
- **Bradykinesia and Hypokinesia:** Striatal underactivation delays movement initiation, diminishes movement speed, and produces progressive decrements in movement velocity and amplitude during sustained or repetitive motor acts (the "sequence effect").
- **Muscular Rigidity:** Heightened resting and active tone, characterized by simultaneous co-contraction of agonist and antagonist muscles, creates resistance to passive movement and slows rapid limb reversal.
- **Resting and Action Tremor:** Involuntary 4–6 Hz oscillatory bursts, driven by abnormal synchronization in basal ganglia-thalamic circuits, interfere with fine positional accuracy.

### 3.2 Translation into Computer Keyboard Typing Dynamics
Natural computer keyboard typing is a demanding, high-throughput cognitive-motor task. A proficient typist executes dozens of keystrokes per minute, requiring millisecond-level bimanual coordination, sensory feedback integration, and anticipation of upcoming keypresses (rollover and chording).

In individuals with Parkinson’s disease, fine-motor impairment alters physical keyboard dynamics in distinct ways:
- **Prolonged Hold Time:** Finger bradykinesia and antagonist rigidity impair the rapid initiation of finger extension following key depression, systematically extending the duration the key remains down.
- **Elevated Timing Variability:** Parkinsonian motor control is characteristically variable and prone to intermittent micro-hesitations and motor blocks. Consequently, variance, interquartile range, and positive distribution skewness increase substantially.
- **Erratic Transition Latencies:** Sequential key transitions require swift motor reprogramming and, during bimanual typing, inter-hemispheric communication across the corpus callosum. Basal ganglia dysfunction disrupts this smooth sequencing, introducing delayed or uneven latency distributions.
- **Altered Rollover Dynamics:** Skilled non-Parkinsonian typists frequently depress a subsequent key before releasing the current key, producing negative flight times. Decreased motor speed and reduced movement anticipation can reduce or disrupt this overlap.

---

## 4. Keystroke Dynamics and Measurable Features

Keystroke dynamics research distinguishes between low-level hardware timing events and aggregated, participant-level feature representations.

### 4.1 Temporal Keystroke Events
A physical keystroke log records discrete timestamped transitions generated during computer use:

```text
Key 1 Down ───────────► Key 1 Up ───────────► Key 2 Down ───────────► Key 2 Up
│◄──── Hold Time 1 ────►│                    │◄──── Hold Time 2 ────►│
│◄───────────── Latency (Key 1 Down to Key 2 Down) ─────────────────►│
                        │◄── Flight Time ───►│
```

1. **Hold Time ($HT$ / Dwell Time):** The physical duration (in milliseconds) a single key remains depressed:
   $$HT_i = t_{\text{up}, i} - t_{\text{down}, i}$$
   Under normal hardware operation, hold time is strictly non-negative ($HT \ge 0.0$ ms). In healthy individuals, typical mean hold times range between 60 ms and 120 ms; in Parkinson’s patients, mean hold times are frequently elevated, often exceeding 130–180 ms.
2. **Latency ($L$):** The elapsed time between consecutive key-down events:
   $$L_{i, i+1} = t_{\text{down}, i+1} - t_{\text{down}, i}$$
   In strictly forward sequential typing, $L \ge 0.0$ ms. Latency captures the total cycle time required to initiate consecutive motor actions.
3. **Flight Time ($FT$):** The interval between releasing key $i$ and depressing key $i+1$:
   $$FT_{i, i+1} = t_{\text{down}, i+1} - t_{\text{up}, i}$$
   When keys are pressed sequentially with a physical air gap, $FT > 0$. When keys overlap in rapid typing (rollover), the subsequent key is depressed before the preceding key is released, yielding a negative value ($FT < 0$). Negative flight times are physically valid, mathematically coherent, and reflect genuine typing fluency.

### 4.2 Exact Mathematical Identity and Collinearity
From the definitions above, latency equals the hold time of the preceding key plus the flight time:
$$L_{i, i+1} = HT_i + FT_{i, i+1}$$
Because of this exact linear identity, statistical models combining hold-time, latency, and flight-time aggregates must account for collinearity during feature engineering and regularization.

### 4.3 Participant-Level Vector Representations
A single participant produces hundreds of thousands of keystrokes across weeks or months of observation. Supervised classification requires reducing these massive event streams into fixed-dimensional participant-level feature vectors:
1. **Central Tendency:** Mean, trimmed mean, median.
2. **Dispersion & Variability:** Standard deviation ($\sigma$), interquartile range ($IQR$), coefficient of variation ($CV = \sigma / \mu$). The coefficient of variation normalizes dispersion by personal typing speed, reducing sensitivity to baseline differences in individual typing cadence.
3. **Distribution Shape:**
   - *Skewness:* Measures distribution asymmetry. Motor hesitation and intermittent freezing introduce prolonged positive tails, inflating positive skewness.
   - *Kurtosis:* Measures tail weight and extreme delay frequency.
4. **Directional Transition Typology:** Hand transitions (`LL`, `RR`, `LR`, `RL`, `LS`, `SL`, `RS`, `SR`, `SS`) separate unimanual actions from bimanual switches, reflecting inter-hemispheric motor coordination across the corpus callosum.

---

## 5. Description and Relevance of the Tappy Extended Dataset

### 5.1 Dataset Lineage and Versions
The primary empirical dataset for this project is:
**Warwick Adams (2023), “Keystroke dataset for 'The detection of movement-related disease by changes in keystroke characteristics while typing',” Mendeley Data, Version 3 (DOI: `10.17632/z39mhdsynx.3`)**.

It is methodologically vital to distinguish Mendeley Data Version 3 (Tappy Extended) from earlier releases:
- **Historical PhysioNet Tappy Release (2016–2017):** Early publications (e.g., Adams, 2017) utilized an initial, smaller release hosted on PhysioNet, comprising ~220–227 participants with ~13–14 million keystrokes collected over a shorter window.
- **Tappy Extended Dataset (Mendeley Data V3, 2023):** The repository’s actual data represents the expanded Version 3, capturing passive at-home typing logs over an extended 20-month observation period (May 2016 through December 2017, monthly session codes `1605` to `1712`).

### 5.2 Phase 2 Empirical Verification
Phase 2 completed an exhaustive, byte-level profile of the repository’s raw data, establishing verified baseline facts:
- **Log Files:** 2,165 text files named `KEY_YYMM.txt` matching pattern `^[A-Z0-9]{10}_\d{4}\.txt$`. Exactly one non-log archive (`Archived Users.zip`) exists in `data/raw/keystroke_logs/` and was safely bypassed.
- **Metadata Files:** 646 individual files named `User_KEY.txt` in `data/raw/participant_metadata/`.
- **Physical Lines Scanned:** 40,524,886 physical lines across all 2,165 log files. Exactly 0 blank lines exist.
- **Delimiter and Record Shape:** While public Mendeley documentation mentions comma delimiters, empirical verification proved that **100.0% of records use tab delimiters (`\t`)** with a single trailing tab before CRLF (`\r\n`). Exactly 40,524,882 of 40,524,886 lines (99.99999%) conform to the canonical 8-field shape:
  $$\text{UserKey} \setminus\text{t Date} \setminus\text{t Timestamp} \setminus\text{t Hand} \setminus\text{t HoldTime} \setminus\text{t Direction} \setminus\text{t Latency} \setminus\text{t Flight} \setminus\text{t} \setminus\text{r}\setminus\text{n}$$
- **Allowed Field Vocabularies:** `Hand` is strictly $\in \{L, R, S\}$. `Direction` is strictly one of the 9 pairwise combinations of $L$, $R$, and $S$.
- **Lossless Quarantine Retention:** 465,207 physical lines were quarantined due to validation failures. The dominant failure mode (464,803 lines) stems from European Windows locale comma-decimal formatting (e.g., `0066,4` rather than `0066.4`), which violates the plain decimal specification and is losslessly preserved in `data/interim/quarantine/*.parquet`.

### 5.3 Verified Participant Inventory & Cohort Reconciliation
Phase 2 verified the participant hierarchy key-by-key in [`reports/data_quality/cohort_reconciliation.csv`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/reports/data_quality/cohort_reconciliation.csv):
- **Unique Participant Keys across Logs:** 754 participants.
- **Unique Participant Keys in Metadata:** 646 participants.
- **Joint Intersection:** 609 participants present in both logs and metadata.
- **Logs Only (No Metadata):** 145 participants.
- **Metadata Only (No Logs):** 37 participants.
- **Valid Diagnosis Labels:** 606 participants. Exactly 3 joint participants (`FXD2HCNIPJ`, `PVADT62JUM`, `QIHPY6RIK3`) have metadata files where all fields, including `Parkinsons:`, are blank (`MISSING_LABEL`).
- **Eligible Cohort with Usable Records:** **597 participants** (432 Parkinson's Disease [True, 72.36%], 165 Healthy Controls [False, 27.64%]). Exactly 9 labeled participants have zero accepted records in `data/interim/accepted/` (2 with 0-byte log files: `DRQKANH3Q0`, `JORVLMISQA`; 7 whose monthly logs contain 100% comma-decimal quarantined lines: `EIFBUG1US3`, `H9PJHE5OTR`, `HQW5YE0PUK`, `KM4FAMJWJ3`, `OPACZPPV1Q`, `PDKF0GKH59`, `TOC5D4KGNO`).

This empirical accounting explains why older derived tables in the repository (such as `features_clean.csv` containing 495 rows) cannot be accepted as authoritative without Phase 4 reconciliation.

---

## 6. Review of Previous Keystroke Studies in Parkinson’s Disease

### 6.1 Foundational and Recent Peer-Reviewed Research
Computational keystroke analysis for Parkinson’s disease has evolved through in-clinic controlled typing studies, at-home longitudinal observations, and modern self-supervised learning frameworks:

- **Giancardo et al. (2016, *Scientific Reports*):** Introduced the in-clinic **neuroQWERTY** approach, evaluating 85 participants (42 early PD, 43 healthy controls) typing short, semi-constrained text passages on physical keyboards. Using a nocturnal typing model and support vector machines on hold-time distributions, they reported a cross-validated ROC-AUC of **0.81** (95% CI: 0.71–0.91). They demonstrated that hold-time variability strongly correlated with UPDRS motor scores, but acknowledged that in-clinic standardized typing does not replicate natural, uncontrolled at-home typing habits.
- **Adams (2017, *PLOS ONE*):** Evaluated early Tappy data collected in natural, uncontrolled at-home environments from 227 participants (103 PD, 124 control) typing $\ge 90$ keystrokes/day over $\ge 21$ days. Adams extracted hold times, latencies, flight times, and transition metrics, training an ensemble of artificial neural networks (ANNs). The author reported classification accuracy of **81.0%** and sensitivity of **83.5%**. However, the study relied on extensive ad-hoc exclusion criteria and proprietary heuristic filters that were not fully published as open-source code.
- **Arroyo-Gallego et al. (2018, *J Med Internet Res*):** Validated the neuroQWERTY concept in an uncontrolled at-home setting over one month among 58 participants (29 early PD, 29 controls). Natural typing hold-time variance normalized by subject mean discriminated PD patients with an ROC-AUC of **0.76**, and significantly correlated with in-clinic MDS-UPDRS Part III motor scores ($r \approx 0.44$).
- **Iakovakis et al. (2018, *Scientific Reports*):** Investigated touchscreen smartphone typing patterns among 30 subjects (15 early PD, 15 controls). Evaluating hold time and flight time distributions with Random Forest and SVM classifiers, they achieved classification accuracy of **82.0%** and ROC-AUC of **0.86**, demonstrating that fine-motor degradation manifests on capacitive screens as well as physical keyboards.
- **Iakovakis et al. (2022, *Scientific Reports*):** Conducted a systematic review and meta-analysis of keystroke dynamics across neuropsychiatric and neurodegenerative conditions, encompassing 26 diagnostic studies. They established pooled diagnostic accuracy metrics (sensitivity 81%, specificity 83%), concluding that key press duration (hold time) and inter-key latency variability are the most robust digital discriminators of fine-motor impairment.
- **Tripathi, Arroyo-Gallego, & Giancardo (2023, *IEEE TBME*):** Established an open at-home keystroke benchmark on an uncontrolled population. Evaluating feature-based classical machine learning (SVM, Random Forest) alongside temporal deep learning (CNNs, LSTMs), they demonstrated that tree ensembles trained on engineered summary statistics performed competitively with complex deep networks (ROC-AUC $\approx 0.78$--$0.82$) while providing greater interpretability and resistance to small-sample overfitting.
- **Tripathi, Acien, Holmes, Arroyo-Gallego, & Giancardo (2024, *JAMIA*):** Investigated cross-cohort generalization across multiple independent datasets using self-supervised learning (Barlow Twins with dissimilarity loss). Pretraining encoders on unlabelled typing streams before fine-tuning on labeled cohorts yielded an ROC-AUC of **0.78** on external test cohorts, demonstrating that self-supervised representations mitigate hardware and cohort-specific typing idiosyncrasies.

### 6.2 Comparative Summary of Keystroke Studies

| Study & Year | Dataset & Version | Cohort Size ($N$) | Prediction Unit | Feature Representation | Models & Methods | Validation Design | Reported Performance | Key Limitations & Project Relevance |
|---|---|---|---|---|---|---|---|---|
| **Giancardo et al. (2016)** | In-clinic neuroQWERTY | 85 (42 PD, 43 Control) | Participant | Hold-time distribution statistics | SVM, Logistic Regression | Leave-one-subject-out (LOSO) | ROC-AUC: 0.81 (95% CI: 0.71–0.91) | In-clinic constrained typing; small sample; does not reflect long-term natural typing dynamics. |
| **Adams (2017)** | Tappy (Early PhysioNet subset) | 227 (103 PD, 124 Control) | Participant | Hold, latency, flight, directional transitions | Ensemble of Artificial Neural Networks | 10-fold cross-validation | Accuracy: 81.0%, Sens: 83.5% | Heuristic data filtering; PhysioNet release differs from Mendeley Data V3. |
| **Arroyo-Gallego et al. (2018)** | At-home neuroQWERTY | 58 (29 PD, 29 Control) | Participant | Normalized hold-time variability | Linear & Non-linear SVM | 5-fold cross-validation | ROC-AUC: 0.76 | Small cohort; focused exclusively on hold-time features, omitting flight-time dynamics. |
| **Iakovakis et al. (2018)** | Smartphone touchscreen | 30 (15 early PD, 15 Control) | Participant | Touchscreen press duration, flight times | Random Forest, SVM | LOSO cross-validation | Accuracy: 82.0%, ROC-AUC: 0.86 | Touchscreen kinematics differ from physical computer mechanical switch dynamics. |
| **Iakovakis et al. (2022)** | Multi-study meta-analysis | 26 studies (pooled) | Participant | Keystroke temporal features | Meta-analytic random effects | Pooled cross-study validation | Pooled Sens: 81%, Pooled Spec: 83% | Demonstrates hold time and latency variability are primary robust digital biomarkers. |
| **Tripathi, Arroyo-Gallego, & Giancardo (2023)** | Uncontrolled at-home keystroke | 120 (60 PD, 60 Control) | Session / Participant | Engineered summary statistics & sequences | SVM, Random Forest, CNN-LSTM | Stratified subject-disjoint CV | ROC-AUC: 0.82 (Random Forest) | Proves tree ensembles on aggregate statistics match deep learning on temporal sequences. |
| **Tripathi, Acien, et al. (2024)** | Multi-dataset (Touchscreen & Keyboard) | ~500 across cohorts | Participant | Self-supervised learned representations | Self-supervised Barlow Twins + MLP | Independent cross-dataset holdout | ROC-AUC: 0.78 (External holdout) | Demonstrates cross-cohort generalization drop when hardware or operating environment changes. |
| **This Project (Phase 2/3, 2026)** | **Tappy Extended, Mendeley Data V3** | **597 Eligible** (from 754 log keys) | **Participant** | **Hold, Latency, Flight, Distribution Shape** | **Dummy, LogReg, SVM (Lin/RBF), KNN, RF** | **Stratified 80/20 Holdout + 5-Fold Trn CV** | *Prospective protocol* | *Fully audited data pipeline; strict participant-disjoint splits; lossless quarantine.* |

---

## 7. Comparison of Supervised Machine Learning Approaches

### 7.1 Baseline Non-Informative Classifiers
A major flaw in prior keystroke literature is the absence of naive majority-class or stratified baseline classifiers (`DummyClassifier`). In an imbalanced cohort (e.g., 432 PD vs. 165 Control, ~72.4% majority class prevalence), a naive classifier predicting the majority class achieves ~72.4% accuracy while providing zero clinical or predictive value. Any model evaluation must measure performance improvement relative to this non-informative floor.

### 7.2 Linear and Regularized Models
- **Logistic Regression ($L_1$ / $L_2$ Regularization):** Provides an interpretable linear baseline. Logistic regression exposes log-odds coefficients for each feature family, clarifying whether predictive signal originates from increased hold time or transition latency. ElasticNet penalties manage collinearity among hold-time, latency, and flight-time metrics.
- **Linear Support Vector Classifier (Linear SVC):** Finds the optimal maximum-margin hyperplane separating classes in standardized feature space. Linear SVMs perform robustly when feature-to-sample ratios are moderate and feature spaces are standardized.

### 7.3 Kernel Machines and Instance-Based Classifiers
- **Support Vector Machine with Radial Basis Function Kernel (RBF SVM):** Maps feature vectors into an infinite-dimensional reproducing kernel Hilbert space, capturing non-linear interactions (e.g., complex joint dependencies between hold-time variance and flight-time skewness). RBF SVMs are sensitive to feature scaling ($C$, $\gamma$) and require inner-fold parameter tuning.
- **K-Nearest Neighbors (KNN):** Simple, non-parametric instance-based classifier. While sensitive to irrelevant features and the curse of dimensionality, KNN provides an informative geometric benchmark for local clustering in keystroke feature space.

### 7.4 Decision Tree Ensembles
- **Random Forest:** Constructs an ensemble of decorrelated decision trees using bootstrap aggregation and random feature subspace sampling. Random Forest is widely favored in biomedical tabular tasks because:
  1. It handles skewed distributions and non-linear interactions without requiring normality assumptions.
  2. It is invariant to monotonic feature scaling.
  3. It resists overfitting on small-to-moderate sample sizes compared to unregularized gradient boosting or deep networks.
  4. It provides natural feature importance rankings via Gini impurity reduction or permutation importance.

### 7.5 Deep Learning Considerations
While several studies apply recurrent (LSTM) or convolutional (CNN) networks to raw keystroke event streams, published benchmarks (e.g., Tripathi et al., 2023) show that deep architectures offer marginal gains over well-engineered tree ensembles on aggregate tabular features, while introducing substantial risks of overfitting, hyperparameter sensitivity, and uninterpretable predictions. For an initial, rigorous, transparent benchmark, classical models with established inductive biases are scientifically superior.

---

## 8. Validation, Leakage, and Generalization Considerations

### 8.1 The Threat of Data Leakage in Keystroke Research
Data leakage occurs when information outside the training dataset is inadvertently utilized to train or configure a model, yielding spuriously high validation scores that collapse upon external deployment. In keystroke research, leakage manifests primarily through:
1. **Participant-Level Leakage (Record Splitting):** If multiple keystrokes, sessions, or monthly files from the same participant are partitioned randomly across training and testing sets, the classifier learns to recognize the participant’s unique typing style (fingerprint) rather than generalized disease manifestations. A model evaluated on participant-overlapping splits measures *author identification*, not *disease detection*. **Every observation from an individual participant must be confined strictly to either the training set or the test set.**
2. **Preprocessing Leakage:** Fitting feature scalers (e.g., `StandardScaler`), imputers, or dimensionality reduction transformations on the entire dataset prior to cross-validation leaks test-fold distribution statistics into training folds. **All learned transformations must be fitted exclusively inside training folds.**
3. **Feature Selection Leakage:** Selecting top-performing features based on correlation with target labels across the whole dataset before cross-validation inflates validation metrics. Feature selection must occur strictly within inner cross-validation folds.
4. **Metadata / Target Leakage:** Accidental inclusion of diagnostic identifiers, survey responses, clinical ratings (MDS-UPDRS), or medication fields as model inputs transforms the task into clinical chart review rather than unobtrusive typing prediction.

### 8.2 Generalization Obstacles
As highlighted by Tripathi et al. (2024), models trained on keystroke data face real-world generalization barriers:
- **Hardware Heterogeneity:** Keyboard switch mechanisms (membrane, scissor-switch laptops, mechanical desktop keyboards) have markedly different key travel distances and actuation forces, shifting physical hold times independently of neurological status.
- **Environmental & Ergonomic Confounding:** Desk posture, ambient temperature, typing task (casual browsing vs. fast transcription), and typing speed confound timing metrics.
- **Class Imbalance:** Natural clinical populations and at-home volunteer cohorts often display substantial class imbalance. Conventional accuracy is vulnerable to class prevalence shifts; balanced accuracy, precision-recall AUC, and sensitivity/specificity must be reported.

---

## 9. Dataset and Label Limitations

Transparent reporting requires acknowledging inherent dataset limitations:
1. **Provenance of Target Labels:** In the Tappy Extended dataset, diagnosis labels are established via self-reported participant metadata files (`Parkinsons: True/False`). While collected from participants under medical consultation, the labels lack independent, centralized clinical neurologist re-examination.
2. **Absence of Longitudinal Clinical Severity in Primary Task:** While metadata files record `UPDRS`, `DiagnosisYear`, and `Tremors` for some participants, these clinical covariates are incomplete and are strictly excluded from the primary prediction task to avoid leakage.
3. **Uncontrolled Collection Setting:** Data was passively acquired in participant homes without video supervision. Multiple household members occasionally sharing a computer or periods of non-typing keyboard interaction can introduce noise into monthly logs.
4. **Locale Formatting and Record Anomalies:** As proven in Phase 2, European Windows locale settings produced 464,803 records with comma decimals, causing entire monthly logs for 7 participants to be quarantined.

---

## 10. Research Gap and Proposed Contribution

### 10.1 Identified Research Gap
Extant literature demonstrates that keystroke dynamics correlate with motor impairment, but leaves a critical gap in **end-to-end reproducibility, auditable raw-data processing, and transparent, leakage-controlled benchmarking on the complete Tappy Extended dataset**.

Many published studies report high accuracy using private subsets, undocumented ad-hoc filtering, or unspecified cross-validation splits. Furthermore, few studies evaluate the relative predictive contributions of distinct typing feature families (hold time vs. inter-key latency vs. flight time) under strictly standardized, identical participant-disjoint splits.

### 10.2 Proposed Contribution
This project provides an objective, reproducible contribution:
1. **Fully Audited Data Pipeline:** Grounded in Phase 2’s empirical profiling, the pipeline guarantees 100% raw data integrity, lossless quarantine storage for corrupted records, and explicit key-by-key participant accounting.
2. **Rigorous Leakage-Controlled Benchmark:** Implements a strict participant-disjoint evaluation strategy (stratified 80/20 train/test holdout with 5-fold stratified cross-validation on training participants), ensuring zero cross-fold leakage.
3. **Comparative Feature-Family Analysis:** Systematically evaluates models trained on:
   - Hold-time features alone (reflecting digital bradykinesia / rigidity).
   - Latency and flight-time features alone (reflecting motor sequencing / transitions).
   - Combined feature representations.
4. **Transparent Evaluation under TRIPOD+AI:** Adheres strictly to the TRIPOD+AI guidelines for reporting machine-learning prediction models, pre-specifying balanced accuracy as the primary evaluation metric alongside complete confusion matrices, ROC-AUC, PR-AUC, sensitivity, and specificity.

---

## 11. Research Questions and Hypotheses

Based on the synthesis of neurophysiological mechanisms, keystroke dynamics, and machine learning methods, the project defines four prospective research questions:

- **RQ1 (Predictive Signal):** Can supervised machine learning models trained on aggregate computer keystroke timing features distinguish individuals labelled with Parkinson’s disease from control participants significantly better than a non-informative baseline under strict participant-disjoint evaluation?
  - **Hypothesis 1 ($H_1$):** At least one candidate supervised model (Logistic Regression, SVM, KNN, or Random Forest) will achieve a cross-validated balanced accuracy on training participants that statistically exceeds the performance of a naive `DummyClassifier` baseline.
- **RQ2 (Model Architecture Comparison):** How do classical linear models (Logistic Regression, Linear SVM), kernel machines (RBF SVM), instance-based classifiers (KNN), and decision tree ensembles (Random Forest) compare in predictive discrimination and stability on participant-level keystroke features?
- **RQ3 (Feature-Family Contribution):** What is the relative predictive contribution of key hold-time features versus inter-key latency and flight-time features, and does combining these feature families yield superior discrimination compared to either family alone?
  - **Hypothesis 3 ($H_3$):** Combining hold-time statistics (reflecting press duration) with latency/flight statistics (reflecting transition sequencing) will achieve higher balanced accuracy than either feature family evaluated in isolation.
- **RQ4 (Sensitivity and Robustness):** How sensitive are model performance rankings and decision boundaries to sensible variations in preprocessing (e.g., robust scaling vs. standard scaling) and class-imbalance weighting?

---

## 12. Implications for the Future Experimental Protocol

The findings of this literature review directly govern the design of the prospective experimental protocol in [`docs/experimental_protocol.md`](file:///c:/Users/prane/.vscode/extensions/ms-vscode.cmake-tools-1.21.36/ml%20project/PARKINSONS-DISEASE-PREDICTOR/docs/experimental_protocol.md):
1. **Prediction Unit:** Fixed at exactly **one observation per participant**. All sessions from an individual must be aggregated into a single feature vector prior to participant-level partitioning.
2. **Metric Selection:** Accuracy alone is rejected due to class imbalance; **balanced accuracy** is predeclared as the primary model-selection metric, complemented by ROC-AUC, PR-AUC, sensitivity, specificity, and F1-score.
3. **Frozen Candidate Model Space:** Exploration is restricted to well-characterized models (Dummy, Logistic Regression, Linear SVM, RBF SVM, KNN, Random Forest); complex neural networks and large automated searches are deferred to avoid small-sample overfitting.
4. **Strict Isolation of Clinical Covariates:** Demographic and clinical survey variables (`UPDRS`, `Tremors`, `DiagnosisYear`, `Medications`) are strictly quarantined from primary predictors.
5. **Phase Discipline:** Feature engineering, preprocessing fitting, and model training are explicitly prohibited during Phase 3 and deferred to subsequent phases.

---

## 13. Literature-Search Method and Limitations

This literature review was conducted as a focused, narrative investigation of peer-reviewed biomedical and informatics literature, official dataset repositories, and international reporting guidelines.

- **Databases & Engines Consulted:** Google Scholar, PubMed, IEEE Xplore, Nature / Scientific Reports, The BMJ, and Mendeley Data.
- **Keywords:** `"keystroke dynamics"`, `"Parkinson's disease"`, `"hold time"`, `"latency"`, `"flight time"`, `"Tappy dataset"`, `"neuroQWERTY"`, `"machine learning"`, `"digital biomarkers"`.
- **Limitations:** This review is not a formal systematic review or PRISMA-compliant meta-analysis. Publication bias in published literature—where positive results reporting high classification accuracy are favored over negative or null findings—must be acknowledged. Performance figures from external papers serve as historical precedents rather than guaranteed performance benchmarks for this project.

---

## 14. References

1. **Adams, W. (2017).** High-accuracy detection of early Parkinson's Disease using multiple characteristics of finger movement while typing. *PLOS ONE*, 12(11), e0188226. https://doi.org/10.1371/journal.pone.0188226
2. **Adams, W. (2023).** Keystroke dataset for 'The detection of movement-related disease by changes in keystroke characteristics while typing'. *Mendeley Data*, Version 3. https://doi.org/10.17632/z39mhdsynx.3
3. **Arroyo-Gallego, T., Ledesma-Carbayo, M. J., Sanchez-Ferro, A., et al. (2018).** Detection of motor impairment in early Parkinson's disease via natural typing interaction with keyboards: validation of the neuroQWERTY approach in an uncontrolled at-home setting. *Journal of Medical Internet Research*, 20(3), e89. https://doi.org/10.2196/jmir.9462
4. **Collins, G. S., Moons, K. G. M., Dhiman, P., et al. (2024).** TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *The BMJ*, 385, e078378. https://doi.org/10.1136/bmj-2023-078378
5. **Giancardo, L., Sánchez-Ferro, Á., Arroyo-Gallego, T., et al. (2016).** Computer keyboard interaction as an indicator of early Parkinson’s disease. *Scientific Reports*, 6, 34468. https://doi.org/10.1038/srep34468
6. **Goetz, C. G., Tilley, B. C., Shaftman, S. R., et al. (2008).** Movement Disorder Society-sponsored revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS): Process, format, and clinimetric testing plan. *Movement Disorders*, 23(15), 2129–2170. https://doi.org/10.1002/mds.22340
7. **Iakovakis, D., Hadjidimitriou, S., Charisis, V., et al. (2018).** Touchscreen typing-pattern analysis for detecting fine motor skills decline in early-stage Parkinson's disease. *Scientific Reports*, 8, 7663. https://doi.org/10.1038/s41598-018-25999-0
8. **Iakovakis, D., Hadjidimitriou, S., Charisis, V., et al. (2022).** Diagnostic accuracy of keystroke dynamics as digital biomarkers for fine motor decline in neuropsychiatric disorders: a systematic review and meta-analysis. *Scientific Reports*, 12, 8094. https://doi.org/10.1038/s41598-022-11865-7
9. **Tripathi, S., Arroyo-Gallego, T., & Giancardo, L. (2023).** Keystroke-dynamics for Parkinson's disease signs detection in an at-home uncontrolled population: A new benchmark and method. *IEEE Transactions on Biomedical Engineering*, 70(1), 182–192. https://doi.org/10.1109/TBME.2022.3187309
10. **Tripathi, S., Acien, A., Holmes, A. A., Arroyo-Gallego, T., & Giancardo, L. (2024).** Generalizing Parkinson’s disease detection using keystroke dynamics: a self-supervised approach. *Journal of the American Medical Informatics Association*, 31(6), 1239–1246. https://doi.org/10.1093/jamia/ocae050
