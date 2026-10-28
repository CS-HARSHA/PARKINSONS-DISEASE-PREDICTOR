# Pipeline glossary

**Subject:** One person in the dataset, identified by `pID`.

**Session:** One typing recording from one subject. MIT-CS1PD usually has two sessions per subject; MIT-CS2PD usually has one.

**Cohort:** A study subgroup collected under one experiment. `MIT-CS1PD` is the early-PD cohort; `MIT-CS2PD` is the de novo, untreated-PD cohort.

**Cross-cohort evaluation:** Train the model on every eligible subject in one cohort and test it on subjects from the other cohort. Then swap the direction and pool the two held-out predictions. This tests whether the signal transfers across study populations and collection sessions. It is stricter than randomly mixing all subjects, because a random split can make the task easier through cohort-specific patterns.

**Subject-level split:** All sessions from one subject stay in one split. Splitting individual keystrokes or sessions from the same person across train and test is leakage.

**EDA:** Exploratory data analysis. It describes data quality, cohort sizes, session lengths, labels, and feature distributions before modeling.
