# Student Dropout Early-Warning Console: project report (generated from the run)

**User:** a university academic advisor at the mid-semester review.  **Task:** binary classification: will this student
drop out (Dropout = 1) rather than stay or graduate (0)?  **Data:** UCI *Predict Students' Dropout and Academic Success*,
4424 raw rows, 4424 after cleaning.  **Technical challenge:** class imbalance with asymmetric cost.

## A. Data and documentation
* Positive class share: 32.1% of the training portion; the majority class is 'did not drop out'.
* Measurement types present: nominal, ordinal, interval and ratio (all four of Lecture 2).
* Full data dictionary (every column, type, range, role): see the table below and the *Data & quality* screen.
* **Columns not known when the advisor asks, therefore excluded:** all six 2nd-semester columns
  (Curricular units 2nd sem (credited), Curricular units 2nd sem (enrolled), Curricular units 2nd sem (evaluations), Curricular units 2nd sem (approved), Curricular units 2nd sem (grade), Curricular units 2nd sem (without evaluations)) because they are only available after the decision point. The advisor reviews a student
  after the 1st-semester results; those columns would leak the future. The six unexplainable code columns and the three
  sensitive attributes (gender, marital status, international) are also not used for ranking; gender is kept for the
  fairness audit only.
* **Final-outcome caveat:** 'Enrolled' students have no final outcome yet. They are treated as 'not dropped out so far',
  which is a limitation, not a fact.

| column | plain label | measurement type | plausible range | observed in data | role | note |
|---|---|---|---|---|---|---|
| Marital Status | marital status | nominal | codes 1-6 | 6 distinct values | must not be used (sensitive; fairness audit only) | Sensitive personal attribute: kept for the fairness audit only, never used to rank a student. |
| Application mode | how the student applied | nominal | 18 documented codes | 18 distinct values | explanatory attribute (candidate) |  |
| Application order | application choice (0 = first choice) | ordinal | 0-9 | 0 to 9 | explanatory attribute (candidate) |  |
| Course | programme | nominal | 17 programme codes | 17 distinct values | explanatory attribute (candidate) |  |
| Daytime/evening attendance | class schedule | nominal | 1 = daytime, 0 = evening | 2 distinct values | explanatory attribute (candidate) |  |
| Previous qualification | previous qualification code | nominal | 17 codes | 17 distinct values | must not be used (unexplainable code) | Unordered code with 17-46 levels: cannot be explained to an advisor in plain words, so not used. |
| Previous qualification (grade) | grade of previous qualification (0-200) | interval | 0-200 | 95 to 190 | explanatory attribute (candidate) |  |
| Nacionality | nationality code | nominal | 21 codes | 21 distinct values | must not be used (unexplainable code) | Unordered code with 17-46 levels: cannot be explained to an advisor in plain words, so not used. |
| Mother's qualification | mother's qualification code | nominal | 29 codes | 29 distinct values | must not be used (unexplainable code) | Unordered code with 17-46 levels: cannot be explained to an advisor in plain words, so not used. |
| Father's qualification | father's qualification code | nominal | 34 codes | 34 distinct values | must not be used (unexplainable code) | Unordered code with 17-46 levels: cannot be explained to an advisor in plain words, so not used. |
| Mother's occupation | mother's occupation code | nominal | 32 codes | 32 distinct values | must not be used (unexplainable code) | Unordered code with 17-46 levels: cannot be explained to an advisor in plain words, so not used. |
| Father's occupation | father's occupation code | nominal | 46 codes | 46 distinct values | must not be used (unexplainable code) | Unordered code with 17-46 levels: cannot be explained to an advisor in plain words, so not used. |
| Admission grade | admission grade (0-200) | interval | 0-200 | 95 to 190 | explanatory attribute (candidate) |  |
| Displaced | living situation | nominal | 1 = yes, 0 = no | 2 distinct values | explanatory attribute (candidate) |  |
| Educational special needs | special educational needs | nominal | 1 = yes, 0 = no | 2 distinct values | explanatory attribute (candidate) |  |
| Debtor | debt to the university | nominal | 1 = yes, 0 = no | 2 distinct values | explanatory attribute (candidate) |  |
| Tuition fees up to date | tuition fees | nominal | 1 = yes, 0 = no | 2 distinct values | explanatory attribute (candidate) |  |
| Gender | gender | nominal | 1 = male, 0 = female | 2 distinct values | must not be used (sensitive; fairness audit only) | Sensitive personal attribute: kept for the fairness audit only, never used to rank a student. |
| Scholarship holder | scholarship | nominal | 1 = yes, 0 = no | 2 distinct values | explanatory attribute (candidate) |  |
| Age at enrollment | age at enrolment (years) | ratio | 17-70 | 17 to 70 | explanatory attribute (candidate) |  |
| International | international student | nominal | 1 = yes, 0 = no | 2 distinct values | must not be used (sensitive; fairness audit only) | Sensitive personal attribute: kept for the fairness audit only, never used to rank a student. |
| Curricular units 1st sem (credited) | units credited from earlier study (1st sem) | ratio | 0-30 | 0 to 20 | explanatory attribute (candidate) |  |
| Curricular units 1st sem (enrolled) | units enrolled (1st sem) | ratio | 0-30 | 0 to 26 | explanatory attribute (candidate) |  |
| Curricular units 1st sem (evaluations) | exams/evaluations taken (1st sem) | ratio | 0-50 | 0 to 45 | explanatory attribute (candidate) |  |
| Curricular units 1st sem (approved) | units passed (1st sem) | ratio | 0-30 | 0 to 26 | explanatory attribute (candidate) |  |
| Curricular units 1st sem (grade) | average grade (1st sem, 0-20) | interval | 0-20; 0 = no grade | 0 to 18.875 | explanatory attribute (candidate) |  |
| Curricular units 1st sem (without evaluations) | units with no evaluation (1st sem) | ratio | 0-15 | 0 to 12 | explanatory attribute (candidate) |  |
| Curricular units 2nd sem (credited) | credited (2nd sem) | ratio | 0-30 | 0 to 19 | must not be used (not known at decision time) | NOT known when the advisor asks (2nd-semester result, available only after the decision point). |
| Curricular units 2nd sem (enrolled) | enrolled (2nd sem) | ratio | 0-30 | 0 to 23 | must not be used (not known at decision time) | NOT known when the advisor asks (2nd-semester result, available only after the decision point). |
| Curricular units 2nd sem (evaluations) | evaluations (2nd sem) | ratio | 0-50 | 0 to 33 | must not be used (not known at decision time) | NOT known when the advisor asks (2nd-semester result, available only after the decision point). |
| Curricular units 2nd sem (approved) | approved (2nd sem) | ratio | 0-30 | 0 to 20 | must not be used (not known at decision time) | NOT known when the advisor asks (2nd-semester result, available only after the decision point). |
| Curricular units 2nd sem (grade) | grade (2nd sem) | interval | 0-20; 0 = no grade | 0 to 18.5714 | must not be used (not known at decision time) | NOT known when the advisor asks (2nd-semester result, available only after the decision point). |
| Curricular units 2nd sem (without evaluations) | without evaluations (2nd sem) | ratio | 0-15 | 0 to 12 | must not be used (not known at decision time) | NOT known when the advisor asks (2nd-semester result, available only after the decision point). |
| Unemployment rate | regional unemployment rate (%) | ratio | 7-17 | 7.6 to 16.2 | explanatory attribute (candidate) |  |
| Inflation rate | inflation rate (%) | interval | -1 to 4 | -0.8 to 3.7 | explanatory attribute (candidate) |  |
| GDP | GDP growth (%) | interval | -5 to 4 | -4.06 to 3.51 | explanatory attribute (candidate) |  |
| Target | final outcome | nominal | Dropout / Enrolled / Graduate | Graduate 2209, Dropout 1421, Enrolled 794 | target (source of the label) | Dropout = 1, others = 0 |
| Dropout | dropped out (0/1) | nominal | 0/1 | 1 = Dropout, 0 = Graduate or Enrolled | target | derived by the cleaning step |


## B. Data-quality audit and repair
The UCI data was pre-cleaned by its authors, so several classic faults are genuinely absent; they are reported as 0, not invented.
Fault categories found with at least one record: **7** (Ambiguous label, Inconsistent naming, Inconsistent values, Missing values, Near-constant attribute, Outliers, Wrong data type).

| category | fault | detecting expression | records_affected | handling | share |
|---|---|---|---|---|---|
| Duplicate data | Fully identical records | df.duplicated().sum() | 0 | dropped | 0.000 |
| Duplicate data | Identical attributes, different outcome label | df.duplicated(subset=attributes) & ~df.duplicated() | 0 | reported only | 0.000 |
| Missing values | Explicit missing cells in an attribute | df[attributes].isna().any(axis=1) | 0 | none to repair | 0.000 |
| Missing values | Grade stored as 0 although no evaluation was taken (1st sem) | (evaluations == 0) & (grade == 0) | 349 | grade set to missing | 0.079 |
| Missing values | Same, 2nd semester | (evaluations == 0) & (grade == 0) | 401 | grade set to missing | 0.091 |
| Outliers | Age at enrolment above Tukey fence Q3 + 1.5 IQR | age > Q3 + 1.5*IQR | 441 | capped at Q3 + 3 IQR | 0.100 |
| Outliers | Admission grade outside Tukey fences | x < Q1-1.5*IQR or x > Q3+1.5*IQR | 86 | kept (plausible values) | 0.019 |
| Outliers | Previous qualification grade outside Tukey fences | x < Q1-1.5*IQR or x > Q3+1.5*IQR | 179 | kept (plausible values) | 0.041 |
| Inconsistent values | Units approved > units enrolled | approved > enrolled | 0 | flagged, not altered | 0.000 |
| Inconsistent values | Units approved > evaluations taken | approved > evaluations | 0 | flagged, not altered | 0.000 |
| Inconsistent values | Positive grade but no unit approved | (grade > 0) & (approved == 0) | 0 | flagged, not altered | 0.000 |
| Inconsistent values | Units approved but grade is 0 | (grade == 0) & (approved > 0) | 0 | flagged, not altered | 0.000 |
| Inconsistent values | Evening programme but daytime attendance (or the reverse) | (course in {8014, 9991}) != (attendance == 0) | 0 | flagged, not altered | 0.000 |
| Inconsistent values | Enrolled in a programme but 0 units enrolled in 1st semester | enrolled (1st sem) == 0 | 180 | flagged, not altered | 0.041 |
| Out-of-domain values | Code or value outside the documented domain | isin(documented codes) / between(lo, hi) | 0 | none to repair | 0.000 |
| Wrong data type | 9 nominal code attributes stored as integers (arithmetic is meaningless) | is_numeric_dtype(col) for nominal attributes | 4424 | cast to category | 1.000 |
| Inconsistent naming | Column names with stray whitespace (0) or misspelt (Nacionality) | [c for c in cols if c != c.strip()] | 4424 | whitespace stripped; typo documented, name kept to match the code book | 1.000 |
| Ambiguous label | Target = 'Enrolled': outcome not final yet | Target == 'Enrolled' | 794 | treated as 'not dropped out so far' | 0.179 |
| Near-constant attribute | Attribute with >= 97% one value: Educational special needs | value_counts(normalize=True).max() >= .97 | 4424 | kept; feature selection decides | 1.000 |


Repairs (each repaired field has a `prov__<field>` column in the cleaned file) and their measured effect (5-fold CV AUC
of a small decision tree, before vs after each repair on its own):

| repair | records_changed | field_mean_before | field_mean_after | cv_auc_before | cv_auc_after | auc_change |
|---|---|---|---|---|---|---|
| names | 0.0000 |  |  | 0.8651 | 0.8651 | 0.0000 |
| duplicates | 0.0000 |  |  | 0.8651 | 0.8651 | 0.0000 |
| disguised_missing | 750.0000 | 10.6408 | 11.5521 | 0.8651 | 0.8674 | 0.0023 |
| age_cap | 156.0000 | 23.2651 | 23.0536 | 0.8651 | 0.8651 | 0.0000 |
| flags | 180.0000 |  |  | 0.8651 | 0.8651 | 0.0000 |
| target | 794.0000 |  |  | 0.8651 | 0.8651 | 0.0000 |
| types | 9.0000 |  |  | 0.8651 | 0.8651 | 0.0000 |
| ALL repairs together |  |  |  | 0.8651 | 0.8674 | 0.0023 |


`validate()` has 15 assertions. It **fails 6 of 15 on the raw data** and **0 of 15 on the cleaned data**.

| data | assertion | passed |
|---|---|---|
| raw | 1 column names have no stray whitespace | True |
| raw | 2 no fully duplicated rows | True |
| raw | 3 missing values only where a grade cannot exist (no evaluation) | True |
| raw | 4 no grade of 0 disguised as a value when no evaluation was taken | False |
| raw | 5 nominal code attributes are categorical, not numeric | False |
| raw | 6 age not above Q3 + 3 IQR | False |
| raw | 7 binary attributes contain only 0/1 | True |
| raw | 8 application order within 0-9 | True |
| raw | 9 programme / application-mode / marital codes are documented codes | True |
| raw | 10 grades in 0-20 (semester) and 0-200 (admission, previous) | True |
| raw | 11 unit counts are non-negative | True |
| raw | 12 every approved > enrolled record carries a flag | False |
| raw | 13 every repaired (missing) grade has a provenance entry | False |
| raw | 14 Dropout label present and consistent with Target | False |
| raw | 15 Target has only Dropout / Enrolled / Graduate | True |
| cleaned | 1 column names have no stray whitespace | True |
| cleaned | 2 no fully duplicated rows | True |
| cleaned | 3 missing values only where a grade cannot exist (no evaluation) | True |
| cleaned | 4 no grade of 0 disguised as a value when no evaluation was taken | True |
| cleaned | 5 nominal code attributes are categorical, not numeric | True |
| cleaned | 6 age not above Q3 + 3 IQR | True |
| cleaned | 7 binary attributes contain only 0/1 | True |
| cleaned | 8 application order within 0-9 | True |
| cleaned | 9 programme / application-mode / marital codes are documented codes | True |
| cleaned | 10 grades in 0-20 (semester) and 0-200 (admission, previous) | True |
| cleaned | 11 unit counts are non-negative | True |
| cleaned | 12 every approved > enrolled record carries a flag | True |
| cleaned | 13 every repaired (missing) grade has a provenance entry | True |
| cleaned | 14 Dropout label present and consistent with Target | True |
| cleaned | 15 Target has only Dropout / Enrolled / Graduate | True |


## C. Pre-processing
**Rescaling, decided per model family and measured (CV AUC on the training portion):**

| model | rescaled | cv_auc_mean | cv_auc_sd |
|---|---|---|---|
| Decision tree | no | 0.858 | 0.013 |
| Decision tree | min-max | 0.858 | 0.013 |
| k-nearest neighbours | no | 0.845 | 0.010 |
| k-nearest neighbours | min-max | 0.860 | 0.005 |


* Decision tree: no rescaling. Splits compare one attribute with a threshold, so monotone rescaling cannot change the tree (the two rows agree).
* k-NN: min-max rescaling to [0, 1]. Distance adds attributes together, so an attribute measured in hundreds (admission grade) would swamp one counted in single digits.
* Naive Bayes: rescaling is irrelevant because every attribute becomes a category.

**Dimensionality reduction vs feature subset selection, same model (k-NN):**

| representation | dimensions | cv_auc_mean | cv_auc_sd |
|---|---|---|---|
| all 21 attributes | 54 | 0.794 | 0.022 |
| selected subset | 4 | 0.860 | 0.005 |
| PCA (90% variance) on all 21 | 25 | 0.786 | 0.019 |


Selection ranks attributes by mutual information with the label, re-ranked inside every CV fold so the selection never sees
the validation fold. k = **4** is the smallest subset whose mean CV AUC is within 0.005 of the best. The product uses the subset
because PCA components cannot be explained to an advisor. Chosen attributes: Curricular units 1st sem (approved), Curricular units 1st sem (grade), Tuition fees up to date, Curricular units 1st sem (evaluations).

| k | tree | tree_sd | knn | knn_sd | nb | nb_sd | mean |
|---|---|---|---|---|---|---|---|
| 4 | 0.858 | 0.015 | 0.856 | 0.012 | 0.849 | 0.012 | 0.854 |
| 6 | 0.863 | 0.013 | 0.854 | 0.008 | 0.856 | 0.012 | 0.858 |
| 8 | 0.861 | 0.015 | 0.823 | 0.009 | 0.861 | 0.009 | 0.848 |
| 10 | 0.860 | 0.012 | 0.822 | 0.011 | 0.861 | 0.008 | 0.848 |
| 12 | 0.862 | 0.010 | 0.815 | 0.008 | 0.859 | 0.007 | 0.845 |
| 15 | 0.861 | 0.010 | 0.796 | 0.019 | 0.855 | 0.008 | 0.838 |
| 18 | 0.863 | 0.011 | 0.793 | 0.023 | 0.852 | 0.007 | 0.836 |
| 21 | 0.863 | 0.011 | 0.794 | 0.022 | 0.852 | 0.007 | 0.836 |


**Discretization / binarization.** Naive Bayes needs categories: equal-frequency vs equal-width bins and 3/5/8 bins are
chosen by the grid search below (equal-frequency keeps every bin populated for skewed counts). Nominal attributes are
binarized (one-hot) for the tree and k-NN.

## D. Models and tuning
Three families: decision tree (the explainer), k-nearest neighbours, naive Bayes; all trained on the same prepared training portion.
Hyper-parameters: exhaustive grid, 5-fold stratified CV, scoring ROC AUC (AUC does not depend on the threshold, which is
set separately from the cost matrix). The test set is touched once, at the end.

| model | best_params | cv_auc | grid_size |
|---|---|---|---|
| Decision tree | {"m__criterion": "gini", "m__max_depth": 5, "m__min_samples_leaf": 50} | 0.859 | 30 |
| k-nearest neighbours | {"m__metric": "manhattan", "m__n_neighbors": 51} | 0.866 | 10 |
| Naive Bayes | {"m__alpha": 1.0, "x__n_bins": 8, "x__strategy": "quantile"} | 0.852 | 18 |


## E. Evaluation
**Split:** the data has no date column, so a **stratified random 80%/20% split** is used (the split the dataset's authors used too). A time split is impossible here and this is stated deliberately.

**Cost matrix** (units: one advisor meeting): TN 0, FP 1, TP 1, FN 5.
A flag always costs one meeting; a missed dropout costs five meetings' worth (lost fees, a lost student, and a much harder
conversation later). For calibrated probabilities the cost-optimal threshold is 0.20;
the product instead picks the empirical optimum on out-of-fold training predictions, per model. The cost ratio is an
assumption to defend at the viva, and the *Settings* screen lets you change it live.

**Test-set results** (majority baseline = flag nobody, beside every figure):

| model | AUC | threshold | accuracy @0.5 | accuracy @threshold | majority baseline accuracy | recall (dropouts caught) | precision (flags that were right) | flagged | cost/student | baseline cost/student |
|---|---|---|---|---|---|---|---|---|---|---|
| Decision tree | 0.880 | 0.200 | 0.838 | 0.785 | 0.679 | 0.842 | 0.622 | 384 | 0.688 | 1.605 |
| k-nearest neighbours | 0.877 | 0.220 | 0.837 | 0.774 | 0.679 | 0.856 | 0.604 | 402 | 0.686 | 1.605 |
| Naive Bayes | 0.881 | 0.150 | 0.825 | 0.746 | 0.679 | 0.863 | 0.568 | 431 | 0.707 | 1.605 |


**Cross-validation on the training portion, mean ± sd over 5 folds:**

| model | auc (mean ± sd) | accuracy (mean ± sd) | recall (mean ± sd) | cost_per_student (mean ± sd) | baseline accuracy (mean) |
|---|---|---|---|---|---|
| Decision tree | 0.859 ± 0.015 | 0.785 ± 0.010 | 0.822 ± 0.030 | 0.708 ± 0.031 | 0.679 |
| k-nearest neighbours | 0.866 ± 0.007 | 0.778 ± 0.009 | 0.822 ± 0.028 | 0.715 ± 0.026 | 0.679 |
| Naive Bayes | 0.852 ± 0.008 | 0.754 ± 0.013 | 0.842 ± 0.033 | 0.720 ± 0.031 | 0.679 |


**Why recall and precision.** The advisor's two questions are 'how many students who will drop out do I catch?' (recall) and
'how many of the students I am told to see actually needed it?' (precision, their workload). Accuracy alone rewards doing nothing,
since the baseline already scores 67.9%.

**What accuracy is given up:** compare 'accuracy @0.5' with 'accuracy @threshold' above. Lowering the threshold trades accuracy for catching more dropouts, deliberately.

**Fairness check** (recall and flag rate by gender at each model's threshold):

| model | group | n | dropout_rate | flagged_rate | recall | precision |
|---|---|---|---|---|---|---|
| Decision tree | female | 564 | 0.245 | 0.351 | 0.819 | 0.571 |
| Decision tree | male | 321 | 0.455 | 0.579 | 0.863 | 0.677 |
| k-nearest neighbours | female | 564 | 0.245 | 0.383 | 0.841 | 0.537 |
| k-nearest neighbours | male | 321 | 0.455 | 0.579 | 0.870 | 0.683 |
| Naive Bayes | female | 564 | 0.245 | 0.408 | 0.855 | 0.513 |
| Naive Bayes | male | 321 | 0.455 | 0.626 | 0.870 | 0.632 |


## F. Explanation
* Decision tree: the conditions this student actually met, written in words, with the change in dropout rate each condition caused.
* Naive Bayes: each attribute's multiplier on the odds of dropout.
* k-NN: the most similar past students and their outcomes.
Advisor-facing text never shows node numbers or impurity values.

## G. Application
Six screens: Data & quality, Model performance, Single prediction, Batch scoring, Settings, Advisor caseload.
Start with `streamlit run app.py`. Pipeline run time: 42 s.

## Honest limitations
* Probabilities from the tree and k-NN are not calibrated; they rank well but should be read as scores.
* 'Enrolled' outcomes are unknown; the label is 'not dropped out so far'.
* Macro-economic attributes are constant within an enrolment cohort, so they partly identify the cohort rather than the student.
* The dataset has no semester field, so the caseload's semester filter works only if an uploaded file has a `Semester` column.
