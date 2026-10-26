"""Write REPORT.md: every number comes from the run, so the report cannot drift from the product."""
import pandas as pd

from . import models as M
from .config import FOLDS, ROOT, S2, TEST_SIZE


def md(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in df.values) + "\n"


def write_report(b):
    c, yte = b["costs"], b["y_test"]
    base = M.baseline(yte, c)
    rows, cvrows = [], []
    for k, r in b["results"].items():
        m05, mt = M.metrics(yte, r["test_proba"], 0.5, c), M.metrics(yte, r["test_proba"], r["thr"], c)
        rows.append({"model": M.MODEL_NAMES[k], "AUC": r["auc_test"], "threshold": r["thr"],
                     "accuracy @0.5": m05["accuracy"], "accuracy @threshold": mt["accuracy"],
                     "majority baseline accuracy": base["accuracy"], "recall (dropouts caught)": mt["recall"],
                     "precision (flags that were right)": mt["precision"], "flagged": mt["flagged"],
                     "cost/student": mt["cost_per_student"], "baseline cost/student": base["cost_per_student"]})
        f = r["folds"]
        cvrows.append({"model": M.MODEL_NAMES[k], **{f"{col} (mean ± sd)": f"{f[col].mean():.3f} ± {f[col].std():.3f}"
                                                         for col in ("auc", "accuracy", "recall", "cost_per_student")},
                       "baseline accuracy (mean)": f"{f['baseline_accuracy'].mean():.3f}"})
    res, cv = pd.DataFrame(rows), pd.DataFrame(cvrows)
    aud = b["audit"]
    found = aud[aud["records_affected"] > 0]
    cats = sorted(found["category"].unique())
    k, attrs = b["sel"]["k"], b["attrs"]
    sc, red = b["scaling"], b["reduction"]
    t = f"""# Student Dropout Early-Warning Console: project report (generated from the run)

**User:** a university academic advisor at the mid-semester review.  **Task:** binary classification: will this student
drop out (Dropout = 1) rather than stay or graduate (0)?  **Data:** UCI *Predict Students' Dropout and Academic Success*,
{b['n_raw']} raw rows, {b['n_clean']} after cleaning.  **Technical challenge:** class imbalance with asymmetric cost.

## A. Data and documentation
* Positive class share: {b['y_train'].mean():.1%} of the training portion; the majority class is 'did not drop out'.
* Measurement types present: nominal, ordinal, interval and ratio (all four of Lecture 2).
* Full data dictionary (every column, type, range, role): see the table below and the *Data & quality* screen.
* **Columns not known when the advisor asks, therefore excluded:** all six 2nd-semester columns
  ({', '.join(S2.values())}) because they are only available after the decision point. The advisor reviews a student
  after the 1st-semester results; those columns would leak the future. The six unexplainable code columns and the three
  sensitive attributes (gender, marital status, international) are also not used for ranking; gender is kept for the
  fairness audit only.
* **Final-outcome caveat:** 'Enrolled' students have no final outcome yet. They are treated as 'not dropped out so far',
  which is a limitation, not a fact.

{md(b['dictionary'])}

## B. Data-quality audit and repair
The UCI data was pre-cleaned by its authors, so several classic faults are genuinely absent; they are reported as 0, not invented.
Fault categories found with at least one record: **{len(cats)}** ({', '.join(cats)}).{"  **WARNING: the brief asks for at least six.**" if len(cats) < 6 else ""}

{md(aud)}

Repairs (each repaired field has a `prov__<field>` column in the cleaned file) and their measured effect (5-fold CV AUC
of a small decision tree, before vs after each repair on its own):

{md(b['effects'], 4)}

`validate()` has 15 assertions. It **fails {int((~b['validate_raw'].passed).sum())} of 15 on the raw data** and **{int((~b['validate_clean'].passed).sum())} of 15 on the cleaned data**.

{md(pd.concat([b['validate_raw'].assign(data='raw'), b['validate_clean'].assign(data='cleaned')])[['data','assertion','passed']])}

## C. Pre-processing
**Rescaling, decided per model family and measured (CV AUC on the training portion):**

{md(sc)}

* Decision tree: no rescaling. Splits compare one attribute with a threshold, so monotone rescaling cannot change the tree (the two rows agree).
* k-NN: min-max rescaling to [0, 1]. Distance adds attributes together, so an attribute measured in hundreds (admission grade) would swamp one counted in single digits.
* Naive Bayes: rescaling is irrelevant because every attribute becomes a category.

**Dimensionality reduction vs feature subset selection, same model (k-NN):**

{md(red)}

Selection ranks attributes by mutual information with the label, re-ranked inside every CV fold so the selection never sees
the validation fold. k = **{k}** is the smallest subset whose mean CV AUC is within 0.005 of the best. The product uses the subset
because PCA components cannot be explained to an advisor. Chosen attributes: {', '.join(attrs)}.

{md(b['sel']['table'])}

**Discretization / binarization.** Naive Bayes needs categories: equal-frequency vs equal-width bins and 3/5/8 bins are
chosen by the grid search below (equal-frequency keeps every bin populated for skewed counts). Nominal attributes are
binarized (one-hot) for the tree and k-NN.

## D. Models and tuning
Three families: decision tree (the explainer), k-nearest neighbours, naive Bayes; all trained on the same prepared training portion.
Hyper-parameters: exhaustive grid, {FOLDS}-fold stratified CV, scoring ROC AUC (AUC does not depend on the threshold, which is
set separately from the cost matrix). The test set is touched once, at the end.

{md(b['tuning'])}

## E. Evaluation
**Split:** the data has no date column, so a **stratified random {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%} split** is used (the split the dataset's authors used too). A time split is impossible here and this is stated deliberately.

**Cost matrix** (units: one advisor meeting): TN {c['TN']:g}, FP {c['FP']:g}, TP {c['TP']:g}, FN {c['FN']:g}.
A flag always costs one meeting; a missed dropout costs five meetings' worth (lost fees, a lost student, and a much harder
conversation later). For calibrated probabilities the cost-optimal threshold is {M.theoretical_threshold(c):.2f};
the product instead picks the empirical optimum on out-of-fold training predictions, per model. The cost ratio is an
assumption to defend at the viva, and the *Settings* screen lets you change it live.

**Test-set results** (majority baseline = flag nobody, beside every figure):

{md(res)}

**Cross-validation on the training portion, mean ± sd over {FOLDS} folds:**

{md(cv)}

**Why recall and precision.** The advisor's two questions are 'how many students who will drop out do I catch?' (recall) and
'how many of the students I am told to see actually needed it?' (precision, their workload). Accuracy alone rewards doing nothing,
since the baseline already scores {base['accuracy']:.1%}.

**What accuracy is given up:** compare 'accuracy @0.5' with 'accuracy @threshold' above. Lowering the threshold trades accuracy for catching more dropouts, deliberately.

**Fairness check** (recall and flag rate by gender at each model's threshold):

{md(b['fairness'])}

## F. Explanation
* Decision tree: the conditions this student actually met, written in words, with the change in dropout rate each condition caused.
* Naive Bayes: each attribute's multiplier on the odds of dropout.
* k-NN: the most similar past students and their outcomes.
Advisor-facing text never shows node numbers or impurity values.

## G. Application
Six screens: Data & quality, Model performance, Single prediction, Batch scoring, Settings, Advisor caseload.
Start with `streamlit run app.py`. Pipeline run time: {b['seconds']:.0f} s.

## Honest limitations
* Probabilities from the tree and k-NN are not calibrated; they rank well but should be read as scores.
* 'Enrolled' outcomes are unknown; the label is 'not dropped out so far'.
* Macro-economic attributes are constant within an enrolment cohort, so they partly identify the cohort rather than the student.
* The dataset has no semester field, so the caseload's semester filter works only if an uploaded file has a `Semester` column.
"""
    (ROOT / "REPORT.md").write_text(t, encoding="utf-8")
