# Student Dropout Early-Warning Console 

For an academic advisor at the mid-semester review: a ranked tutee list, the two or three factors behind each score,
and a suggested first conversation.

## Run
```
pip install -r requirements.txt
python fetch_data.py        # once: writes data/students_raw.csv  (skip if that file is already there)
python -m core.pipeline     # once: audit, cleaning, models, REPORT.md  (~1 min). Optional: the app does it on first start.
streamlit run app.py
```

## Layout
| file | stage |
|---|---|
| `core/config.py` | data dictionary, plain-language labels, cost defaults |
| `core/quality.py` | B: audit, repairs with provenance, `validate()` (15 assertions), effect table |
| `core/features.py` | C: rescaling, one-hot, discretization, PCA comparison inputs; naive Bayes from scratch |
| `core/models.py` | C-E: feature selection vs PCA, tree / k-NN / naive Bayes, grid search, cost threshold, CV |
| `core/explain.py` | F: decision path, likelihood-ratio contributions, nearest cases, talking points |
| `core/report.py` | writes `REPORT.md` with every number from the run |
| `app.py` | G: six screens |

## Design decisions
* Label: Dropout = 1; Graduate and Enrolled = 0 ("Enrolled" has no final outcome yet).
* 2nd-semester columns are excluded: not known when the advisor asks.
* Gender, marital status, international: not used to rank people; gender is used only in the fairness table.
* Split: stratified random 80/20, because the data has no date column.
* Threshold: from the cost matrix on out-of-fold training predictions, never from the test set; default FN:FP = 5:1 (editable in Settings).
