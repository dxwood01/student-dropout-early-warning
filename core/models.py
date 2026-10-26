"""Stages C-E: feature selection vs PCA, three model families, tuning, threshold, evaluation."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import CANDIDATES, DEFAULT_COSTS, FOLDS, KIND, SEED
from .features import CatNB, Design, num

MODEL_NAMES = {"tree": "Decision tree", "knn": "k-nearest neighbours", "nb": "Naive Bayes"}
SKF = lambda: StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)


def make_pipe(kind, attrs, **kw):
    """Default-hyper-parameter pipeline of one family (tuned later by GridSearchCV)."""
    if kind == "tree":
        return Pipeline([("x", Design(attrs, "tree", scale=kw.get("scale", False))),
                         ("m", DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=SEED))])
    if kind == "knn":
        return Pipeline([("x", Design(attrs, "knn", scale=kw.get("scale", True))),
                         ("m", KNeighborsClassifier(n_neighbors=21))])
    return Pipeline([("x", Design(attrs, "nb", n_bins=5)), ("m", CatNB(alpha=1.0))])


GRIDS = {
    "tree": {"m__max_depth": [3, 4, 5, 6, 8], "m__min_samples_leaf": [10, 25, 50], "m__criterion": ["gini", "entropy"]},
    "knn": {"m__n_neighbors": [5, 11, 21, 31, 51], "m__metric": ["euclidean", "manhattan"]},
    "nb": {"x__n_bins": [3, 5, 8], "x__strategy": ["quantile", "uniform"], "m__alpha": [0.5, 1.0, 2.0]},
}


# ------------------------------------------------------ feature selection
def mi_ranking(X, y, attrs=CANDIDATES):
    """Mutual information between each attribute and the label (numeric attributes cut into 10 equal-frequency bins)."""
    out = []
    for a in attrs:
        v = num(X, a)
        if KIND[a] == "numeric":
            c = pd.qcut(v.fillna(v.median()), 10, duplicates="drop", labels=False)
        else:
            c = v.fillna(-1).astype(int)
        out.append((a, float(mutual_info_score(y, c))))
    return sorted(out, key=lambda t: -t[1])


def cv_auc(factory, X, y, k=None, attrs=None):
    """Fold AUCs. With k, the top-k attributes are re-ranked inside every fold (no selection leakage)."""
    aucs = []
    for tr, va in SKF().split(X, y):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        cols = [a for a, _ in mi_ranking(Xtr, y[tr])][:k] if k else attrs
        pipe = factory(cols).fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[va], pipe.predict_proba(Xva)[:, 1]))
    return np.array(aucs)


def select_attributes(X, y, ks=(4, 6, 8, 10, 12, 15, 18, 21)):
    """Choose k: smallest k whose mean CV-AUC (average of the 3 families) is within 0.005 of the best."""
    ranking = mi_ranking(X, y)
    rows = []
    for k in ks:
        row = {"k": k}
        for m in MODEL_NAMES:
            a = cv_auc(lambda cols, m=m: make_pipe(m, cols), X, y, k=k)
            row[m] = a.mean()
            row[m + "_sd"] = a.std()
        row["mean"] = np.mean([row[m] for m in MODEL_NAMES])
        rows.append(row)
    tab = pd.DataFrame(rows)
    best = tab["mean"].max()
    k = int(tab.loc[tab["mean"] >= best - 0.005, "k"].min())
    return {"ranking": ranking, "table": tab, "k": k, "attrs": [a for a, _ in ranking][:k]}


def compare_reduction(X, y, attrs_sel):
    """Feature subset vs PCA, same model (k-NN): the Stage C comparison."""
    rows = []
    def add(name, factory, dims):
        a = cv_auc(factory, X, y, attrs=CANDIDATES if name != "selected subset" else attrs_sel)
        rows.append({"representation": name, "dimensions": dims, "cv_auc_mean": a.mean(), "cv_auc_sd": a.std()})
    full_dim = Design(CANDIDATES, "knn").fit(X).transform(X.head(5)).shape[1]
    sel_dim = Design(attrs_sel, "knn").fit(X).transform(X.head(5)).shape[1]
    add("all 21 attributes", lambda c: make_pipe("knn", c), full_dim)
    add("selected subset", lambda c: make_pipe("knn", c), sel_dim)
    pca = lambda c: Pipeline([("x", Design(c, "knn")), ("p", PCA(n_components=0.90, random_state=SEED)),
                              ("m", KNeighborsClassifier(n_neighbors=21))])
    p = pca(CANDIDATES).fit(X, y)
    add("PCA (90% variance) on all 21", pca, p.named_steps["p"].n_components_)
    return pd.DataFrame(rows)


def compare_scaling(X, y, attrs):
    """Rescaling decision per model family, measured (Stage C)."""
    rows = []
    for m, scales in (("tree", (False, True)), ("knn", (False, True))):
        for sc in scales:
            a = cv_auc(lambda c, m=m, sc=sc: make_pipe(m, c, scale=sc), X, y, attrs=attrs)
            rows.append({"model": MODEL_NAMES[m], "rescaled": "min-max" if sc else "no", "cv_auc_mean": a.mean(),
                         "cv_auc_sd": a.std()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ tuning
def tune(kind, X, y, attrs):
    gs = GridSearchCV(make_pipe(kind, attrs), GRIDS[kind], scoring="roc_auc", cv=SKF(), n_jobs=1, refit=True)
    gs.fit(X, y)
    return gs


# ------------------------------------------------- costs and thresholds
def confusion(y, p, thr):
    f = p >= thr
    y = np.asarray(y).astype(bool)
    return dict(tp=int((f & y).sum()), fp=int((f & ~y).sum()), fn=int((~f & y).sum()), tn=int((~f & ~y).sum()))


def total_cost(c, costs):
    return c["tp"] * costs["TP"] + c["fp"] * costs["FP"] + c["fn"] * costs["FN"] + c["tn"] * costs["TN"]


def optimal_threshold(y, p, costs):
    grid = np.linspace(0.01, 0.99, 99)
    cs = [total_cost(confusion(y, p, t), costs) for t in grid]
    return float(grid[int(np.argmin(cs))]), grid, np.array(cs)


def theoretical_threshold(costs):
    """Flag when expected cost of flagging is lower: p* = (FP-TN) / ((FP-TN) + (FN-TP)), for calibrated p."""
    a, b = costs["FP"] - costs["TN"], costs["FN"] - costs["TP"]
    return a / (a + b) if a + b > 0 else 0.5


def metrics(y, p, thr, costs):
    c = confusion(y, p, thr)
    n = len(y)
    prec = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    return dict(threshold=thr, accuracy=(c["tp"] + c["tn"]) / n, recall=rec, precision=prec,
                flagged=c["tp"] + c["fp"], cost=total_cost(c, costs), cost_per_student=total_cost(c, costs) / n, **c)


def baseline(y, costs):
    """Majority-class baseline = never flag anybody."""
    y = np.asarray(y)
    return metrics(y, np.zeros(len(y)), 0.5, costs)


def cv_fold_stats(est, X, y, thr, costs):
    """OOF probabilities + per-fold AUC / accuracy / recall / cost at the chosen threshold."""
    oof = np.zeros(len(y))
    folds = []
    for tr, va in SKF().split(X, y):
        m = clone(est).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        mt = metrics(y[va], oof[va], thr, costs)
        folds.append(dict(auc=roc_auc_score(y[va], oof[va]), accuracy=mt["accuracy"], recall=mt["recall"],
                          precision=mt["precision"], cost_per_student=mt["cost_per_student"],
                          baseline_accuracy=1 - y[va].mean()))
    return oof, pd.DataFrame(folds)


def fold_table(oof, y, thr, costs):
    """Per-fold metrics at the chosen threshold from out-of-fold probabilities (same splits as cv_fold_stats)."""
    rows = []
    for _, va in SKF().split(np.zeros(len(y)), y):
        mt = metrics(y[va], oof[va], thr, costs)
        rows.append(dict(auc=roc_auc_score(y[va], oof[va]), accuracy=mt["accuracy"], baseline_accuracy=1 - y[va].mean(),
                         recall=mt["recall"], precision=mt["precision"], cost_per_student=mt["cost_per_student"]))
    return pd.DataFrame(rows)
