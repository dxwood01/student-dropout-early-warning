"""Pre-processing (Lectures 3-4) and a from-scratch categorical naive Bayes (Lecture 8).

`Design` turns the cleaned table into the matrix each model family needs:
  tree : numeric as-is, nominal attributes one-hot (binarization), NO rescaling
  knn  : same matrix, min-max rescaled to [0, 1] so no attribute dominates the distance
  nb   : every attribute becomes a category code; numeric attributes are discretized
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin

from .config import BY, KIND, LABEL, value_text


def num(X, a):
    return pd.to_numeric(X[a].astype(object), errors="coerce")


def fmt(x):
    return f"{x:g}"


class Design(TransformerMixin, BaseEstimator):
    def __init__(self, attrs=None, mode="tree", n_bins=5, strategy="quantile", scale=True):
        self.attrs = attrs
        self.mode = mode
        self.n_bins = n_bins
        self.strategy = strategy
        self.scale = scale

    # ---------------------------------------------------------------- fit
    def fit(self, X, y=None):
        X = pd.DataFrame(X).reset_index(drop=True)
        self.attrs_ = list(self.attrs)
        self.median_ = {a: float(num(X, a).median()) for a in self.attrs_ if KIND[a] != "nominal"}
        if self.mode in ("tree", "knn"):
            self.levels_ = {a: sorted(num(X, a).dropna().unique()) for a in self.attrs_ if KIND[a] == "nominal"}
            M = self._matrix(X)
            self.cols_ = list(M.columns)
            if self.mode == "knn" and self.scale:
                self.lo_ = M.min().values
                rng = M.max().values - self.lo_
                self.rng_ = np.where(rng == 0, 1.0, rng)
        else:
            self._fit_codes(X)
        return self

    def _matrix(self, X):
        parts = {}
        for a in self.attrs_:
            v = num(X, a)
            if KIND[a] == "nominal":
                for lv in self.levels_[a]:
                    parts[f"{a}={int(lv)}"] = (v == lv).astype(float)
            else:
                parts[a] = v.fillna(self.median_[a]).astype(float)
        return pd.DataFrame(parts)

    def _fit_codes(self, X):
        self.spec_ = {}
        for a in self.attrs_:
            v = num(X, a)
            if KIND[a] in ("nominal", "binary", "ordinal"):
                self.spec_[a] = {"cat": True, "levels": [float(x) for x in sorted(v.dropna().unique())]}
            else:
                x = v.fillna(self.median_[a])
                inner = np.linspace(0, 1, self.n_bins + 1)[1:-1]
                if self.strategy == "quantile":
                    edges = np.unique(np.quantile(x, inner))
                else:
                    edges = np.unique(np.linspace(x.min(), x.max(), self.n_bins + 1)[1:-1])
                self.spec_[a] = {"cat": False, "edges": edges}

    # ---------------------------------------------------------- transform
    def transform(self, X):
        X = pd.DataFrame(X).reset_index(drop=True)
        if self.mode in ("tree", "knn"):
            M = self._matrix(X)
            if self.mode == "knn" and self.scale:
                return ((M.values - self.lo_) / self.rng_).clip(0, 1)
            return M.values
        cols = []
        for a in self.attrs_:
            v, s = num(X, a), self.spec_[a]
            if s["cat"]:
                code = np.full(len(v), len(s["levels"]))            # last code = "unseen level"
                for i, lv in enumerate(s["levels"]):
                    code[(v == lv).values] = i
            else:
                code = np.searchsorted(s["edges"], v.fillna(self.median_[a]).values, side="left")
            cols.append(code)
        return np.column_stack(cols).astype(int)

    def code_label(self, a, code):
        """Words for one category code of attribute a (naive Bayes explanations)."""
        s = self.spec_[a]
        if s["cat"]:
            return value_text(a, s["levels"][code]) if code < len(s["levels"]) else "a value not seen in training"
        e = s["edges"]
        if len(e) == 0:
            return "any value"
        if code == 0:
            return f"at most {fmt(e[0])}"
        if code >= len(e):
            return f"more than {fmt(e[-1])}"
        return f"between {fmt(e[code - 1])} (excl.) and {fmt(e[code])}"


class CatNB(ClassifierMixin, BaseEstimator):
    """Categorical naive Bayes with Laplace smoothing; log-likelihood ratios are the explanation."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = np.asarray(X, int), np.asarray(y, int)
        self.classes_ = np.array([0, 1])
        self.n_cat_ = X.max(axis=0) + 1
        cnt = np.array([(y == c).sum() for c in (0, 1)], float)
        self.log_prior_ = np.log(cnt / cnt.sum())
        self.log_lik_ = []
        for j, k in enumerate(self.n_cat_):
            tab = np.zeros((2, k + 1))                                # last column = unseen category
            for c in (0, 1):
                counts = np.bincount(X[y == c, j], minlength=k + 1).astype(float)
                tab[c] = (counts + self.alpha) / (cnt[c] + self.alpha * (k + 1))
            self.log_lik_.append(np.log(tab))
        return self

    def _idx(self, X):
        X = np.asarray(X, int)
        return np.minimum(X, self.n_cat_)

    def contributions(self, X):
        """Per-attribute log-likelihood ratio log P(x|dropout) - log P(x|stay); shape (n, n_attrs)."""
        I = self._idx(X)
        return np.column_stack([self.log_lik_[j][1, I[:, j]] - self.log_lik_[j][0, I[:, j]] for j in range(I.shape[1])])

    def predict_proba(self, X):
        z = self.log_prior_[1] - self.log_prior_[0] + self.contributions(X).sum(axis=1)
        p1 = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
