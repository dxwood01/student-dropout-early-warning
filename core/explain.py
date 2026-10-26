"""Stage F: every prediction comes with its reason, in the advisor's language.

tree -> the decision path (conditions actually met by this student)
nb   -> the attributes that moved the posterior (log-likelihood ratios)
knn  -> the most similar past students and what happened to them
"""
import numpy as np
import pandas as pd

from .config import BY, INTEGER_ATTRS, KIND, LABEL, S1, value_text
from .features import fmt, num


def _rate(tree, n):
    v = tree.value[n][0]
    return float(v[1] / v.sum())


def _cond_text(col, went_left, thr, val):
    """One tree condition in words."""
    if "=" in col:                                      # one-hot nominal level
        a, lv = col.rsplit("=", 1)
        name = value_text(a, lv)
        return f"{LABEL[a]} is {name}" if val >= 0.5 else f"{LABEL[a]} is not {name}"
    a = BY[col]
    if a["kind"] == "binary":
        return f"student {a['yes'] if val >= 0.5 else a['no']}"
    if col in INTEGER_ATTRS:
        lo = int(np.floor(thr))
        rule = f"at most {lo}" if went_left else f"at least {lo + 1}"
    else:
        rule = f"at most {fmt(round(thr, 2))}" if went_left else f"above {fmt(round(thr, 2))}"
    return f"{LABEL[col]} is {rule} (this student: {fmt(round(val, 2))})"


def explain_tree(pipe, row):
    x, m = pipe.named_steps["x"], pipe.named_steps["m"]
    Z = x.transform(row)
    path = m.decision_path(Z).indices
    leaf = int(m.apply(Z)[0])
    t = m.tree_
    steps = []
    for i, n in enumerate(path[:-1]):
        f, thr = t.feature[n], t.threshold[n]
        left = Z[0, f] <= thr
        delta = _rate(t, path[i + 1]) - _rate(t, n)
        steps.append((_cond_text(x.cols_[f], left, thr, Z[0, f]), delta))
    support = (f"{int(t.n_node_samples[leaf])} past students with this profile: "
               f"{_rate(t, leaf):.0%} dropped out.")
    up = [s for s, d in sorted(steps, key=lambda s: -s[1]) if d > 0.02][:3]
    reasons = [f"{s}  ->  {'raises' if d > 0 else 'lowers'} risk by {abs(d) * 100:.0f} points" for s, d in steps]
    return dict(p=float(m.predict_proba(Z)[0, 1]), factors=up, reasons=reasons, support=support, detail=None)


def explain_nb(pipe, row):
    x, m = pipe.named_steps["x"], pipe.named_steps["m"]
    Z = x.transform(row)
    c = m.contributions(Z)[0]
    items = [(a, x.code_label(a, int(Z[0, j])), float(c[j])) for j, a in enumerate(x.attrs_)]
    items.sort(key=lambda t: -abs(t[2]))
    say = lambda a, lab: lab if KIND[a] == "binary" else f"{LABEL[a]}: {lab}"
    prior = float(np.exp(m.log_prior_[1]))
    detail = pd.DataFrame([{"attribute": LABEL[a], "this student": lab,
                            "effect on odds": f"x{np.exp(l):.2f}", "direction": "raises risk" if l > 0 else "lowers risk"}
                           for a, lab, l in items])
    up = [say(a, lab) for a, lab, l in sorted(items, key=lambda t: -t[2]) if l > 0.1][:3]
    reasons = [f"{say(a, lab)}  ->  odds of dropout x{np.exp(l):.2f}" for a, lab, l in items[:6]]
    return dict(p=float(m.predict_proba(Z)[0, 1]), factors=up, reasons=reasons,
                support=f"Starting point: {prior:.0%} of past students dropped out; each attribute then multiplies the odds.",
                detail=detail)


def explain_knn(pipe, row, X_train, y_train, nb_design=None, Zall=None):
    x, m = pipe.named_steps["x"], pipe.named_steps["m"]
    Z = x.transform(row)
    dist, idx = m.kneighbors(Z)
    idx = idx[0]
    k = len(idx)
    out = y_train[idx]
    detail = X_train.iloc[idx].copy()
    show = [a for a in x.attrs_][:8]
    detail = pd.DataFrame({LABEL[a]: [value_text(a, v) for v in detail[a]] for a in show})
    detail.insert(0, "outcome", np.where(out == 1, "dropped out", "did not drop out"))
    factors = []
    if nb_design is not None and out.sum() > 0:
        Zs = nb_design.transform(row)[0]
        Zn = nb_design.transform(X_train.iloc[idx[out == 1]])
        Zall = nb_design.transform(X_train) if Zall is None else Zall
        cand = []
        for j, a in enumerate(nb_design.attrs_):
            share = float((Zn[:, j] == Zs[j]).mean())
            base = float((Zall[:, j] == Zs[j]).mean())
            if share >= 0.6 and share > base * 1.15:
                lab = nb_design.code_label(a, int(Zs[j]))
                cand.append((share / max(base, 1e-9), lab if KIND[a] == "binary" else f"{LABEL[a]}: {lab}"))
        factors = [t for _, t in sorted(cand, reverse=True)][:3]
    return dict(p=float(m.predict_proba(Z)[0, 1]), factors=factors,
                reasons=[f"{int(out.sum())} of the {k} most similar past students dropped out."],
                support=f"Based on the {k} students most similar to this one.", detail=detail)


def explain(art, key, row):
    p = art["pipes"][key]
    if key == "tree":
        return explain_tree(p, row)
    if key == "nb":
        return explain_nb(p, row)
    nbd = art["pipes"]["nb"].named_steps["x"]
    if "_nb_codes" not in art:                       # cache the training codes once (used for every k-NN explanation)
        art["_nb_codes"] = nbd.transform(art["X_train"])
    return explain_knn(p, row, art["X_train"], art["y_train"], nbd, art["_nb_codes"])


# ------------------------------------------------------- suggested talking points
def talking_points(row):
    """First-conversation suggestions from the student's own values (only for attributes the model uses)."""
    g = lambda a: float(num(row, a).iloc[0]) if a in row and pd.notna(num(row, a).iloc[0]) else None
    out = []
    enr, app, ev = g(S1["enrolled"]), g(S1["approved"]), g(S1["evaluations"])
    if app is not None and (app <= 2 or (enr and app / enr < 0.5)):
        out.append((3, "Start with the units not passed: which ones, and what got in the way? Discuss tutoring, "
                       "resit dates and whether the workload is realistic."))
    if ev is not None and ev == 0:
        out.append((3, "No exams were taken in the 1st semester: ask about attendance and engagement before anything else."))
    if g("Tuition fees up to date") == 0 or g("Debtor") == 1:
        out.append((2, "Raise money early and kindly: payment plan, hardship fund or scholarship options."))
    if g("Age at enrollment") is not None and g("Age at enrollment") >= 25:
        out.append((1, "Ask about work and family commitments and whether the timetable still fits them."))
    if g("Application order") is not None and g("Application order") >= 3:
        out.append((1, "This was not their first-choice programme: explore whether they still want it or would prefer a transfer."))
    if g("Displaced") == 1:
        out.append((1, "Ask about housing, travel and who they can lean on locally."))
    if g("Scholarship holder") == 0 and (g(S1["grade"]) or 20) < 11:
        out.append((1, "Low grades and no scholarship: check whether they know which support schemes they qualify for."))
    out.sort(key=lambda t: -t[0])
    tips = [t for _, t in out][:2]
    return tips or ["Open with: 'How is the semester going, and what is the hardest part right now?'"]
