"""Student Dropout Early-Warning Console: run with `streamlit run app.py`."""
import html
import io
from datetime import date

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

from core import explain as E
from core import models as M
from core.config import BUNDLE, CLEAN_CSV, INTEGER_ATTRS, KIND, LABEL, RAW_CSV, S1, value_text
from core.features import num
from core.quality import strip_names

st.set_page_config(page_title="Dropout Early-Warning Console", page_icon="🎓", layout="wide")


# ----------------------------------------------------------------- loading
@st.cache_resource(show_spinner="First start: training the models (about a minute)...")
def load():
    try:
        return joblib.load(BUNDLE)
    except Exception:                                   # missing or built by another library version: rebuild
        from core.pipeline import run
        return run(log=lambda *_: None)


if not BUNDLE.exists() and not RAW_CSV.exists():
    st.error("data/students_raw.csv is missing. Run `python fetch_data.py` once, then restart the app.")
    st.stop()
art = load()
ss = st.session_state
ss.setdefault("cfg", {"model": "tree", "costs": dict(art["costs"]), "thr": None})
cfg = ss.cfg
KEYS = list(M.MODEL_NAMES)


def opt_thr(key=None, costs=None):
    key = key or cfg["model"]
    return M.optimal_threshold(art["y_train"], art["results"][key]["oof"], costs or cfg["costs"])[0]


if cfg["thr"] is None:
    cfg["thr"] = opt_thr()
THR, COSTS, KEY = cfg["thr"], cfg["costs"], cfg["model"]


def band(p):
    return "Flag" if p >= THR else ("Watch" if p >= THR / 2 else "OK")


def course_name(v):
    return value_text("Course", v)


# -------------------------------------------------------- scoring helpers
def prepare_input(df):
    """Clean an uploaded caseload the same way the training data was cleaned, or say what is wrong."""
    df = strip_names(df.copy())
    miss = [a for a in art["attrs"] if a not in df.columns]
    if miss:
        return None, [f"missing column: {m}" for m in miss]
    notes = []
    for a in art["attrs"]:
        df[a] = pd.to_numeric(df[a], errors="coerce")
        bad = df[a].isna().sum()
        if bad:
            notes.append(f"{bad} unreadable/empty value(s) in '{a}' (replaced by the training median)")
    g, e = S1["grade"], S1["evaluations"]
    if g in df and e in df:
        mk = (num(df, e) == 0) & (num(df, g) == 0)
        df.loc[mk, g] = np.nan                           # same repair as in the cleaning step
        if mk.sum():
            notes.append(f"{int(mk.sum())} grade(s) of 0 with no evaluation treated as missing")
    return df, notes


@st.cache_data(show_spinner="Scoring...")
def score_frame(_art, key, thr, df):
    pipe = _art["pipes"][key]
    p = pipe.predict_proba(df)[:, 1]
    factors, tips = [], []
    for i in range(len(df)):
        r = df.iloc[[i]]
        ex = E.explain(_art, key, r)
        low = ex["p"] < thr / 2                            # nothing to explain for a clearly low score
        factors.append("no risk factor stands out" if low or not ex["factors"] else " • ".join(ex["factors"]))
        tips.append(E.talking_points(r)[0])
    return p, factors, tips


def scored_table(df):
    p, fac, tips = score_frame(art, KEY, THR, df)
    idcol = next((c for c in df.columns if c.lower() in ("studentid", "student_id", "id")), None)
    t = pd.DataFrame({"Student": df[idcol].astype(str).values if idcol else [f"S-{i + 1:04d}" for i in range(len(df))],
                      "Programme": [course_name(v) for v in df["Course"]] if "Course" in df else "",
                      "Risk score": p, "Status": [band(x) for x in p], "Main factors": fac,
                      "Suggested first conversation": tips})
    if "Semester" in df:
        t["Semester"] = df["Semester"].values
    t.index = df.index
    return t.sort_values("Risk score", ascending=False).reset_index().rename(columns={"index": "row"})


def risk_style(t):
    return t.style.format({"Risk score": "{:.0%}"}).background_gradient(subset=["Risk score"], cmap="Reds", vmin=0, vmax=1)


def get_caseload(prefix):
    up = st.file_uploader("Upload a CSV of students", type="csv", key=f"{prefix}_up")
    demo = st.checkbox("Use the held-out test students as a demo caseload", value=up is None, key=f"{prefix}_demo")
    if up is not None and not demo:
        raw = pd.read_csv(up, sep=None, engine="python")
        df, notes = prepare_input(raw)
        if df is None:
            st.error("The file cannot be scored. " + "; ".join(notes))
            st.download_button("Download a template with the right columns", template().to_csv(index=False),
                               "template.csv", key=f"{prefix}_tpl")
            return None
        for n in notes:
            st.warning(n)
        return df
    st.caption("Demo caseload: students the models never saw during training (885 held-out records).")
    return art["X_test"].copy()


def template():
    cols = art["attrs"] + (["Course"] if "Course" not in art["attrs"] else [])
    return art["X_test"][cols].head(5)


# ----------------------------------------------------------------- pages
def page_data():
    st.header("1 · Data & quality")
    a, b = art["n_raw"], art["n_clean"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw records", a)
    c2.metric("After cleaning", b)
    c3.metric("Attributes used by the product", len(art["attrs"]))
    st.subheader("Audit report")
    st.caption("Faults are named in the vocabulary of Lecture 2, each with the expression that detects it. "
               "Zero means: checked, not found.")
    st.dataframe(art["audit"], width="stretch", hide_index=True)
    st.subheader("Repairs and their measured effect")
    st.dataframe(art["effects"].round(4), width="stretch", hide_index=True)
    st.subheader("validate(): 15 assertions")
    v = art["validate_raw"].merge(art["validate_clean"], on="assertion", suffixes=(" (raw)", " (cleaned)"))
    st.dataframe(v[["assertion", "passed (raw)", "passed (cleaned)"]], width="stretch", hide_index=True)
    st.subheader("Data dictionary")
    st.dataframe(art["dictionary"], width="stretch", hide_index=True)
    st.subheader("Cleaned data")
    st.dataframe(pd.concat([art["X_train"], art["X_test"]]).head(200), width="stretch")
    st.download_button("Download cleaned file (CSV, with provenance columns)", CLEAN_CSV.read_bytes(),
                       "students_clean.csv", "text/csv")


def page_perf():
    st.header("2 · Model performance")
    y = art["y_test"]
    base = M.baseline(y, COSTS)
    rows = []
    for k in KEYS:
        r = art["results"][k]
        t = opt_thr(k)
        m5, mt = M.metrics(y, r["test_proba"], 0.5, COSTS), M.metrics(y, r["test_proba"], t, COSTS)
        rows.append({"Model": M.MODEL_NAMES[k], "AUC": r["auc_test"], "Threshold (cost-optimal)": t,
                     "Accuracy @0.5": m5["accuracy"], "Accuracy @threshold": mt["accuracy"],
                     "Baseline accuracy": base["accuracy"], "Recall": mt["recall"], "Baseline recall": 0.0,
                     "Precision": mt["precision"], "Cost / student": mt["cost_per_student"],
                     "Baseline cost / student": base["cost_per_student"],
                     "CV AUC (mean ± sd)": f"{r['folds'].auc.mean():.3f} ± {r['folds'].auc.std():.3f}"})
    st.dataframe(pd.DataFrame(rows).style.format(
        {c: "{:.3f}" for c in ["AUC", "Threshold (cost-optimal)", "Accuracy @0.5", "Accuracy @threshold", "Baseline accuracy",
                               "Recall", "Baseline recall", "Precision", "Cost / student", "Baseline cost / student"]}),
        width="stretch", hide_index=True)
    st.info("The baseline always predicts 'will not drop out': it looks accurate and catches nobody. "
            "Recall = share of real dropouts that get flagged; precision = share of flags that were right (advisor workload). "
            "The gap between 'Accuracy @0.5' and 'Accuracy @threshold' is the accuracy deliberately given up to catch more dropouts.")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        for k in KEYS:
            r = art["results"][k]
            fpr, tpr, _ = roc_curve(y, r["test_proba"])
            ax.plot(fpr, tpr, label=f"{M.MODEL_NAMES[k]} (AUC {r['auc_test']:.3f})")
            mt = M.metrics(y, r["test_proba"], opt_thr(k), COSTS)
            ax.plot(mt["fp"] / (mt["fp"] + mt["tn"]), mt["recall"], "o", color="k", ms=4)
        ax.plot([0, 1], [0, 1], "--", color="grey", label="chance / majority baseline")
        ax.set(xlabel="False-positive rate", ylabel="True-positive rate (recall)", title="ROC, test set (dots = operating points)")
        ax.legend(fontsize=7)
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        r = art["results"][KEY]
        t, grid, cs = M.optimal_threshold(art["y_train"], r["oof"], COSTS)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(grid, cs / len(art["y_train"]))
        ax.axvline(t, color="g", ls="--", label=f"cost-optimal {t:.2f}")
        ax.axvline(0.5, color="r", ls=":", label="default 0.5")
        ax.set(xlabel="Threshold", ylabel="Cost per student", title=f"Cost vs threshold ({M.MODEL_NAMES[KEY]}, CV predictions)")
        ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    st.subheader(f"Confusion matrix: {M.MODEL_NAMES[KEY]} at threshold {THR:.2f}")
    mt = M.metrics(y, art["results"][KEY]["test_proba"], THR, COSTS)
    cm = pd.DataFrame([[mt["tn"], mt["fp"]], [mt["fn"], mt["tp"]]], index=["Actually stayed", "Actually dropped out"],
                      columns=["Not flagged", "Flagged"])
    st.dataframe(cm)
    st.subheader("Cross-validation spread (training portion, 5 folds)")
    st.dataframe(art["results"][KEY]["folds"].round(3), width="stretch")
    st.subheader("Fairness check by gender (recall and flag rate)")
    st.dataframe(art["fairness"].round(3), width="stretch", hide_index=True)


def input_widget(a, defaults):
    X = art["X_train"]
    lab = LABEL[a][0].upper() + LABEL[a][1:]
    key = f"f_{a}"
    if KIND[a] == "binary":
        return st.selectbox(lab, [1, 0], format_func=lambda v: value_text(a, v), key=key,
                            index=0 if defaults[a] == 1 else 1)
    if KIND[a] in ("nominal", "ordinal"):
        lv = sorted(num(X, a).dropna().unique())
        return st.selectbox(lab, lv, format_func=lambda v: value_text(a, v) if KIND[a] == "nominal" else f"{int(v)}",
                            key=key, index=lv.index(defaults[a]) if defaults[a] in lv else 0)
    x = num(X, a)
    integer = a in INTEGER_ATTRS
    lo, hi = float(x.min()), float(x.max())
    if integer:
        return st.number_input(lab, int(lo), int(hi) + 10, int(defaults[a]), 1, key=key)
    return st.number_input(lab, lo - abs(lo) * 0.2 if lo > 0 else lo, hi * 1.1 if hi > 0 else hi + 1, float(defaults[a]),
                           0.1, key=key)


def summary_html(sid, prog, p, status, ex, tips, vals):
    rows = "".join(f"<tr><td>{html.escape(LABEL[a])}</td><td>{html.escape(value_text(a, v))}</td></tr>" for a, v in vals.items())
    li = lambda xs: "".join(f"<li>{html.escape(x)}</li>" for x in xs)
    return f"""<html><head><meta charset='utf-8'><title>Student summary {html.escape(str(sid))}</title><style>
body{{font-family:Arial,sans-serif;max-width:780px;margin:24px auto;color:#222}} h1{{font-size:22px;margin-bottom:0}}
.big{{font-size:34px;font-weight:bold}} .Flag{{color:#b00020}} .Watch{{color:#b26a00}} .OK{{color:#1b7f3b}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td{{border-bottom:1px solid #ddd;padding:3px 6px}}
@media print{{body{{margin:0}}}}</style></head><body>
<h1>Student {html.escape(str(sid))} · mid-semester review</h1><div>{html.escape(prog)} · {date.today():%d %B %Y}</div>
<p><span class='big {status}'>{p:.0%}</span> estimated risk of dropping out · status <b class='{status}'>{status}</b>
(flag threshold {THR:.0%}, model: {M.MODEL_NAMES[KEY]})</p>
<h3>Main factors</h3><ul>{li(ex['factors']) or '<li>No single factor stands out</li>'}</ul>
<h3>Why the model says this</h3><ul>{li(ex['reasons'][:6])}</ul><p><i>{html.escape(ex['support'])}</i></p>
<h3>Suggested first conversation</h3><ul>{li(tips)}</ul>
<h3>Record used by the model</h3><table>{rows}</table>
<p style='font-size:11px;color:#666'>A score is a prompt for a conversation, not a verdict. Advisor judgement comes first.</p>
</body></html>"""


def explain_block(ex, row):
    if ex["p"] < THR / 2:
        st.markdown("**Main factors:** none stand out; this student's score is low.")
    elif ex["factors"]:
        st.markdown("**Main factors driving this score**\n\n" + "\n".join(f"- {f}" for f in ex["factors"]))
    st.markdown("**Reasons**\n\n" + "\n".join(f"- {r}" for r in ex["reasons"]))
    st.caption(ex["support"])
    if ex["detail"] is not None:
        with st.expander("Detail" if KEY == "nb" else "The most similar past students"):
            st.dataframe(ex["detail"], width="stretch", hide_index=True)
    st.markdown("**Suggested first conversation**\n\n" + "\n".join(f"- {t}" for t in E.talking_points(row)))


def page_single():
    st.header("3 · Single prediction")
    st.caption(f"Model: {M.MODEL_NAMES[KEY]} · flag threshold {THR:.0%} (change both in Settings).")
    X = art["X_train"]
    defaults = {a: (num(X, a).median() if KIND[a] in ("numeric", "ordinal") else num(X, a).mode().iloc[0]) for a in art["attrs"]}

    def load_example():
        r = art["X_test"].sample(1, random_state=int(np.random.randint(1e6))).iloc[0]
        for a in art["attrs"]:
            ss[f"f_{a}"] = int(r[a]) if KIND[a] != "numeric" or a in INTEGER_ATTRS else float(r[a])
    st.button("Fill the form with a random student the models have not seen", on_click=load_example)
    cols = st.columns(3)
    vals = {}
    for i, a in enumerate(art["attrs"]):
        with cols[i % 3]:
            vals[a] = input_widget(a, defaults)
    row = pd.DataFrame([vals])
    ex = E.explain(art, KEY, row)
    p = ex["p"]
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated risk of dropping out", f"{p:.0%}")
    c2.metric("Status", band(p), help=f"Flag if at or above {THR:.0%}; Watch if at or above {THR / 2:.0%}")
    c3.metric("Distance from flag threshold", f"{(p - THR) * 100:+.0f} points")
    st.progress(min(max(p, 0.0), 1.0))
    explain_block(ex, row)
    st.download_button("Printable one-page summary (open, then print)",
                       summary_html("(manual entry)", course_name(vals["Course"]) if "Course" in vals else "", p, band(p), ex,
                                    E.talking_points(row), vals), "student_summary.html", "text/html")


def page_batch():
    st.header("4 · Batch scoring")
    df = get_caseload("batch")
    if df is None:
        return
    t = scored_table(df)
    f1, f2, f3 = st.columns(3)
    status = f1.multiselect("Status", ["Flag", "Watch", "OK"], ["Flag", "Watch"])
    progs = sorted(t["Programme"].unique()) if "Programme" in t and t["Programme"].ne("").any() else []
    chosen = f2.multiselect("Programme", progs)
    lo = f3.slider("Minimum risk", 0.0, 1.0, 0.0, 0.01)
    v = t[t["Status"].isin(status) & (t["Risk score"] >= lo)]
    if chosen:
        v = v[v["Programme"].isin(chosen)]
    st.write(f"{len(v)} of {len(t)} students shown · {int((t['Status'] == 'Flag').sum())} flagged at threshold {THR:.0%}")
    st.dataframe(risk_style(v.drop(columns=["row"]).reset_index(drop=True)), width="stretch", hide_index=True)
    st.download_button("Download this table (CSV)", v.drop(columns=["row"]).to_csv(index=False), "scored_students.csv", "text/csv")


def page_settings():
    st.header("5 · Settings")
    m = st.selectbox("Model", KEYS, index=KEYS.index(KEY), format_func=M.MODEL_NAMES.get)
    st.subheader("Cost matrix (units: one advisor meeting)")
    c = st.columns(4)
    new = {"TN": c[0].number_input("Correctly left alone (TN)", value=float(COSTS["TN"]), step=0.5),
           "FP": c[1].number_input("Unneeded flag (FP)", value=float(COSTS["FP"]), step=0.5),
           "FN": c[2].number_input("Missed dropout (FN)", value=float(COSTS["FN"]), step=0.5),
           "TP": c[3].number_input("Helpful flag (TP)", value=float(COSTS["TP"]), step=0.5)}
    if m != KEY or new != COSTS:
        cfg["model"], cfg["costs"] = m, new
        cfg["thr"] = opt_thr(m, new)
        st.rerun()
    if st.button("Reset threshold to the cost-optimal value"):
        cfg["thr"] = opt_thr()
        st.rerun()
    thr = st.slider("Decision threshold: flag a student when risk ≥", 0.01, 0.99, float(THR), 0.01)
    cfg["thr"] = thr
    st.caption(f"Cost-optimal threshold for these costs (from cross-validation predictions): {opt_thr():.2f}. "
               f"Textbook value for calibrated probabilities: {M.theoretical_threshold(COSTS):.2f}.")
    y, p = art["y_test"], art["results"][KEY]["test_proba"]
    now, b0 = M.metrics(y, p, thr, COSTS), M.baseline(y, COSTS)
    st.subheader("Consequences on the held-out test students")
    k = st.columns(5)
    k[0].metric("Flagged", now["flagged"], f"{now['flagged'] / len(y):.0%} of students")
    k[1].metric("Dropouts caught", f"{now['recall']:.0%}", f"baseline 0%")
    k[2].metric("Flags that were right", f"{now['precision']:.0%}")
    k[3].metric("Accuracy", f"{now['accuracy']:.1%}", f"baseline {b0['accuracy']:.1%}", delta_color="off")
    k[4].metric("Cost per student", f"{now['cost_per_student']:.3f}", f"baseline {b0['cost_per_student']:.3f}", delta_color="inverse")
    st.dataframe(pd.DataFrame([[now["tn"], now["fp"]], [now["fn"], now["tp"]]], index=["Actually stayed", "Actually dropped out"],
                              columns=["Not flagged", "Flagged"]))


def page_caseload():
    st.header("6 · Advisor caseload")
    st.caption("Your tutees, riskiest first. Pick a student for a printable one-page summary.")
    df = get_caseload("case")
    if df is None:
        return
    t = scored_table(df)
    f1, f2, f3 = st.columns(3)
    progs = sorted(t["Programme"].unique()) if t["Programme"].ne("").any() else []
    chosen = f1.multiselect("Programme", progs)
    sems = sorted(t["Semester"].astype(str).unique()) if "Semester" in t else []
    sem = f2.multiselect("Semester", sems) if sems else []
    if not sems:
        f2.caption("No Semester column in this file, so no semester filter (the UCI data has none).")
    status = f3.multiselect("Status", ["Flag", "Watch", "OK"], ["Flag", "Watch"])
    v = t[t["Status"].isin(status)]
    if chosen:
        v = v[v["Programme"].isin(chosen)]
    if sem:
        v = v[v["Semester"].astype(str).isin(sem)]
    v = v.reset_index(drop=True)
    st.dataframe(risk_style(v.drop(columns=["row"])), width="stretch", hide_index=True)
    if v.empty:
        return
    sid = st.selectbox("Student", v["Student"] + "  ·  " + v["Risk score"].map("{:.0%}".format))
    rec = v.iloc[list(v["Student"] + "  ·  " + v["Risk score"].map("{:.0%}".format)).index(sid)]
    row = df.loc[[rec["row"]]]
    ex = E.explain(art, KEY, row)
    c1, c2 = st.columns([1, 2])
    c1.metric("Risk", f"{ex['p']:.0%}", band(ex["p"]), delta_color="off")
    with c2:
        explain_block(ex, row)
    vals = {a: row[a].iloc[0] for a in art["attrs"]}
    st.download_button("Printable one-page summary (open, then print)",
                       summary_html(rec["Student"], rec["Programme"], ex["p"], band(ex["p"]), ex, E.talking_points(row), vals),
                       f"summary_{rec['Student']}.html", "text/html")


PAGES = {"1 · Data & quality": page_data, "2 · Model performance": page_perf, "3 · Single prediction": page_single,
         "4 · Batch scoring": page_batch, "5 · Settings": page_settings, "6 · Advisor caseload": page_caseload}
st.sidebar.title("🎓 Dropout Early-Warning")
choice = st.sidebar.radio("Screen", list(PAGES), label_visibility="collapsed")
st.sidebar.caption(f"Model: **{M.MODEL_NAMES[KEY]}**  \nFlag threshold: **{THR:.0%}**  \nMiss costs {COSTS['FN']:g}× a meeting")
PAGES[choice]()
