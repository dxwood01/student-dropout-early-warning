"""Stage B: data-quality audit, repair with provenance, validate() (Lectures 2-3)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import (APP_MODES, BY, CANDIDATES, COURSES, FOLDS, KIND, MARITAL, RAW_CSV, CLEAN_CSV, S1, S2, SEED,
                     TARGET, Y)
from .features import Design

KNOWN_TYPOS = {"Nacionality": "Nationality"}
NOMINAL = [n for n, k in KIND.items() if k == "nominal"]
BINARY = [n for n, k in KIND.items() if k == "binary"]
GRADE_COLS = [S1["grade"], S2["grade"]]
FEATURES = list(BY)


def load_raw(path=RAW_CSV):
    """Read the UCI file whether it is ';'-separated (data.csv) or ','-separated (ucimlrepo export)."""
    with open(path, encoding="utf-8-sig") as f:
        head = f.readline()
    return pd.read_csv(path, sep=";" if head.count(";") > head.count(",") else ",", encoding="utf-8-sig")


def load_clean(path=CLEAN_CSV):
    df = pd.read_csv(path)
    for c in NOMINAL:
        df[c] = df[c].astype("category")
    return df


def strip_names(df):
    return df.rename(columns=lambda c: str(c).strip())


def fence(x, k=1.5):
    q1, q3 = x.quantile([0.25, 0.75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


def _n(df, a):
    return pd.to_numeric(df[a].astype(object), errors="coerce")


# ------------------------------------------------------------------ audit
def record_masks(df):
    """Boolean masks (one per record) for every record-level fault we test for."""
    m = {}
    f = [c for c in FEATURES if c in df]
    m["dup"] = df.duplicated()
    m["dup_attr"] = df.duplicated(subset=f) & ~df.duplicated()
    m["missing"] = df[f].isna().any(axis=1)
    for tag, S in (("s1", S1), ("s2", S2)):
        m[f"disguised_{tag}"] = (_n(df, S["evaluations"]) == 0) & (_n(df, S["grade"]) == 0)
    age = _n(df, "Age at enrollment")
    m["out_age"] = age > fence(age)[1]
    for a in ("Admission grade", "Previous qualification (grade)"):
        x = _n(df, a)
        lo, hi = fence(x)
        m[f"out_{a}"] = (x < lo) | (x > hi)
    ap1, en1, ev1, g1 = (_n(df, S1[k]) for k in ("approved", "enrolled", "evaluations", "grade"))
    ap2, en2, ev2, g2 = (_n(df, S2[k]) for k in ("approved", "enrolled", "evaluations", "grade"))
    m["inc_app_enr"] = (ap1 > en1) | (ap2 > en2)
    m["inc_app_eval"] = (ap1 > ev1) | (ap2 > ev2)
    m["inc_grade_noapp"] = ((g1 > 0) & (ap1 == 0)) | ((g2 > 0) & (ap2 == 0))
    m["inc_app_nograde"] = ((g1 == 0) & (ap1 > 0)) | ((g2 == 0) & (ap2 > 0))
    att, crs = _n(df, "Daytime/evening attendance"), _n(df, "Course")
    evening = crs.isin([8014, 9991])                     # the two evening programmes in the code book
    m["inc_attendance"] = (evening & (att == 1)) | (~evening & (att == 0))
    m["zero_units_s1"] = en1 == 0
    bad = pd.Series(False, index=df.index)
    for a in BINARY:
        bad |= ~_n(df, a).isin([0, 1])
    bad |= ~_n(df, "Application order").between(0, 9)
    bad |= ~_n(df, "Course").isin(list(COURSES)) | ~_n(df, "Application mode").isin(list(APP_MODES))
    bad |= ~_n(df, "Marital Status").isin(list(MARITAL))
    bad |= ~g1.between(0, 20) | ~g2.between(0, 20)
    bad |= ~_n(df, "Admission grade").between(0, 200) | ~_n(df, "Previous qualification (grade)").between(0, 200)
    bad |= age < 16
    m["invalid"] = bad
    m["label_pending"] = df[TARGET] == "Enrolled" if TARGET in df else pd.Series(False, index=df.index)
    return m


def audit(raw):
    """One row per fault found (or checked and not found), in the vocabulary of Lecture 2."""
    names_bad = [c for c in raw.columns if c != str(c).strip()]
    typos = [c for c in raw.columns if str(c).strip() in KNOWN_TYPOS]
    df = strip_names(raw)
    m = record_masks(df)
    wrong_type = [c for c in NOMINAL if c in df and pd.api.types.is_numeric_dtype(df[c])]
    near_const = [c for c in CANDIDATES if c in df and _n(df, c).value_counts(normalize=True).iloc[0] >= 0.97]
    n = len(df)
    rows = [
        ("Duplicate data", "Fully identical records", "df.duplicated().sum()", m["dup"].sum(), "dropped"),
        ("Duplicate data", "Identical attributes, different outcome label",
         "df.duplicated(subset=attributes) & ~df.duplicated()", m["dup_attr"].sum(), "reported only"),
        ("Missing values", "Explicit missing cells in an attribute", "df[attributes].isna().any(axis=1)",
         m["missing"].sum(), "none to repair"),
        ("Missing values", "Grade stored as 0 although no evaluation was taken (1st sem)",
         "(evaluations == 0) & (grade == 0)", m["disguised_s1"].sum(), "grade set to missing"),
        ("Missing values", "Same, 2nd semester", "(evaluations == 0) & (grade == 0)", m["disguised_s2"].sum(),
         "grade set to missing"),
        ("Outliers", "Age at enrolment above Tukey fence Q3 + 1.5 IQR", "age > Q3 + 1.5*IQR", m["out_age"].sum(),
         "capped at Q3 + 3 IQR"),
        ("Outliers", "Admission grade outside Tukey fences", "x < Q1-1.5*IQR or x > Q3+1.5*IQR",
         m["out_Admission grade"].sum(), "kept (plausible values)"),
        ("Outliers", "Previous qualification grade outside Tukey fences", "x < Q1-1.5*IQR or x > Q3+1.5*IQR",
         m["out_Previous qualification (grade)"].sum(), "kept (plausible values)"),
        ("Inconsistent values", "Units approved > units enrolled", "approved > enrolled",
         m["inc_app_enr"].sum(), "flagged, not altered"),
        ("Inconsistent values", "Units approved > evaluations taken", "approved > evaluations",
         m["inc_app_eval"].sum(), "flagged, not altered"),
        ("Inconsistent values", "Positive grade but no unit approved", "(grade > 0) & (approved == 0)",
         m["inc_grade_noapp"].sum(), "flagged, not altered"),
        ("Inconsistent values", "Units approved but grade is 0", "(grade == 0) & (approved > 0)",
         m["inc_app_nograde"].sum(), "flagged, not altered"),
        ("Inconsistent values", "Evening programme but daytime attendance (or the reverse)",
         "(course in {8014, 9991}) != (attendance == 0)", m["inc_attendance"].sum(), "flagged, not altered"),
        ("Inconsistent values", "Enrolled in a programme but 0 units enrolled in 1st semester",
         "enrolled (1st sem) == 0", m["zero_units_s1"].sum(), "flagged, not altered"),
        ("Out-of-domain values", "Code or value outside the documented domain",
         "isin(documented codes) / between(lo, hi)", m["invalid"].sum(), "none to repair"),
        ("Wrong data type", f"{len(wrong_type)} nominal code attributes stored as integers (arithmetic is meaningless)",
         "is_numeric_dtype(col) for nominal attributes", n if wrong_type else 0, "cast to category"),
        ("Inconsistent naming", f"Column names with stray whitespace ({len(names_bad)}) or misspelt "
                                f"({', '.join(typos) or 'none'})", "[c for c in cols if c != c.strip()]",
         n if (names_bad or typos) else 0, "whitespace stripped; typo documented, name kept to match the code book"),
        ("Ambiguous label", "Target = 'Enrolled': outcome not final yet", "Target == 'Enrolled'",
         m["label_pending"].sum(), "treated as 'not dropped out so far'"),
        ("Near-constant attribute", f"Attribute with >= 97% one value: {', '.join(near_const) or 'none'}",
         "value_counts(normalize=True).max() >= .97", n if near_const else 0, "kept; feature selection decides"),
    ]
    out = pd.DataFrame(rows, columns=["category", "fault", "detecting expression", "records_affected", "handling"])
    out["records_affected"] = out["records_affected"].astype(int)
    out["share"] = (out["records_affected"] / n).round(4)
    return out


# ----------------------------------------------------------------- repair
ALL_REPAIRS = ("names", "duplicates", "disguised_missing", "age_cap", "flags", "target", "types")


def clean(raw, only=None):
    """Return (cleaned frame, repair log). `only` restricts to some repairs (used to measure each one)."""
    only = set(ALL_REPAIRS if only is None else only)
    df = strip_names(raw).copy()
    log = []
    if "names" in only:
        changed = [c for c in raw.columns if c != str(c).strip()]
        log.append(dict(repair="names", field=", ".join(changed) or "-", records=len(changed),
                        rule="strip whitespace from column names"))
    if "duplicates" in only:
        d = df.duplicated()
        df = df[~d].reset_index(drop=True)
        log.append(dict(repair="duplicates", field="(rows)", records=int(d.sum()), rule="drop repeated rows"))
    if "disguised_missing" in only:
        for g, e in zip(GRADE_COLS, (S1["evaluations"], S2["evaluations"])):
            mk = (_n(df, e) == 0) & (_n(df, g) == 0)
            df[f"prov__{g}"] = None
            df.loc[mk, g] = np.nan
            df.loc[mk, f"prov__{g}"] = "grade 0 with no evaluation -> set to missing"
            log.append(dict(repair="disguised_missing", field=g, records=int(mk.sum()),
                            rule="grade 0 and evaluations 0 -> NaN (imputed later, inside each model)"))
    if "age_cap" in only:
        hi = float(np.floor(fence(_n(df, "Age at enrollment"), 3)[1]))
        df["Age at enrollment"] = _n(df, "Age at enrollment")
        mk = df["Age at enrollment"] > hi
        df["prov__Age at enrollment"] = None
        df.loc[mk, "prov__Age at enrollment"] = f"age capped at {hi:g} (Q3 + 3 IQR)"
        df.loc[mk, "Age at enrollment"] = hi
        log.append(dict(repair="age_cap", field="Age at enrollment", records=int(mk.sum()),
                        rule=f"cap at Q3 + 3 IQR = {hi:g}"))
    if "flags" in only:
        m = record_masks(df)
        for k, col in (("inc_app_enr", "flag__approved_gt_enrolled"), ("inc_app_eval", "flag__approved_gt_evaluations"),
                       ("inc_grade_noapp", "flag__grade_without_approval"), ("inc_app_nograde", "flag__approved_no_grade"),
                       ("inc_attendance", "flag__attendance_vs_course"), ("zero_units_s1", "flag__no_units_1st_sem")):
            df[col] = m[k].astype(int)
            log.append(dict(repair="flags", field=col, records=int(m[k].sum()),
                            rule="flag column added; values kept (cannot tell which field is wrong)"))
    if "target" in only:
        df[Y] = (df[TARGET] == "Dropout").astype(int)
        df["prov__Dropout"] = np.where(df[TARGET] == "Enrolled", "Enrolled -> 0 (not dropped out so far)", None)
        log.append(dict(repair="target", field=Y, records=int((df[TARGET] == "Enrolled").sum()),
                        rule="Dropout = 1; Graduate and Enrolled = 0"))
    if "types" in only:
        for c in NOMINAL:
            df[c] = df[c].astype("category")
        log.append(dict(repair="types", field=", ".join(NOMINAL), records=len(NOMINAL),
                        rule="nominal code attributes cast to category"))
    return df, pd.DataFrame(log)


def quick_auc(df):
    """5-fold CV AUC of a small tree on all candidate attributes: the 'effect' number of a repair."""
    d = strip_names(df)
    y = (d[TARGET] == "Dropout").astype(int).values
    pipe = Pipeline([("x", Design(CANDIDATES, "tree", scale=False)),
                     ("m", DecisionTreeClassifier(max_depth=5, min_samples_leaf=25, random_state=SEED))])
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    return float(cross_val_score(pipe, d, y, cv=cv, scoring="roc_auc").mean())


def effects(raw):
    """Before/after table: records changed, field mean before -> after, CV AUC before -> after."""
    base = quick_auc(raw)
    raw_s = strip_names(raw)
    rows = []
    focus = {"names": None, "duplicates": None, "disguised_missing": S1["grade"], "age_cap": "Age at enrollment",
             "flags": None, "target": None, "types": None}
    for r in ALL_REPAIRS:
        cl, lg = clean(raw, only={r})
        field = focus[r]
        before = after = np.nan
        if field:
            before, after = _n(raw_s, field).mean(), _n(cl, field).mean()
        rows.append(dict(repair=r, records_changed=int(lg["records"].sum()),
                         field_mean_before=before, field_mean_after=after, cv_auc_before=base,
                         cv_auc_after=quick_auc(cl) if r not in ("target", "types", "names", "flags") else base))
    rows.append(dict(repair="ALL repairs together", records_changed=np.nan, field_mean_before=np.nan,
                     field_mean_after=np.nan, cv_auc_before=base, cv_auc_after=quick_auc(clean(raw)[0].drop(columns=[Y]))))
    out = pd.DataFrame(rows)
    out["auc_change"] = out["cv_auc_after"] - out["cv_auc_before"]
    return out


# --------------------------------------------------------------- validate
def validate(df):
    """Fifteen assertions. Returns a list of (assertion, passed, detail). Passes on cleaned data, fails on raw."""
    d, out = df, []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), detail))

    def col(c):
        return _n(d, c) if c in d else pd.Series(np.nan, index=d.index)

    add("1 column names have no stray whitespace", all(c == str(c).strip() for c in d.columns))
    add("2 no fully duplicated rows", not d.duplicated().any(), f"{int(d.duplicated().sum())} duplicates")
    miss = d[[c for c in FEATURES if c in d]].isna()
    allowed = pd.DataFrame(False, index=d.index, columns=miss.columns)
    for g, e in zip(GRADE_COLS, (S1["evaluations"], S2["evaluations"])):
        if g in allowed:
            allowed[g] = col(e) == 0
    add("3 missing values only where a grade cannot exist (no evaluation)", not (miss & ~allowed).any().any())
    z = int(sum(((col(e) == 0) & (col(g) == 0)).sum() for g, e in zip(GRADE_COLS, (S1["evaluations"], S2["evaluations"]))))
    add("4 no grade of 0 disguised as a value when no evaluation was taken", z == 0, f"{z} records")
    add("5 nominal code attributes are categorical, not numeric",
        all(isinstance(d[c].dtype, pd.CategoricalDtype) for c in NOMINAL if c in d))
    age = col("Age at enrollment")
    add("6 age not above Q3 + 3 IQR", bool((age <= fence(age.dropna(), 3)[1] + 1e-9).all()))
    add("7 binary attributes contain only 0/1", all(col(c).dropna().isin([0, 1]).all() for c in BINARY))
    add("8 application order within 0-9", bool(col("Application order").between(0, 9).all()))
    add("9 programme / application-mode / marital codes are documented codes",
        bool(col("Course").isin(list(COURSES)).all() and col("Application mode").isin(list(APP_MODES)).all()
             and col("Marital Status").isin(list(MARITAL)).all()))
    add("10 grades in 0-20 (semester) and 0-200 (admission, previous)",
        bool(all(col(g).dropna().between(0, 20).all() for g in GRADE_COLS)
             and col("Admission grade").between(0, 200).all() and col("Previous qualification (grade)").between(0, 200).all()))
    cnt = [S1[k] for k in ("credited", "enrolled", "evaluations", "approved", "without evaluations")]
    add("11 unit counts are non-negative", all((col(c).dropna() >= 0).all() for c in cnt))
    ok12 = "flag__approved_gt_enrolled" in d and bool(
        ((col(S1["approved"]) > col(S1["enrolled"])) | (col(S2["approved"]) > col(S2["enrolled"]))).astype(int)
        .le(d["flag__approved_gt_enrolled"]).all())
    add("12 every approved > enrolled record carries a flag", ok12)
    add("13 every repaired (missing) grade has a provenance entry",
        all(f"prov__{g}" in d and d.loc[d[g].isna(), f"prov__{g}"].notna().all() for g in GRADE_COLS if g in d))
    add("14 Dropout label present and consistent with Target",
        Y in d and TARGET in d and bool(((d[TARGET] == "Dropout").astype(int) == d[Y]).all()))
    add("15 Target has only Dropout / Enrolled / Graduate",
        TARGET in d and bool(d[TARGET].isin(["Dropout", "Enrolled", "Graduate"]).all()))
    return out


def validate_table(df):
    return pd.DataFrame(validate(df), columns=["assertion", "passed", "detail"])
