"""Central configuration, data dictionary and plain-language labels."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, ART = ROOT / "data", ROOT / "artifacts"
RAW_CSV = DATA / "students_raw.csv"
CLEAN_CSV = DATA / "students_clean.csv"
BUNDLE = ART / "bundle.joblib"
TARGET, Y = "Target", "Dropout"          # Y = 1 means the student dropped out
SEED, TEST_SIZE, FOLDS = 42, 0.2, 5
# Cost matrix in "advisor-meeting units": a meeting (any flag) costs 1, a missed dropout costs 5.
DEFAULT_COSTS = {"TN": 0.0, "FP": 1.0, "FN": 5.0, "TP": 1.0}

SEM_KEYS = ["credited", "enrolled", "evaluations", "approved", "grade", "without evaluations"]
S1 = {k: f"Curricular units 1st sem ({k})" for k in SEM_KEYS}
S2 = {k: f"Curricular units 2nd sem ({k})" for k in SEM_KEYS}

COURSES = {33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
           8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
           9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
           9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
           9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
           9773: "Journalism and Communication", 9853: "Basic Education",
           9991: "Management (evening)"}
APP_MODES = {1: "1st phase, general contingent", 2: "Ordinance 612/93", 5: "1st phase, special contingent (Azores)",
             7: "Holder of other higher course", 10: "Ordinance 854-B/99", 15: "International student (bachelor)",
             16: "1st phase, special contingent (Madeira)", 17: "2nd phase, general contingent",
             18: "3rd phase, general contingent", 26: "Ordinance 533-A/99 b2 (different plan)",
             27: "Ordinance 533-A/99 b3 (other institution)", 39: "Over 23 years old", 42: "Transfer",
             43: "Change of course", 44: "Technological specialisation diploma", 51: "Change of institution/course",
             53: "Short-cycle diploma holder", 57: "Change of institution/course (international)"}
MARITAL = {1: "single", 2: "married", 3: "widower", 4: "divorced", 5: "facto union", 6: "legally separated"}


def _a(name, label, mtype, kind, rng, role="explanatory", note="", yes="", no=""):
    return dict(name=name, label=label, mtype=mtype, kind=kind, range=rng, role=role, note=note, yes=yes, no=no)


SENS = "Sensitive personal attribute: kept for the fairness audit only, never used to rank a student."
CODE = "Unordered code with 17-46 levels: cannot be explained to an advisor in plain words, so not used."
FUT = "NOT known when the advisor asks (2nd-semester result, available only after the decision point)."

ATTRS = [
    _a("Marital Status", "marital status", "nominal", "nominal", "codes 1-6", "audit_only", SENS),
    _a("Application mode", "how the student applied", "nominal", "nominal", "18 documented codes"),
    _a("Application order", "application choice (0 = first choice)", "ordinal", "ordinal", "0-9"),
    _a("Course", "programme", "nominal", "nominal", "17 programme codes"),
    _a("Daytime/evening attendance", "class schedule", "nominal", "binary", "1 = daytime, 0 = evening",
       yes="attends daytime classes", no="attends evening classes"),
    _a("Previous qualification", "previous qualification code", "nominal", "nominal", "17 codes", "excluded_code", CODE),
    _a("Previous qualification (grade)", "grade of previous qualification (0-200)", "interval", "numeric", "0-200"),
    _a("Nacionality", "nationality code", "nominal", "nominal", "21 codes", "excluded_code", CODE),
    _a("Mother's qualification", "mother's qualification code", "nominal", "nominal", "29 codes", "excluded_code", CODE),
    _a("Father's qualification", "father's qualification code", "nominal", "nominal", "34 codes", "excluded_code", CODE),
    _a("Mother's occupation", "mother's occupation code", "nominal", "nominal", "32 codes", "excluded_code", CODE),
    _a("Father's occupation", "father's occupation code", "nominal", "nominal", "46 codes", "excluded_code", CODE),
    _a("Admission grade", "admission grade (0-200)", "interval", "numeric", "0-200"),
    _a("Displaced", "living situation", "nominal", "binary", "1 = yes, 0 = no",
       yes="is a displaced student (lives away from home)", no="lives at home"),
    _a("Educational special needs", "special educational needs", "nominal", "binary", "1 = yes, 0 = no",
       yes="has special educational needs", no="has no special educational needs"),
    _a("Debtor", "debt to the university", "nominal", "binary", "1 = yes, 0 = no",
       yes="owes money to the university", no="owes nothing to the university"),
    _a("Tuition fees up to date", "tuition fees", "nominal", "binary", "1 = yes, 0 = no",
       yes="tuition fees are up to date", no="tuition fees are NOT up to date"),
    _a("Gender", "gender", "nominal", "binary", "1 = male, 0 = female", "audit_only", SENS, yes="male", no="female"),
    _a("Scholarship holder", "scholarship", "nominal", "binary", "1 = yes, 0 = no",
       yes="holds a scholarship", no="has no scholarship"),
    _a("Age at enrollment", "age at enrolment (years)", "ratio", "numeric", "17-70"),
    _a("International", "international student", "nominal", "binary", "1 = yes, 0 = no", "audit_only", SENS,
       yes="is an international student", no="is a domestic student"),
    _a(S1["credited"], "units credited from earlier study (1st sem)", "ratio", "numeric", "0-30"),
    _a(S1["enrolled"], "units enrolled (1st sem)", "ratio", "numeric", "0-30"),
    _a(S1["evaluations"], "exams/evaluations taken (1st sem)", "ratio", "numeric", "0-50"),
    _a(S1["approved"], "units passed (1st sem)", "ratio", "numeric", "0-30"),
    _a(S1["grade"], "average grade (1st sem, 0-20)", "interval", "numeric", "0-20; 0 = no grade"),
    _a(S1["without evaluations"], "units with no evaluation (1st sem)", "ratio", "numeric", "0-15"),
] + [_a(S2[k], f"{k} (2nd sem)", "interval" if k == "grade" else "ratio", "numeric",
        {"evaluations": "0-50", "grade": "0-20; 0 = no grade", "without evaluations": "0-15"}.get(k, "0-30"), "excluded_future", FUT)
     for k in SEM_KEYS] + [
    _a("Unemployment rate", "regional unemployment rate (%)", "ratio", "numeric", "7-17"),
    _a("Inflation rate", "inflation rate (%)", "interval", "numeric", "-1 to 4"),
    _a("GDP", "GDP growth (%)", "interval", "numeric", "-5 to 4"),
]

BY = {a["name"]: a for a in ATTRS}
KIND = {n: a["kind"] for n, a in BY.items()}
LABEL = {n: a["label"] for n, a in BY.items()}
CANDIDATES = [n for n, a in BY.items() if a["role"] == "explanatory"]       # 21 attributes
INTEGER_ATTRS = {S1[k] for k in SEM_KEYS if k != "grade"} | {"Age at enrollment"}


def value_text(attr, v):
    """Human wording of one attribute value."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "unknown"
    i = int(round(f))
    a = BY[attr]
    if attr == "Course":
        return f"{i} - {COURSES.get(i, 'unknown programme')}"
    if attr == "Application mode":
        return APP_MODES.get(i, f"code {i}")
    if attr == "Marital Status":
        return MARITAL.get(i, f"code {i}")
    if a["kind"] == "binary":
        return a["yes"] if i == 1 else a["no"]
    return f"{f:g}"
