"""Run the whole study once and save everything the app and the report need."""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from . import models as M
from .config import (ART, ATTRS, BUNDLE, CANDIDATES, CLEAN_CSV, DEFAULT_COSTS, RAW_CSV, S1, S2, SEED, TARGET,
                     TEST_SIZE, Y, FOLDS)
from .quality import audit, clean, effects, load_raw, strip_names, validate_table

warnings.filterwarnings("ignore")


def dictionary(raw):
    d = strip_names(raw)
    rows = []
    for a in ATTRS:
        role = {"explanatory": "explanatory attribute (candidate)", "audit_only": "must not be used (sensitive; fairness audit only)",
                "excluded_future": "must not be used (not known at decision time)",
                "excluded_code": "must not be used (unexplainable code)"}[a["role"]]
        v = pd.to_numeric(d[a["name"]], errors="coerce") if a["name"] in d else None
        obs = "" if v is None else (f"{v.min():g} to {v.max():g}" if a["kind"] in ("numeric", "ordinal") else f"{v.nunique()} distinct values")
        rows.append({"column": a["name"], "plain label": a["label"], "measurement type": a["mtype"],
                     "plausible range": a["range"], "observed in data": obs, "role": role, "note": a["note"]})
    rows.append({"column": TARGET, "plain label": "final outcome", "measurement type": "nominal", "plausible range":
                 "Dropout / Enrolled / Graduate", "observed in data": ", ".join(f"{k} {v}" for k, v in d[TARGET].value_counts().items()), "role": "target (source of the label)", "note": "Dropout = 1, others = 0"})
    rows.append({"column": Y, "plain label": "dropped out (0/1)", "measurement type": "nominal", "plausible range": "0/1",
                 "observed in data": "1 = Dropout, 0 = Graduate or Enrolled", "role": "target", "note": "derived by the cleaning step"})
    return pd.DataFrame(rows)


def run(log=print):
    t0 = time.time()
    ART.mkdir(exist_ok=True)
    raw = load_raw(RAW_CSV)
    log(f"[1/8] loaded {raw.shape[0]} rows x {raw.shape[1]} columns")
    aud = audit(raw)
    cl, rlog = clean(raw)
    cl.to_csv(CLEAN_CSV, index=False)
    eff = effects(raw)
    v_raw, v_clean = validate_table(strip_names(raw)), validate_table(cl)
    log(f"[2/8] audit done; validate(): raw fails {int((~v_raw.passed).sum())}/15, cleaned fails {int((~v_clean.passed).sum())}/15")

    df = cl.reset_index(drop=True)
    y = df[Y].values
    tr, te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE, stratify=y, random_state=SEED)
    Xtr, Xte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    ytr, yte = y[tr], y[te]
    log(f"[3/8] split: train {len(tr)} / test {len(te)}; dropout share train {ytr.mean():.3f}, test {yte.mean():.3f}")

    sel = M.select_attributes(Xtr, ytr)
    attrs = sel["attrs"]
    log(f"[4/8] selected {len(attrs)} attributes: {attrs}")
    reduction = M.compare_reduction(Xtr, ytr, attrs)
    scaling = M.compare_scaling(Xtr, ytr, attrs)

    costs = dict(DEFAULT_COSTS)
    pipes, results, tuning = {}, {}, []
    for key in M.MODEL_NAMES:
        gs = M.tune(key, Xtr, ytr, attrs)
        best = gs.best_estimator_
        oof, _ = M.cv_fold_stats(best, Xtr, ytr, 0.5, costs)
        thr, grid, cs = M.optimal_threshold(ytr, oof, costs)
        folds = M.fold_table(oof, ytr, thr, costs)
        p_test = best.predict_proba(Xte)[:, 1]
        pipes[key] = best
        results[key] = dict(params={k: (v.item() if hasattr(v, "item") else v) for k, v in gs.best_params_.items()},
                            tune_auc=float(gs.best_score_), oof=oof, test_proba=p_test, thr=thr,
                            thr_grid=grid, thr_cost=cs, folds=folds, auc_test=float(roc_auc_score(yte, p_test)))
        tuning.append(dict(model=M.MODEL_NAMES[key], best_params=json.dumps(results[key]["params"]),
                           cv_auc=gs.best_score_, grid_size=len(gs.cv_results_["params"])))
        log(f"[5/8] {key}: tuned {results[key]['params']}  CV-AUC {gs.best_score_:.3f}  threshold {thr:.2f}  test AUC {results[key]['auc_test']:.3f}")

    # fairness check: recall/flag rate by gender at each model's cost-optimal threshold
    fair = []
    for key, r in results.items():
        for g, name in ((0, "female"), (1, "male")):
            mk = (pd.to_numeric(Xte["Gender"].astype(object)) == g).values
            if mk.sum() and yte[mk].sum():
                mt = M.metrics(yte[mk], r["test_proba"][mk], r["thr"], costs)
                fair.append(dict(model=M.MODEL_NAMES[key], group=name, n=int(mk.sum()), dropout_rate=yte[mk].mean(),
                                 flagged_rate=mt["flagged"] / mk.sum(), recall=mt["recall"], precision=mt["precision"]))

    bundle = dict(pipes=pipes, results=results, attrs=attrs, sel=sel, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
                  costs=costs, audit=aud, repairs=rlog, effects=eff, validate_raw=v_raw, validate_clean=v_clean,
                  reduction=reduction, scaling=scaling, tuning=pd.DataFrame(tuning), fairness=pd.DataFrame(fair),
                  dictionary=dictionary(raw), n_raw=len(raw), n_clean=len(cl), seconds=time.time() - t0)
    joblib.dump(bundle, BUNDLE)
    log(f"[6/8] saved {BUNDLE.name}")
    from .report import write_report
    write_report(bundle)
    log(f"[7/8] wrote REPORT.md   [8/8] done in {time.time() - t0:.0f}s")
    return bundle


if __name__ == "__main__":
    run()
