"""Frozen development-only selection and held-out evaluation for protocol v1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data_private"
RESULTS = ROOT / "results"
SEED = 240924
BASE = ["phos_first"]
MAIN = ["age", "sex_female", "phos_first", "phos_min", "potassium_baseline",
        "magnesium_baseline", "calcium_baseline", "creatinine_baseline",
        "glucose_baseline", "sodium_baseline"]


def load(name):
    df = pd.read_csv(PRIVATE / f"{name}_feasibility.csv", low_memory=False)
    sex = df.sex.astype(str).str.lower()
    df["sex_female"] = sex.map({"f":1.0,"female":1.0,"m":0.0,"male":0.0})
    for col in MAIN:
        if col != "sex_female":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["event"] = df.event.astype(int)
    df["followup_count"] = df.followup_count.astype(int)
    assert df.person_id.nunique() == len(df), "More than one stay per person"
    return df


def pipe(C):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("fit", LogisticRegression(C=C, max_iter=2000, solver="lbfgs", random_state=SEED)),
    ])


def choose_c(dev, cols):
    X, y = dev[cols], dev.event
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = {}
    for C in [0.01, 0.1, 1.0, 10.0]:
        vals = []
        for tr, va in cv.split(X,y):
            p = pipe(C).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:,1]
            vals.append(float(brier_score_loss(y.iloc[va], p)))
        scores[str(C)] = {"fold_brier": vals, "mean_brier": float(np.mean(vals))}
    chosen = min(scores, key=lambda c: scores[c]["mean_brier"])
    return float(chosen), scores


def calibration(y, p, weights=None):
    p = np.clip(np.asarray(p), 1e-6, 1-1e-6)
    logits = np.log(p / (1-p)).reshape(-1,1)
    fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000).fit(logits, y, sample_weight=weights)
    return float(fit.intercept_[0]), float(fit.coef_[0,0])


def metrics(y, p, weights=None):
    y, p = np.asarray(y), np.asarray(p)
    out = {"n": int(len(y)), "events": int(y.sum()),
           "event_rate": float(np.average(y, weights=weights)),
           "auroc": float(roc_auc_score(y,p,sample_weight=weights)),
           "auprc": float(average_precision_score(y,p,sample_weight=weights)),
           "brier": float(brier_score_loss(y,p,sample_weight=weights))}
    out["calibration_intercept"], out["calibration_slope"] = calibration(y,p,weights)
    return out


def alert_metrics(y,p,thresholds):
    out = {}
    for frac, threshold in thresholds.items():
        hit = np.asarray(p) >= threshold
        tp = int(np.sum(np.asarray(y)[hit]))
        n = int(hit.sum())
        out[frac] = {"development_threshold": float(threshold), "alert_fraction": float(hit.mean()),
                     "sensitivity": float(tp/np.sum(y)), "ppv": float(tp/n) if n else None,
                     "alerts_per_event": float(n/tp) if tp else None}
    return out


def bootstrap_ci(y,p,clusters=None,B=400):
    rng = np.random.default_rng(SEED)
    y,p = np.asarray(y),np.asarray(p)
    vals = {k:[] for k in ["auroc","auprc","brier","calibration_slope"]}
    if clusters is not None:
        clusters = np.asarray(clusters).astype(str)
        uniq = np.unique(clusters)
        groups = {h:np.flatnonzero(clusters==h) for h in uniq}
    for _ in range(B):
        if clusters is None:
            idx = rng.integers(len(y),size=len(y))
        else:
            draw = rng.choice(uniq,size=len(uniq),replace=True)
            idx = np.concatenate([groups[h] for h in draw])
        if len(np.unique(y[idx])) != 2:
            continue
        try:
            m=metrics(y[idx],p[idx])
            for k in vals: vals[k].append(m[k])
        except (ValueError, FloatingPointError):
            continue
    return {k:[float(x) for x in np.percentile(v,[2.5,97.5])] if v else None for k,v in vals.items()}


def main():
    mimic, eicu = load("mimic"), load("eicu")
    mimic_tested = mimic[mimic.followup_count > 0].copy()
    eicu_tested = eicu[eicu.followup_count > 0].copy()
    split = mimic_tested.person_id.astype(str).map(lambda s:int(hashlib.sha256(s.encode()).hexdigest()[:12],16)%100 < 70)
    dev, test = mimic_tested[split].copy(), mimic_tested[~split].copy()
    if dev.event.sum()<200 or eicu_tested.event.sum()<100:
        raise RuntimeError("Prespecified event-count gate failed")
    report = {"protocol": "PROTOCOL_LOCKED_20260924.md", "split": {"development_n":len(dev),
              "internal_n":len(test),"external_n":len(eicu_tested),"development_events":int(dev.event.sum()),
              "internal_events":int(test.event.sum()),"external_events":int(eicu_tested.event.sum())},
              "models":{},"missingness":{},"observation":{}}
    for data,name in [(dev,"mimic_development"),(test,"mimic_internal"),(eicu_tested,"eicu_external")]:
        report["missingness"][name] = {c:float(data[c].isna().mean()) for c in MAIN}
    predictions = {}
    for name, cols in [("baseline_phosphate", BASE),("multivariable",MAIN)]:
        C, cv = choose_c(dev,cols)
        model = pipe(C).fit(dev[cols],dev.event)
        pdev = model.predict_proba(dev[cols])[:,1]
        ptest = model.predict_proba(test[cols])[:,1]
        pext = model.predict_proba(eicu_tested[cols])[:,1]
        thresholds = {str(frac):float(np.quantile(pdev,1-frac)) for frac in [.1,.2]}
        report["models"][name] = {"columns":cols,"chosen_C":C,"development_cv":cv,
            "internal":metrics(test.event,ptest),"external":metrics(eicu_tested.event,pext),
            "internal_ci":bootstrap_ci(test.event,ptest),
            "external_ci_hospital":bootstrap_ci(eicu_tested.event,pext,eicu_tested.hospital),
            "internal_alerts":alert_metrics(test.event,ptest,thresholds),
            "external_alerts":alert_metrics(eicu_tested.event,pext,thresholds)}
        predictions[name] = (pdev,ptest,pext)
    # Store row-level predictions only in the restricted local directory.
    for label,df,idx in [("internal",test,1),("external",eicu_tested,2)]:
        out=df[["id","hospital","event","discharge_min","followup_count"]].copy()
        for name,preds in predictions.items(): out[name]=preds[idx]
        out.to_csv(PRIVATE/f"{label}_predictions.csv",index=False)
    (RESULTS/"model_performance.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"split":report["split"],"models":{n:{"internal":v["internal"],"external":v["external"]} for n,v in report["models"].items()}},indent=2))


if __name__ == "__main__":
    raise SystemExit("Use fit_frozen_models.py, then evaluate_frozen_models.py")
