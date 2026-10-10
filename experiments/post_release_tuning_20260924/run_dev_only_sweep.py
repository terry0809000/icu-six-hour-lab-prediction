"""Post-release exploratory tuning on MIMIC development only.

Never reads eICU or writes patient-level data. Existing frozen artifacts and
the submission archive remain unchanged. The results cannot be treated as a
new independent external validation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from model_and_evaluate import MAIN, SEED, load, metrics  # noqa: E402

HERE = Path(__file__).resolve().parent
PRIVATE = ROOT / "data_private"
RESULTS = ROOT / "results"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def model(spec: dict):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("fit", HistGradientBoostingClassifier(
            max_leaf_nodes=spec["leaves"], max_iter=spec["trees"],
            learning_rate=spec["rate"], l2_regularization=spec["l2"],
            min_samples_leaf=spec["min_leaf"], early_stopping="auto",
            random_state=SEED)),
    ])

# Defined before this run; includes the originally fixed booster as reference.
GRID = [
    {"name":"fixed_reference", "leaves":15, "trees":150, "rate":0.05, "l2":1.0, "min_leaf":20},
    {"name":"shallow", "leaves":7, "trees":200, "rate":0.05, "l2":1.0, "min_leaf":20},
    {"name":"deeper", "leaves":31, "trees":150, "rate":0.05, "l2":1.0, "min_leaf":20},
    {"name":"more_regularized", "leaves":15, "trees":150, "rate":0.05, "l2":10.0, "min_leaf":50},
    {"name":"slower_learning", "leaves":15, "trees":300, "rate":0.03, "l2":1.0, "min_leaf":20},
    {"name":"faster_learning", "leaves":15, "trees":100, "rate":0.10, "l2":1.0, "min_leaf":20},
]

def main():
    assert sha(PRIVATE / "mimic_feasibility.csv") == json.loads((RESULTS / "mimic_feasibility.json").read_text())["private_extract_sha256"]
    manifest = json.loads((RESULTS / "frozen_model_manifest.json").read_text())
    df = load("mimic")
    tested = df[df.followup_count > 0].copy()
    split = tested.person_id.astype(str).map(lambda s: int(hashlib.sha256(s.encode()).hexdigest()[:12],16)%100 < 70)
    dev, internal = tested[split].copy(), tested[~split].copy()
    dev_hash = hashlib.sha256("\n".join(sorted(dev.person_id.astype(str))).encode()).hexdigest()
    assert dev_hash == manifest["development_person_id_set_sha256"]
    X, y = dev[MAIN], dev.event.to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(cv.split(X, y))
    report = {
        "status":"POST_RELEASE_EXPLORATORY_DEV_ONLY",
        "source":"MIMIC-IV 3.1 development partition only for tuning; no eICU access",
        "source_csv_sha256":sha(PRIVATE / "mimic_feasibility.csv"),
        "frozen_manifest_sha256":sha(RESULTS / "frozen_model_manifest.json"),
        "script_sha256":sha(Path(__file__)),
        "development_n":len(dev), "development_events":int(y.sum()),
        "internal_n":len(internal), "internal_events":int(internal.event.sum()),
        "selection_rule":"lowest mean five-fold development Brier among six fixed candidates",
        "grid":GRID, "candidates":[],
    }
    for spec in GRID:
        fold_metrics=[]
        for k,(tr,va) in enumerate(folds,1):
            fit=model(spec).fit(X.iloc[tr],y[tr])
            pred=fit.predict_proba(X.iloc[va])[:,1]
            fold_metrics.append({"brier":float(brier_score_loss(y[va],pred)),
                                 "auroc":float(roc_auc_score(y[va],pred)),
                                 "auprc":float(average_precision_score(y[va],pred))})
            print(f"{spec['name']} fold {k}/5 done",flush=True)
        means={metric:float(np.mean([f[metric] for f in fold_metrics])) for metric in ("brier","auroc","auprc")}
        report["candidates"].append({"name":spec["name"],"folds":fold_metrics,"mean":means})
        print(f"{spec['name']} mean={means}",flush=True)
        (HERE / "development_cv_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    winner=min(report["candidates"],key=lambda c:c["mean"]["brier"])
    spec=next(s for s in GRID if s["name"]==winner["name"])
    trained=model(spec).fit(X,y)
    internal_pred=trained.predict_proba(internal[MAIN])[:,1]
    report["selected_name"]=winner["name"]
    report["selected_internal_metrics"]=metrics(internal.event,internal_pred)
    reference=joblib.load(PRIVATE / "frozen_exploratory_gradient_boosting.joblib")
    reference_pred=reference.predict_proba(internal[MAIN])[:,1]
    report["frozen_booster_internal_metrics_recomputed"]=metrics(internal.event,reference_pred)
    report["internal_difference_selected_minus_frozen"]={k:float(report["selected_internal_metrics"][k]-report["frozen_booster_internal_metrics_recomputed"][k]) for k in ("auroc","auprc","brier")}
    rng=np.random.default_rng(SEED+1)
    labels=internal.event.to_numpy()
    paired={k:[] for k in ("auroc","auprc","brier")}
    for _ in range(600):
        ix=rng.integers(len(labels),size=len(labels))
        if len(np.unique(labels[ix]))<2: continue
        paired["auroc"].append(float(roc_auc_score(labels[ix],internal_pred[ix])-roc_auc_score(labels[ix],reference_pred[ix])))
        paired["auprc"].append(float(average_precision_score(labels[ix],internal_pred[ix])-average_precision_score(labels[ix],reference_pred[ix])))
        paired["brier"].append(float(brier_score_loss(labels[ix],internal_pred[ix])-brier_score_loss(labels[ix],reference_pred[ix])))
    report["paired_patient_bootstrap_600_selected_minus_frozen_ci"]={k:[float(v) for v in np.quantile(values,[.025,.975])] for k,values in paired.items()}
    report["disclaimer"]="Grid and analysis were designed after the external results were known. Internal holdout is reused exploratorily; eICU remains untouched by this script but is no longer a pristine future validation for any post-hoc selected model. Do not replace the original primary model or claim independent external improvement."
    (HERE / "development_cv_results.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"selected":winner["name"],"selected_internal":report["selected_internal_metrics"],
                      "diff_vs_frozen_booster":report["internal_difference_selected_minus_frozen"]},indent=2),flush=True)

if __name__ == "__main__":
    main()
