"""Stage 1: train/select on MIMIC development only, then freeze artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from model_and_evaluate import BASE, MAIN, PRIVATE, RESULTS, ROOT, SEED, choose_c, load, pipe


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_extract_hash(name):
    path=PRIVATE/f"{name}_feasibility.csv"
    summary=json.loads((RESULTS/f"{name}_feasibility.json").read_text(encoding="utf-8"))
    actual=sha(path)
    if actual!=summary["private_extract_sha256"]:
        raise RuntimeError(f"{name} extract differs from extraction summary")
    return actual


def main():
    out=RESULTS/"frozen_model_manifest.json"
    if out.exists():
        raise RuntimeError("Model freeze exists; do not silently refit after external evaluation")
    extract_hashes={name:verified_extract_hash(name) for name in ("mimic","eicu")}
    mimic=load("mimic")
    tested=mimic[mimic.followup_count>0].copy()
    split=tested.person_id.astype(str).map(lambda s:int(hashlib.sha256(s.encode()).hexdigest()[:12],16)%100<70)
    dev=tested[split].copy()
    if dev.event.sum()<200: raise RuntimeError("Prespecified development event gate failed")
    specs={"baseline_phosphate":BASE,"multivariable":MAIN,
           "multivariable_plus_test_count":MAIN+["phos_baseline_count"]}
    manifest={"stage":"FROZEN_BEFORE_EXTERNAL_EVALUATION","seed":SEED,
              "development_n":len(dev),"development_events":int(dev.event.sum()),
              "development_person_id_set_sha256":hashlib.sha256("\n".join(sorted(dev.person_id.astype(str))).encode()).hexdigest(),
              "extract_sha256":extract_hashes,
              "code_hashes":{name:sha(ROOT/"code"/name) for name in
                 ["extract_feasibility.py","augment_censoring.py","model_and_evaluate.py","fit_frozen_models.py","evaluate_frozen_models.py"]},
              "protocol_sha256":sha(ROOT/"review_packages"/"PROTOCOL_LOCKED_20260924.md"),
              "models":{}}
    for name,cols in specs.items():
        C,cv=choose_c(dev,cols)
        model=pipe(C).fit(dev[cols],dev.event)
        pdev=model.predict_proba(dev[cols])[:,1]
        path=PRIVATE/f"frozen_{name}.joblib"
        joblib.dump(model,path)
        manifest["models"][name]={"columns":cols,"class":"penalized_logistic",
            "chosen_C":C,"cv":cv,"model_file":path.name,"model_sha256":sha(path),
            "thresholds":{str(frac):float(np.quantile(pdev,1-frac)) for frac in [.1,.2]}}
    # One prespecified fixed-complexity comparator; it is not selected for the main claim.
    name="exploratory_gradient_boosting"
    cols=MAIN
    model=Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True,keep_empty_features=True)),
        ("fit",HistGradientBoostingClassifier(max_leaf_nodes=15,max_iter=150,learning_rate=.05,
                                                l2_regularization=1.0,random_state=SEED))])
    model.fit(dev[cols],dev.event)
    pdev=model.predict_proba(dev[cols])[:,1]
    path=PRIVATE/f"frozen_{name}.joblib"
    joblib.dump(model,path)
    manifest["models"][name]={"columns":cols,"class":"exploratory_gradient_boosting",
        "fixed_parameters":{"max_leaf_nodes":15,"max_iter":150,"learning_rate":.05,"l2_regularization":1.0},
        "model_file":path.name,"model_sha256":sha(path),
        "thresholds":{str(frac):float(np.quantile(pdev,1-frac)) for frac in [.1,.2]}}
    out.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"development_n":manifest["development_n"],"development_events":manifest["development_events"],
                      "frozen_models":list(manifest["models"])},indent=2))


if __name__=="__main__": main()
