"""Stage 2: use verified frozen development artifacts for internal/external tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_and_evaluate import PRIVATE, RESULTS, ROOT, alert_metrics, bootstrap_ci, load, metrics


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_stats(df,p):
    result={}
    for column,items in [("sex",["Male","Female","M","F"]), ("age_group",["<65","65–89","90+"])]:
        data=df.copy()
        if column=="age_group":
            data[column]=np.where(data.age<65,"<65",np.where(data.age<90,"65–89","90+"))
        for key,chunk in data.groupby(column):
            if len(chunk)<50 or chunk.event.sum()<10 or (len(chunk)-chunk.event.sum())<10: continue
            result[f"{column}:{key}"]=metrics(chunk.event,p[data.index.get_indexer(chunk.index)])
    return result


def main():
    manifest_path=RESULTS/"frozen_model_manifest.json"
    if not manifest_path.exists(): raise RuntimeError("No frozen models: run fit stage first")
    if (RESULTS/"model_performance.json").exists():
        raise RuntimeError("External evaluation already recorded; preserve the one-shot result")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    for name,h in manifest["code_hashes"].items():
        if sha(ROOT/"code"/name)!=h: raise RuntimeError(f"Code changed since freeze: {name}")
    for name,h in manifest["extract_sha256"].items():
        actual=sha(PRIVATE/f"{name}_feasibility.csv")
        summary=json.loads((RESULTS/f"{name}_feasibility.json").read_text(encoding="utf-8"))
        if actual!=h or actual!=summary["private_extract_sha256"]:
            raise RuntimeError(f"{name} extract changed since freeze")
    mimic,eicu=load("mimic"),load("eicu")
    m=mimic[mimic.followup_count>0].copy()
    e=eicu[eicu.followup_count>0].copy()
    split=m.person_id.astype(str).map(lambda s:int(hashlib.sha256(s.encode()).hexdigest()[:12],16)%100<70)
    dev,test=m[split].copy(),m[~split].copy()
    devhash=hashlib.sha256("\n".join(sorted(dev.person_id.astype(str))).encode()).hexdigest()
    if devhash!=manifest["development_person_id_set_sha256"]:
        raise RuntimeError("MIMIC development cohort changed since model freeze")
    if e.event.sum()<100: raise RuntimeError("Prespecified external event gate failed")
    report={"protocol":"PROTOCOL_LOCKED_20260924.md","model_freeze_sha256":sha(manifest_path),
            "split":{"development_n":len(dev),"development_events":int(dev.event.sum()),
                     "internal_n":len(test),"internal_events":int(test.event.sum()),
                     "external_n":len(e),"external_events":int(e.event.sum()),
                     "external_tested_hospitals":int(e.hospital.nunique())},
            "models":{},"missingness":{}}
    for df,label in [(dev,"development"),(test,"internal"),(e,"external")]:
        report["missingness"][label]={c:float(df[c].isna().mean()) for c in manifest["models"]["multivariable"]["columns"]}
    internal=test[["id","hospital","event","discharge_min","followup_count"]].copy()
    external=e[["id","hospital","event","discharge_min","followup_count"]].copy()
    for name,spec in manifest["models"].items():
        path=PRIVATE/spec["model_file"]
        if sha(path)!=spec["model_sha256"]: raise RuntimeError(f"Frozen model changed: {name}")
        model=joblib.load(path)
        cols=spec["columns"]
        pi=model.predict_proba(test[cols])[:,1]
        pe=model.predict_proba(e[cols])[:,1]
        internal[name]=pi; external[name]=pe
        report["models"][name]={"class":spec["class"],"columns":cols,
          "internal":metrics(test.event,pi),"external":metrics(e.event,pe),
          "internal_ci":bootstrap_ci(test.event,pi),
          "external_ci_hospital":bootstrap_ci(e.event,pe,e.hospital),
          "internal_alerts":alert_metrics(test.event,pi,spec["thresholds"]),
          "external_alerts":alert_metrics(e.event,pe,spec["thresholds"]),
          "internal_subgroups":group_stats(test,pi),"external_subgroups":group_stats(e,pe)}
    internal.to_csv(PRIVATE/"internal_predictions.csv",index=False)
    external.to_csv(PRIVATE/"external_predictions.csv",index=False)
    (RESULTS/"model_performance.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"split":report["split"],"models":{n:{"internal":v["internal"],"external":v["external"]} for n,v in report["models"].items()}},indent=2))


if __name__=="__main__": main()
