"""Development-only check beyond the original logistic C-grid upper boundary."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"code"))
from model_and_evaluate import MAIN, SEED, load, pipe  # noqa: E402

df=load("mimic")
tested=df[df.followup_count>0].copy()
split=tested.person_id.astype(str).map(lambda s:int(hashlib.sha256(s.encode()).hexdigest()[:12],16)%100<70)
dev=tested[split]
dev_hash=hashlib.sha256("\n".join(sorted(dev.person_id.astype(str))).encode()).hexdigest()
manifest=json.loads((ROOT/"results"/"frozen_model_manifest.json").read_text())
assert dev_hash==manifest["development_person_id_set_sha256"]
X,y=dev[MAIN],dev.event.to_numpy()
folds=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED).split(X,y))
out={"status":"POST_RELEASE_EXPLORATORY_DEV_ONLY", "development_n":len(dev),"development_events":int(y.sum()),"C_grid":[10,100,1000],"results":{}}
for C in out["C_grid"]:
    scores=[]
    for tr,va in folds:
        p=pipe(C).fit(X.iloc[tr],y[tr]).predict_proba(X.iloc[va])[:,1]
        scores.append({"brier":float(brier_score_loss(y[va],p)),"auroc":float(roc_auc_score(y[va],p)),"auprc":float(average_precision_score(y[va],p))})
    out["results"][str(C)]={"folds":scores,"mean":{k:float(np.mean([s[k] for s in scores])) for k in ("brier","auroc","auprc")}}
(Path(__file__).resolve().parent/"logistic_c_extension.json").write_text(json.dumps(out,indent=2))
print(json.dumps({k:v["mean"] for k,v in out["results"].items()},indent=2))
