"""Post-freeze sensitivity restricted to fully reported baseline phosphate results.

The primary retrospective cohort excludes low phosphate specimens collected in
the first six hours even if their final results arrived later. This restriction
checks the frozen model only in patients whose every baseline phosphate result
was available by the landmark. It cannot add excluded patients back and is not
a redefinition of the primary cohort.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT=Path(__file__).resolve().parents[1]
out={}
for source,predfile in (("mimic","internal_predictions.csv"),("eicu","external_predictions.csv")):
    eligibility=pd.read_csv(ROOT/"data_private"/f"{source}_feasibility.csv",
                            usecols=["id","phos_baseline_count","phos_collected_baseline_count"])
    eligible_n=len(eligibility)
    full=eligibility.phos_baseline_count.eq(eligibility.phos_collected_baseline_count)
    eligible_complete=int(full.sum())
    preds=pd.read_csv(ROOT/"data_private"/predfile)
    joined=preds.merge(eligibility.loc[full,["id"]],on="id",how="inner",validate="one_to_one")
    y=joined.event.to_numpy()
    assert y.min()==0 and y.max()==1
    models={}
    for name in ("baseline_phosphate","multivariable","exploratory_gradient_boosting"):
        p=joined[name].to_numpy()
        models[name]={"auroc":float(roc_auc_score(y,p)),
                      "auprc":float(average_precision_score(y,p)),
                      "brier":float(brier_score_loss(y,p))}
    out[source]={"primary_eligible_n":eligible_n,"all_baseline_phosphate_results_available_eligible_n":eligible_complete,
                 "complete_availability_fraction":eligible_complete/eligible_n,
                 "tested_evaluation_n":len(joined),"tested_events":int(y.sum()),
                 "tested_event_rate":float(y.mean()),"models":models,
                 "scope":"Post-freeze restricted subset; cannot restore retrospectively excluded late-reported low-baseline patients."}
(ROOT/"results"/"result_availability_sensitivity.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(json.dumps(out,indent=2))
