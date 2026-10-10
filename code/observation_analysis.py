"""Prespecified observation-process and censoring sensitivity analyses."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from model_and_evaluate import MAIN, ROOT, PRIVATE, RESULTS, SEED, load, metrics, pipe


def propensity_oof(df):
    y = (df.followup_count > 0).astype(int).to_numpy()
    X = df[MAIN]
    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED)
    p = np.zeros(len(df))
    for tr,va in cv.split(X,y):
        p[va] = pipe(1.0).fit(X.iloc[tr],y[tr]).predict_proba(X.iloc[va])[:,1]
    return np.clip(p,0.01,0.99),float(roc_auc_score(y,p))


def summarize(name):
    df=load(name)
    tested = df.followup_count > 0
    nt, nu = int(tested.sum()), int((~tested).sum())
    events = int(df.loc[tested,"event"].sum())
    p, propensity_auroc = propensity_oof(df)
    w = 1/p[tested.to_numpy()]
    cap = float(np.quantile(w,.99))
    wt = np.minimum(w,cap)
    ess = float(wt.sum()**2/(wt**2).sum())
    select_cols=["age","phos_first","phos_min","creatinine_baseline","potassium_baseline"]
    case_mix={c:{"tested_median":float(df.loc[tested,c].median()),
                 "untested_median":float(df.loc[~tested,c].median()),
                 "tested_missing":float(df.loc[tested,c].isna().mean()),
                 "untested_missing":float(df.loc[~tested,c].isna().mean())} for c in select_cols}
    hospitals=df.groupby("hospital").agg(eligible=("id","size"),tested=("followup_count",lambda x:int((x>0).sum())))
    hospitals["rate"]=hospitals.tested/hospitals.eligible
    high=hospitals[(hospitals.eligible>=50)&(hospitals.rate>=.7)].index
    well_sized=hospitals[hospitals.eligible>=50]
    repeat_by_hospital=df.loc[tested].groupby("hospital").first_repeat_min.median()/60
    out={"eligible_n":len(df),"tested_n":nt,"untested_n":nu,
         "testing_rate":float(nt/len(df)),"events_tested":events,
         "documented_event_rate_among_tested":float(events/nt),
         "full_cohort_event_risk_lower_bound":float(events/len(df)),
         "full_cohort_event_risk_upper_bound":float((events+nu)/len(df)),
         "first_repeat_hour_median":float(df.loc[tested,"first_repeat_min"].median()/60),
         "case_mix":case_mix,"repeat_testing_propensity_auroc":propensity_auroc,
         "weight_cap_p99":cap,"weight_min":float(wt.min()),"weight_median":float(np.median(wt)),
         "weight_max":float(wt.max()),"weight_effective_n":ess,
         "hospitals_total":len(hospitals),"hospitals_eligible_ge_50":int((hospitals.eligible>=50).sum()),
         "hospital_testing_rate_ge50_median":float(well_sized.rate.median()) if len(well_sized) else None,
         "hospital_testing_rate_ge50_iqr":[float(x) for x in well_sized.rate.quantile([.25,.75])],
         "hospital_first_repeat_hour_ge50_median":float(repeat_by_hospital.reindex(well_sized.index).median()),
         "high_testing_hospital_count":len(high),
         "high_testing_patients":int(df.hospital.isin(high).sum()),
         "invalid_followup_phosphate_records":int(df.phos_invalid_followup_count.sum()),
         "complete_48h_tested_n":int((tested & (df.discharge_min>=2880)).sum()),
         "discharged_before_48h_n":int((df.discharge_min<2880).sum()),
         "discharged_before_48h_tested_n":int((tested & (df.discharge_min<2880)).sum()),
         "icu_death_within48_n":int(df.death_within48.sum()),
         "icu_death_within48_tested_n":int(df.loc[tested,"death_within48"].sum()),
         "death_ascertainment":"MIMIC admissions.deathtime within selected ICU stay; eICU unitdischargestatus Expired within 48 hours"}
    if name in ("mimic","eicu"):
        pred=pd.read_csv(PRIVATE/("external_predictions.csv" if name=="eicu" else "internal_predictions.csv"))
        joined=df.loc[tested,["id","hospital","event","discharge_min"]].merge(
            pred[["id","baseline_phosphate","multivariable"]],on="id",validate="one_to_one")
        if name=="eicu": assert len(joined)==nt
        weight_by_id=dict(zip(df.loc[tested,"id"].astype(str),wt))
        joined["weight"]=joined.id.astype(str).map(weight_by_id)
        out["models"]={}
        for model in ["baseline_phosphate","multivariable"]:
            m={"tested_unweighted":metrics(joined.event,joined[model]),
               "tested_ipw_conditional_MAR":metrics(joined.event,joined[model],joined.weight)}
            if name=="eicu":
                h=joined[joined.hospital.isin(high)]
                if h.event.nunique()==2 and len(h)>50:
                    m["high_testing_hospitals_tested"]=metrics(h.event,h[model])
            c=joined[joined.discharge_min>=2880]
            if c.event.nunique()==2 and len(c)>50:
                m["complete_48h_stays_tested"]=metrics(c.event,c[model])
            out["models"][model]=m
    return out


if __name__ == "__main__":
    result={name:summarize(name) for name in ["mimic","eicu"]}
    (RESULTS/"observation_analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:{x:v for x,v in d.items() if x not in ("models","case_mix")} for k,d in result.items()},indent=2))
