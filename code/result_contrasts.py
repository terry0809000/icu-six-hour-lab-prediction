"""Post-freeze paired contrasts and anonymous hospital heterogeneity summary."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from model_and_evaluate import PRIVATE, RESULTS, SEED, load


def contrasts(frame, cluster=False, B=600):
    y=frame.event.to_numpy().astype(int)
    p0=frame.baseline_phosphate.to_numpy()
    p1=frame.multivariable.to_numpy()
    pg=frame.exploratory_gradient_boosting.to_numpy()
    groups=None
    if cluster:
        h=frame.hospital.astype(str).to_numpy()
        uniq=np.unique(h)
        groups={key:np.flatnonzero(h==key) for key in uniq}
    rng=np.random.default_rng(SEED+int(cluster))
    def calc(idx):
        yy=y[idx]
        return {
            "auroc_main_minus_baseline":float(roc_auc_score(yy,p1[idx])-roc_auc_score(yy,p0[idx])),
            "auprc_main_minus_baseline":float(average_precision_score(yy,p1[idx])-average_precision_score(yy,p0[idx])),
            "brier_main_minus_baseline":float(brier_score_loss(yy,p1[idx])-brier_score_loss(yy,p0[idx])),
            "auroc_boost_minus_main":float(roc_auc_score(yy,pg[idx])-roc_auc_score(yy,p1[idx])),
            "brier_boost_minus_main":float(brier_score_loss(yy,pg[idx])-brier_score_loss(yy,p1[idx]))}
    point=calc(np.arange(len(y)))
    samples={k:[] for k in point}
    for _ in range(B):
        if groups is None:
            idx=rng.integers(len(y),size=len(y))
        else:
            idx=np.concatenate([groups[key] for key in rng.choice(uniq,size=len(uniq),replace=True)])
        if np.unique(y[idx]).size<2: continue
        v=calc(idx)
        for k in samples: samples[k].append(v[k])
    return {k:{"estimate":v,"ci95":[float(x) for x in np.quantile(samples[k],[.025,.975])],
               "bootstrap_valid_replicates":len(samples[k])} for k,v in point.items()}


def hospitals(external):
    source=load("eicu")
    eligible=source.groupby("hospital").size().rename("eligible")
    tested=source[source.followup_count>0].groupby("hospital").size().rename("tested")
    site=pd.concat([eligible,tested],axis=1).fillna(0)
    site["testing_rate"]=site.tested/site.eligible
    observed=external.groupby("hospital").agg(events=("event","sum"),
        observed_risk=("event","mean"),predicted_risk=("multivariable","mean"))
    site=site.join(observed)
    eligible_sites=site[(site.eligible>=50)&(site.tested>=50)&(site.events>=10)]
    rho,p=spearmanr(eligible_sites.testing_rate,eligible_sites.observed_risk)
    per=[]
    for _,row in eligible_sites.sort_values("testing_rate").iterrows():
        per.append({"n_eligible":int(row.eligible),"n_tested":int(row.tested),
           "events_tested":int(row.events),"testing_rate":float(row.testing_rate),
           "observed_risk_tested":float(row.observed_risk),
           "mean_predicted_risk_tested":float(row.predicted_risk),
           "observed_minus_predicted":float(row.observed_risk-row.predicted_risk)})
    return {"sites_with_eligible_ge50_tested_ge50_events_ge10":len(per),
            "spearman_testing_vs_observed_event_risk":float(rho),
            "spearman_p_descriptive":float(p),
            "median_abs_site_calibration_difference":float(np.median([abs(x["observed_minus_predicted"]) for x in per])),
            "anonymous_sites_sorted_by_testing_rate":per}


def main():
    internal=pd.read_csv(PRIVATE/"internal_predictions.csv")
    external=pd.read_csv(PRIVATE/"external_predictions.csv")
    result={"internal_paired_patient_bootstrap":contrasts(internal),
            "external_paired_hospital_bootstrap":contrasts(external,cluster=True),
            "external_hospital_summary":hospitals(external)}
    path=RESULTS/"result_contrasts.json"
    path.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="external_hospital_summary"},indent=2))


if __name__=="__main__": main()
