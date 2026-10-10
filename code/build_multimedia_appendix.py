"""Build a nonidentifying supplementary methods/results appendix from frozen aggregates."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
res=lambda n: json.loads((ROOT/"results"/n).read_text(encoding="utf-8"))
m=res("mimic_feasibility.json"); e=res("eicu_feasibility.json")
perf=res("model_performance.json"); obs=res("observation_analysis.json")
ctr=res("result_contrasts.json"); avail=res("result_availability_sensitivity.json")
freeze=res("frozen_model_manifest.json")

def pct(x): return f"{100*x:.1f}%"
def num(x): return f"{x:,}"
def f3(x): return f"{x:.3f}"

lines=["# Multimedia Appendix 1: Additional methods and aggregate results", "",
"This appendix accompanies the Original Paper on six-hour prediction of documented low phosphate. It contains no patient-level rows, predictions, hospital identifiers, or fitted model objects.", "",
"## Table S1. Cohort assembly and outcome ascertainment", "",
"| Stage | MIMIC-IV 3.1 | eICU-CRD 2.0 |", "|---|---:|---:|",
f"| One selected adult stay with ICU duration ≥6 h | {num(m['extraction']['eligible_age_stays'])} | {num(e['extraction']['eligible_age_stays'])} |",
f"| Baseline phosphate profile eligible | {num(m['with_nonlow_baseline_phosphate'])} | {num(e['with_nonlow_baseline_phosphate'])} |",
f"| Qualifying repeat phosphate by discharge or 48 h | {num(m['with_followup_phosphate'])} ({pct(m['repeat_testing_rate'])}) | {num(e['with_followup_phosphate'])} ({pct(e['repeat_testing_rate'])}) |",
f"| No qualifying repeat phosphate | {num(obs['mimic']['untested_n'])} | {num(obs['eicu']['untested_n'])} |",
f"| Documented phosphate <2.0 mg/dL among retested | {num(m['documented_events_among_tested'])} ({pct(m['observed_event_rate_tested'])}) | {num(e['documented_events_among_tested'])} ({pct(e['observed_event_rate_tested'])}) |",
f"| Hospitals with eligible patients | {num(m['hospitals_with_baseline'])} | {num(e['hospitals_with_baseline'])} |",
f"| Discharged before 48 h among eligible | {num(obs['mimic']['discharged_before_48h_n'])} | {num(obs['eicu']['discharged_before_48h_n'])} |",
f"| ICU death within 48 h among eligible | {num(obs['mimic']['icu_death_within48_n'])} | {num(obs['eicu']['icu_death_within48_n'])} |", "",
"The baseline eligibility counts combine all excluded patients without a qualifying available result, with an early collected low result, with an implausible early phosphate value, or with other unsuccessful harmonization. The extractor did not preserve mutually exclusive counts for each of these reasons; no more detailed exclusion flow is claimed.", "",
"## Timing and data mapping", "",
"At least one baseline phosphate result had to be available by six hours; all phosphate specimens drawn during the first six hours had to be ≥2.0 mg/dL. A specimen with a result reported after six hours could therefore exclude a patient retrospectively. MIMIC uses `charttime` for specimen time and `storetime` for predictor availability; eICU uses `labResultOffset` for specimen time and `labResultRevisedOffset` as a conservative availability time. Follow-up specimens were eligible only for (6,48] hours and before ICU discharge. Lack of a mapped eICU laboratory row does not prove that no blood was drawn. Source mapping and exact unit allowlists are preserved in the accompanying code and source audit.", "",
"## Model specification", "",
"One row was retained per source person. The MIMIC retested subset was split by a deterministic SHA-256 patient-identifier rule into development and internal testing. Only MIMIC development data were used for model choice, imputation, scaling, and threshold selection. Five-fold stratified cross-validation chose logistic C from {0.01, 0.1, 1, 10} by mean validation Brier score. Median imputation retained empty features and added missingness indicators; logistic inputs were standardized after imputation. Logistic fitting used `lbfgs`, maximum 2,000 iterations, and random seed 240924. All three logistic models selected C=10. The exploratory histogram gradient booster used 15 maximum leaf nodes, 150 iterations, learning rate 0.05, L2 regularization 1.0, and the same seed without outcome-driven tuning. The external data were evaluated once using the frozen model manifest.", "",
"The main model included age, encoded female sex, first and minimum available baseline phosphate, and latest available potassium, magnesium, calcium, creatinine, glucose, and sodium. Its ablation added prelandmark phosphate count; the comparator used first available phosphate alone. The common laboratory units were mg/dL for phosphate, calcium, magnesium, creatinine, and glucose, and mEq/L or mmol/L for potassium and sodium. Values outside 0.2–20 mg/dL for phosphate were considered implausible. The same extraction rules, rather than site-specific conversion inferred from outcomes, were applied in both sources.", "",
"## Table S2. Available-predictor missingness among retested evaluation patients", "",
"| Variable | MIMIC internal missing | eICU external missing |", "|---|---:|---:|"]
columns=freeze['models']['multivariable']['columns']
frames={name:pd.read_csv(ROOT/"data_private"/fname) for name,fname in
        (("MIMIC","internal_predictions.csv"),("eICU","external_predictions.csv"))}
for col in columns:
    if col=='sex_female':
        raw='sex'
        rates=[]
        for name,src in (("MIMIC","mimic"),("eICU","eicu")):
            d=pd.read_csv(ROOT/"data_private"/f"{src}_feasibility.csv",usecols=['id',raw])
            j=frames[name][['id']].merge(d,on='id',validate='one_to_one')
            rates.append(float(j[raw].isna().mean()))
    else:
        rates=[]
        for name,src in (("MIMIC","mimic"),("eICU","eicu")):
            d=pd.read_csv(ROOT/"data_private"/f"{src}_feasibility.csv",usecols=['id',col])
            j=frames[name][['id']].merge(d,on='id',validate='one_to_one')
            rates.append(float(j[col].isna().mean()))
    lines.append(f"| {col.replace('_',' ')} | {pct(rates[0])} | {pct(rates[1])} |")
lines += ["", "Missingness is computed over the held-out retested analysis sets and is descriptive; imputation parameters came from development data only.", "",
"## Table S3. Performance of all frozen models", "",
"| Model | Cohort | Patients | Events | AUROC | AUPRC | Brier | Calibration intercept | Calibration slope |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for key,label in (("baseline_phosphate","Phosphate only"),("multivariable","Principal multivariable"),
                  ("multivariable_plus_test_count","Principal plus phosphate count"),
                  ("exploratory_gradient_boosting","Exploratory gradient boosting")):
    for cohort,clabel in (("internal","MIMIC internal"),("external","eICU external")):
        d=perf['models'][key][cohort]
        lines.append(f"| {label} | {clabel} | {num(d['n'])} | {num(d['events'])} | {f3(d['auroc'])} | {f3(d['auprc'])} | {f3(d['brier'])} | {f3(d['calibration_intercept'])} | {f3(d['calibration_slope'])} |")
lines += ["", "Point estimates above apply only to retested patients. External model intervals are hospital-cluster bootstrap intervals; paired differences use the same sampled hospitals for both models.", "",
"## Table S4. Paired performance contrasts", "",
"| Contrast | MIMIC internal estimate (95% CI) | eICU external estimate (95% CI) |", "|---|---:|---:|"]
for k,label in (("auroc_main_minus_baseline","Main minus phosphate-only AUROC"),
                ("auprc_main_minus_baseline","Main minus phosphate-only AUPRC"),
                ("brier_main_minus_baseline","Main minus phosphate-only Brier"),
                ("auroc_boost_minus_main","Booster minus main AUROC"),
                ("brier_boost_minus_main","Booster minus main Brier")):
    vals=[]
    for source in ('internal_paired_patient_bootstrap','external_paired_hospital_bootstrap'):
        d=ctr[source][k]; vals.append(f"{d['estimate']:.4f} ({d['ci95'][0]:.4f} to {d['ci95'][1]:.4f})")
    lines.append(f"| {label} | {vals[0]} | {vals[1]} |")
lines += ["", "All paired contrasts use 600 bootstrap replicates. Negative Brier contrasts favor the first-named model.", "",
"## Table S5. Observation-process and restricted-set analyses", "",
"| Analysis | MIMIC source or internal test | eICU source or external test |", "|---|---:|---:|",
f"| Median first-repeat time since ICU admission, h | {obs['mimic']['first_repeat_hour_median']:.2f} | {obs['eicu']['first_repeat_hour_median']:.2f} |",
f"| Repeat-testing propensity AUROC | {f3(obs['mimic']['repeat_testing_propensity_auroc'])} | {f3(obs['eicu']['repeat_testing_propensity_auroc'])} |",
f"| 99th percentile weight cap | {obs['mimic']['weight_cap_p99']:.2f} | {obs['eicu']['weight_cap_p99']:.2f} |",
f"| Effective weighted n among all tested | {obs['mimic']['weight_effective_n']:.0f} | {obs['eicu']['weight_effective_n']:.0f} |",
f"| Conditional weighted main AUROC | {f3(obs['mimic']['models']['multivariable']['tested_ipw_conditional_MAR']['auroc'])} | {f3(obs['eicu']['models']['multivariable']['tested_ipw_conditional_MAR']['auroc'])} |",
f"| Complete-48-h-stay tested main AUROC | {f3(obs['mimic']['models']['multivariable']['complete_48h_stays_tested']['auroc'])} | {f3(obs['eicu']['models']['multivariable']['complete_48h_stays_tested']['auroc'])} |",
f"| All baseline phosphate results available, eligible n | {num(avail['mimic']['all_baseline_phosphate_results_available_eligible_n'])} | {num(avail['eicu']['all_baseline_phosphate_results_available_eligible_n'])} |",
f"| All baseline results available, tested evaluation n | {num(avail['mimic']['tested_evaluation_n'])} | {num(avail['eicu']['tested_evaluation_n'])} |",
f"| All baseline results available, main AUROC | {f3(avail['mimic']['models']['multivariable']['auroc'])} | {f3(avail['eicu']['models']['multivariable']['auroc'])} |", "",
"The inverse-probability analysis is conditional on measured baseline data explaining repeat testing and on positivity. A low propensity-model AUROC does not establish exchangeability. The all-results-available restriction was computed after model freeze and does not restore anyone excluded because of a late-reported low result.", "",
"## Unknown-label assignments", "",
"Among 4,649 MIMIC and 5,435 eICU patients without a qualifying repeat result, assigning every unknown label 0 versus 1 yields, respectively, full-cohort binary-label ranges of 9.4%-24.4% and 10.6%-45.5%. Even these are not physiological incidence bounds because recorded negatives can miss low values between samples. No imputation-based biological incidence estimate is supplied.", "",
"## Provenance and restrictions", "",
f"The locked protocol SHA-256 is `{freeze['protocol_sha256']}`. The freeze manifest records SHA-256 hashes of each development/external extract, model file, and frozen code file. All individual-level source rows, predictions, and fitted objects remain in the credentialed local workspace. Public code must be run only after a user independently obtains access to MIMIC-IV and eICU-CRD under their data-use agreements.", "",
"### Source documentation", "",
"1. MIMIC-IV version 3.1. https://physionet.org/content/mimiciv/3.1/", "",
"2. eICU-CRD version 2.0. https://physionet.org/content/eicu-crd/2.0/", "",
"3. eICU `lab` schema and offset definitions. https://eicu.mit.edu/eicutables/lab/", ""]

dest=ROOT/"deliverables"/"work_in_progress"/"Multimedia_Appendix_1.md"
dest.write_text("\n".join(lines),encoding="utf-8")
print(dest)
