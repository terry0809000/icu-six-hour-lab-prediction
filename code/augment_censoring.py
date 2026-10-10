"""Append early ICU death ascertainment before the model/data freeze.

This uses only small stay/discharge tables. It does not alter eligibility,
predictors, observed event labels, or the modelling sample.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

from extract_feasibility import EICU, MIMIC, ROOT


def rows(z, suffix):
    name=next(n for n in z.namelist() if n.endswith(suffix))
    return csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=z.open(name)),encoding="utf-8",newline=""))


def mimic_deaths(ids):
    with zipfile.ZipFile(MIMIC) as z:
        stays={r["stay_id"]:(r["hadm_id"],datetime.fromisoformat(r["intime"]),datetime.fromisoformat(r["outtime"]))
               for r in rows(z,"/icu/icustays.csv.gz") if r["stay_id"] in ids}
        hadms={v[0] for v in stays.values()}
        dates={r["hadm_id"]:datetime.fromisoformat(r["deathtime"])
               for r in rows(z,"/hosp/admissions.csv.gz") if r["hadm_id"] in hadms and r["deathtime"]}
    out={}
    for sid,(hadm,intime,outtime) in stays.items():
        death=dates.get(hadm)
        out[sid]=int(death is not None and intime <= death <= outtime and (death-intime).total_seconds()<=48*3600)
    if len(out)!=len(ids): raise RuntimeError("MIMIC stay linkage incomplete")
    return out


def eicu_deaths(ids):
    with zipfile.ZipFile(EICU) as z:
        out={}
        for r in rows(z,"/patient.csv.gz"):
            sid=r["patientunitstayid"]
            if sid not in ids: continue
            off=float(r["unitdischargeoffset"])
            out[sid]=int(r["unitdischargestatus"].strip().lower()=="expired" and 0<=off<=2880)
    if len(out)!=len(ids): raise RuntimeError("eICU stay linkage incomplete")
    return out


def main():
    for name,fn in (("mimic",mimic_deaths),("eicu",eicu_deaths)):
        path=ROOT/"data_private"/f"{name}_feasibility.csv"
        summary_path=ROOT/"results"/f"{name}_feasibility.json"
        summary=json.loads(summary_path.read_text(encoding="utf-8"))
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        if before!=summary["private_extract_sha256"]: raise RuntimeError(f"{name} extract already changed")
        with path.open(newline="",encoding="utf-8") as f:
            reader=csv.DictReader(f)
            columns=reader.fieldnames
            if "death_within48" in columns: raise RuntimeError(f"{name} already augmented")
            data=list(reader)
        ids={r["id"] for r in data}
        deaths=fn(ids)
        for r in data: r["death_within48"]=deaths[r["id"]]
        with path.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=columns+["death_within48"])
            writer.writeheader(); writer.writerows(data)
        summary["private_extract_sha256_pre_censoring_augmentation"]=before
        summary["private_extract_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        summary["early_icu_deaths_baseline_eligible"]=sum(deaths.values())
        summary["early_icu_deaths_tested"]=sum(deaths[r["id"]] for r in data if int(r["followup_count"])>0)
        summary["censoring_augmentation_code_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(name,summary["early_icu_deaths_baseline_eligible"],summary["early_icu_deaths_tested"])


if __name__=="__main__": main()
