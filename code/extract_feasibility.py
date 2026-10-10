"""Stream authorized ICU source ZIPs; keep patient-level extracts private.

The protocol was locked before this script was run. No model is fitted here.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIMIC = Path(r"D:\mimic-iv-3.1.zip")
EICU = Path(r"D:\eicu-collaborative-research-database-2.0.zip")
FIELDS = ["phosphate", "potassium", "magnesium", "calcium", "creatinine", "glucose", "sodium"]
MIMIC_ITEMS = {
    "50970": "phosphate", "50971": "potassium", "50960": "magnesium",
    "50893": "calcium", "50912": "creatinine", "50931": "glucose",
    "50983": "sodium", "52546": "creatinine", "52569": "glucose",
    "52610": "potassium", "52623": "sodium",
}
EICU_NAMES = {x: x for x in FIELDS}
EXPECTED_UNIT = {"phosphate": {"mg/dl"}, "potassium": {"meq/l", "mmol/l"},
                 "magnesium": {"mg/dl"}, "calcium": {"mg/dl"},
                 "creatinine": {"mg/dl"}, "glucose": {"mg/dl"},
                 "sodium": {"meq/l", "mmol/l"}}


def csv_member(z: zipfile.ZipFile, suffix: str):
    name = next(n for n in z.namelist() if n.endswith(suffix))
    return csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=z.open(name)), encoding="utf-8", newline=""))


def numeric(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def blank_rec(meta):
    return {**meta, "base": {}, "phos_first": None, "phos_min": None,
            "phos_baseline_count": 0, "followup_count": 0, "event": 0,
            "first_repeat_min": None, "phos_invalid_baseline": False,
            "phos_invalid_followup_count": 0, "phos_collected_baseline_count": 0,
            "phos_collected_baseline_min": None}


def record_phos_collection(rec,value,minutes):
    if not 0 <= minutes <= 360:
        return
    rec["phos_collected_baseline_count"] += 1
    if not 0.2 <= value <= 20:
        rec["phos_invalid_baseline"] = True
    else:
        old=rec["phos_collected_baseline_min"]
        rec["phos_collected_baseline_min"] = value if old is None else min(old,value)


def update(rec, field, value, minutes):
    if value is None:
        return
    if field == "phosphate":
        record_phos_collection(rec,value,minutes)
    if field == "phosphate" and not 0.2 <= value <= 20:
        if 360 < minutes <= min(2880, rec["discharge_min"]):
            rec["phos_invalid_followup_count"] += 1
        return
    if 0 <= minutes <= 360:
        if field == "phosphate":
            rec["phos_baseline_count"] += 1
            if rec["phos_first"] is None or minutes < rec["phos_first"][0]:
                rec["phos_first"] = (minutes, value)
            rec["phos_min"] = value if rec["phos_min"] is None else min(value, rec["phos_min"])
        if field not in rec["base"] or minutes > rec["base"][field][0]:
            rec["base"][field] = (minutes, value)
    elif field == "phosphate" and 360 < minutes <= min(2880, rec["discharge_min"]):
        rec["followup_count"] += 1
        rec["event"] |= int(value < 2.0)
        if rec["first_repeat_min"] is None or minutes < rec["first_repeat_min"]:
            rec["first_repeat_min"] = minutes


def mimic_extract():
    with zipfile.ZipFile(MIMIC) as z:
        ages = {}
        for r in csv_member(z, "/hosp/patients.csv.gz"):
            ages[r["subject_id"]] = (r["gender"], numeric(r["anchor_age"]), numeric(r["anchor_year"]))
        stays = {}
        for r in csv_member(z, "/icu/icustays.csv.gz"):
            sid = r["subject_id"]
            if sid not in ages or ages[sid][1] is None or not r["intime"]:
                continue
            intime = datetime.fromisoformat(r["intime"])
            outtime = datetime.fromisoformat(r["outtime"]) if r["outtime"] else None
            age = ages[sid][1] + intime.year - ages[sid][2]
            if sid not in stays or intime < stays[sid]["intime"]:
                stays[sid] = {"id": r["stay_id"], "hadm_id": r["hadm_id"],
                              "intime": intime, "outtime": outtime, "age": min(age, 90),
                              "sex": ages[sid][0], "hospital": "BIDMC"}
        stays = {sid:s for sid,s in stays.items() if s["age"] >= 18 and s["outtime"] is not None
                 and s["outtime"]-s["intime"] >= timedelta(hours=6)}
        recs = {}
        by_subject = {}
        for sid, s in stays.items():
            s2 = {k: v for k, v in s.items() if k not in ("intime", "outtime")}
            s2["person_id"] = sid
            s2["discharge_min"] = (s["outtime"] - s["intime"]).total_seconds() / 60
            recs[s["id"]] = blank_rec(s2)
            by_subject[sid] = (s, recs[s["id"]])
        n_scanned = 0
        units = Counter()
        rejected_units = Counter()
        baseline_results_not_available = 0
        for r in csv_member(z, "/hosp/labevents.csv.gz"):
            n_scanned += 1
            field = MIMIC_ITEMS.get(r["itemid"])
            if field is None:
                continue
            hit = by_subject.get(r["subject_id"])
            if hit is None or hit[0]["hadm_id"] != r["hadm_id"]:
                continue
            s, rec = hit
            chart = r["charttime"]
            if chart < str(s["intime"]) or chart > str(min(s["intime"] + timedelta(hours=48), s["outtime"])):
                continue
            value = numeric(r["valuenum"])
            if value is None:
                continue
            if r["valueuom"].lower() not in EXPECTED_UNIT[field]:
                rejected_units[(field,r["valueuom"])] += 1
                continue
            units[(field, r["valueuom"])] += 1
            minutes = (datetime.fromisoformat(chart) - s["intime"]).total_seconds() / 60
            if 0 <= minutes <= 360:
                if not r["storetime"] or datetime.fromisoformat(r["storetime"]) > s["intime"] + timedelta(hours=6):
                    baseline_results_not_available += 1
                    if field == "phosphate":
                        record_phos_collection(rec,value,minutes)
                    continue
            update(rec, field, value, minutes)
        return list(recs.values()), {"rows_scanned": n_scanned, "units": dict((str(k), v) for k,v in units.items()),
                                     "rejected_units": dict((str(k),v) for k,v in rejected_units.items()),
                                     "baseline_results_not_available_by_landmark":baseline_results_not_available,
                                     "eligible_age_stays": len(recs)}


def eicu_extract():
    with zipfile.ZipFile(EICU) as z:
        stays = {}
        for r in csv_member(z, "/patient.csv.gz"):
            age = 90 if r["age"].startswith(">") else numeric(r["age"])
            discharge = numeric(r["unitdischargeoffset"])
            pid = r["uniquepid"] or r["patienthealthsystemstayid"]
            visit = numeric(r["unitvisitnumber"])
            s = {"id": r["patientunitstayid"], "age": age, "sex": r["gender"],
                 "hospital": r["hospitalid"], "discharge_min": discharge,
                 "pid": pid, "visit": visit}
            # eICU has no cross-encounter chronology; lowest stay ID is a deterministic
            # one-stay-per-person surrogate, documented in the schema addendum.
            if pid not in stays or int(s["id"]) < int(stays[pid]["id"]):
                stays[pid] = s
        stays = {pid:s for pid,s in stays.items() if s["age"] is not None and s["age"] >= 18
                 and s["discharge_min"] is not None and s["discharge_min"] >= 360
                 and s["visit"] == 1}
        recs = {s["id"]: blank_rec({**{k:v for k,v in s.items() if k not in ("pid","visit")},
                                    "person_id":s["pid"]}) for s in stays.values()}
        n_scanned = 0
        units = Counter()
        rejected_units = Counter()
        baseline_results_not_available = 0
        for r in csv_member(z, "/lab.csv.gz"):
            n_scanned += 1
            rec = recs.get(r["patientunitstayid"])
            if rec is None:
                continue
            field = EICU_NAMES.get(r["labname"].lower())
            if field is None:
                continue
            minutes = numeric(r["labresultoffset"])
            if minutes is None or not 0 <= minutes <= min(2880, rec["discharge_min"]):
                continue
            value = numeric(r["labresult"])
            if value is None:
                continue
            system_unit = r["labmeasurenamesystem"].lower()
            interface_unit = r["labmeasurenameinterface"].lower()
            if system_unit not in EXPECTED_UNIT[field] or (interface_unit and interface_unit not in EXPECTED_UNIT[field]):
                rejected_units[(field,system_unit,interface_unit)] += 1
                continue
            units[(field, r["labmeasurenamesystem"], r["labmeasurenameinterface"])] += 1
            if 0 <= minutes <= 360:
                revised = numeric(r["labresultrevisedoffset"])
                if revised is None or revised > 360:
                    baseline_results_not_available += 1
                    if field == "phosphate":
                        record_phos_collection(rec,value,minutes)
                    continue
            update(rec, field, value, minutes)
        return list(recs.values()), {"rows_scanned": n_scanned, "units": dict((str(k), v) for k,v in units.items()),
                                     "rejected_units": dict((str(k),v) for k,v in rejected_units.items()),
                                     "baseline_results_not_available_by_landmark":baseline_results_not_available,
                                     "eligible_age_stays": len(recs)}


def summarize_and_save(label, records, extraction):
    baseline = [r for r in records if r["phos_baseline_count"] and r["phos_collected_baseline_count"]
                and r["phos_collected_baseline_min"] is not None
                and r["phos_collected_baseline_min"] >= 2.0
                and not r["phos_invalid_baseline"]]
    tested = [r for r in baseline if r["followup_count"]]
    by_hospital = Counter(r["hospital"] for r in baseline)
    hospital_tested = Counter(r["hospital"] for r in tested)
    summary = {
        "source": label, "extraction": extraction,
        "with_nonlow_baseline_phosphate": len(baseline),
        "with_followup_phosphate": len(tested),
        "repeat_testing_rate": len(tested)/len(baseline) if baseline else None,
        "documented_events_among_tested": sum(r["event"] for r in tested),
        "observed_event_rate_tested": sum(r["event"] for r in tested)/len(tested) if tested else None,
        "hospitals_with_baseline": len(by_hospital),
        "hospitals_ge_50": sum(n >= 50 for n in by_hospital.values()),
        "hospitals_ge_50_and_testing_ge_70pct": sum(n >= 50 and hospital_tested[h]/n >= .7 for h,n in by_hospital.items()),
    }
    private = ROOT/"data_private"/f"{label}_feasibility.csv"
    columns = ["id","person_id","age","sex","hospital","discharge_min","phos_first","phos_min",
               "phos_baseline_count","phos_collected_baseline_count","phos_collected_baseline_min",
               "followup_count","event","first_repeat_min",
               "phos_invalid_followup_count"] + [f"{x}_baseline" for x in FIELDS]
    with private.open("w", newline="", encoding="utf-8") as f:
        out = csv.DictWriter(f, fieldnames=columns)
        out.writeheader()
        for r in baseline:
            d = {k: r.get(k) for k in columns if not k.endswith("_baseline")}
            d["phos_first"] = r["phos_first"][1]
            for x in FIELDS:
                d[f"{x}_baseline"] = r["base"].get(x, (None,None))[1]
            out.writerow(d)
    summary["private_extract_sha256"] = hashlib.sha256(private.read_bytes()).hexdigest()
    (ROOT/"results"/f"{label}_feasibility.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import sys
    if sys.argv[1:] not in (["mimic"], ["eicu"]):
        raise SystemExit("usage: extract_feasibility.py mimic|eicu")
    which = sys.argv[1]
    rows, details = mimic_extract() if which == "mimic" else eicu_extract()
    print(json.dumps(summarize_and_save(which, rows, details), indent=2))
