"""Scientific Figure 1: editable, data-true study and ascertainment schematic."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figures"
M=json.loads((ROOT/"results"/"mimic_feasibility.json").read_text())
E=json.loads((ROOT/"results"/"eicu_feasibility.json").read_text())

NAVY="#123047"; BLUE="#1A67B5"; TEAL="#008C8C"; CORAL="#D84B68"
VIOLET="#8055B7"; GOLD="#E0A12E"; GRAY="#5D6C78"; LIGHT="#F3F7FA"
mpl.rcParams.update({"font.family":"DejaVu Sans",
    "font.size":8.0,"svg.fonttype":"none","pdf.fonttype":42,
    "axes.spines.top":False,"axes.spines.right":False})


def card(ax,x,y,w,h,fc="white",edge="#DDE8EE",radius=.025):
    patch=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.012,rounding_size={radius}",
                         linewidth=.8,facecolor=fc,edgecolor=edge)
    ax.add_patch(patch)
    return patch


def label(ax,x,y,s,size=8,color=NAVY,weight="normal",ha="left",va="center"):
    ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va)


def arrow(ax,x0,y0,x1,y1,color=BLUE,lw=2):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=11,
                                 linewidth=lw,color=color,connectionstyle="arc3"))


fig=plt.figure(figsize=(7.1,5.6),facecolor="white")
ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,1),ylim=(0,1));ax.axis("off")

# Structural color is restrained; every hue has one semantic meaning.
label(ax,.04,.955,"WHO HAS A DOCUMENTED LOW PHOSPHATE?",15,NAVY,"bold")
label(ax,.04,.91,"A six-hour ICU prediction study with explicit testing and transportability checks",9,GRAY)
ax.plot([.04,.96],[.885,.885],color="#CADAE5",lw=.8)

# A: the object of prediction, with measured versus latent outcomes clearly encoded.
card(ax,.04,.52,.92,.33,LIGHT)
label(ax,.06,.82,"A  |  THE PATIENT-LEVEL QUESTION",9,NAVY,"bold")
for x,t,c in [(.14,"ICU entry",BLUE),(.45,"Six-hour landmark",TEAL),(.82,"48-hour horizon",CORAL)]:
    ax.add_patch(Circle((x,.695),.013,facecolor=c,edgecolor="white",lw=1.5))
    label(ax,x,.755,t,8,c,"bold",ha="center")
ax.plot([.14,.82],[.695,.695],color="#A4B8C6",lw=2,zorder=0)
label(ax,.14,.625,"0 h",7,GRAY,ha="center")
label(ax,.45,.625,"Baseline P ≥ 2.0 mg/dL",8,TEAL,"bold",ha="center")
label(ax,.82,.625,"P < 2.0 mg/dL?",8,CORAL,"bold",ha="center")
card(ax,.205,.535,.49,.055,fc="white",edge="#CEE2E9",radius=.015)
label(ax,.45,.562,"Model inputs available by 6 h; baseline cohort confirmed later",7,NAVY,ha="center")
arrow(ax,.47,.68,.79,.68,TEAL,1.5)

# B: source cohorts and locked validation design.
card(ax,.04,.15,.44,.33,"white")
label(ax,.06,.45,"B  |  DEVELOPMENT → EXTERNAL TEST",9,NAVY,"bold")
card(ax,.06,.31,.18,.105,fc="#E8F2FC",edge="#BEDAF2",radius=.015)
label(ax,.15,.382,"MIMIC-IV",8,BLUE,"bold",ha="center")
label(ax,.15,.35,f"{M['with_nonlow_baseline_phosphate']:,} eligible",7,NAVY,ha="center")
label(ax,.15,.325,"70% develop · 30% test",7,GRAY,ha="center")
arrow(ax,.25,.36,.29,.36,BLUE,1.7)
card(ax,.30,.31,.16,.105,fc="#F0EAF8",edge="#DCD0EB",radius=.015)
label(ax,.38,.382,"eICU",8,VIOLET,"bold",ha="center")
label(ax,.38,.35,f"{E['with_nonlow_baseline_phosphate']:,} eligible",7,NAVY,ha="center")
label(ax,.38,.325,f"{E['hospitals_with_baseline']} hospitals",7,GRAY,ha="center")
label(ax,.06,.25,"Model selection stays within MIMIC development data.",7,GRAY)
label(ax,.06,.217,"eICU receives a frozen model and fixed alert thresholds.",7,GRAY)

# C: what can and cannot be known from historical lab testing.
card(ax,.52,.15,.44,.33,"white")
label(ax,.54,.45,"C  |  THE OBSERVATION LENS",9,NAVY,"bold")
card(ax,.54,.33,.18,.08,fc="#E7F5F1",edge="#B9DCD2",radius=.015)
label(ax,.63,.383,"RETESTED",7,TEAL,"bold",ha="center")
label(ax,.63,.355,f"{E['with_followup_phosphate']:,} in eICU",7,NAVY,ha="center")
card(ax,.75,.33,.18,.08,fc="#F2F2F5",edge="#DADDE3",radius=.015)
label(ax,.84,.383,"NOT RETESTED",7,GRAY,"bold",ha="center")
label(ax,.84,.355,f"{E['with_nonlow_baseline_phosphate']-E['with_followup_phosphate']:,} unknown",7,NAVY,ha="center")
ax.plot([.735,.735],[.32,.41],color="#CDD6DC",lw=1)
label(ax,.54,.277,"Documented event",7,CORAL,"bold")
label(ax,.72,.277,"≠",9,NAVY,"bold",ha="center")
label(ax,.75,.277,"biological incidence",7,GRAY)
label(ax,.54,.225,"Report testing rates, bounds, weighting assumptions",7,NAVY)
label(ax,.54,.195,"and between-hospital calibration before interpretation.",7,NAVY)

ax.plot([.04,.96],[.115,.115],color="#CADAE5",lw=.8)
label(ax,.04,.078,"RESEARCH QUESTION",7,BLUE,"bold")
label(ax,.04,.051,"Does early low-phosphate risk transport when the outcome is only observed after repeat testing?",8,NAVY)

for ext in ["svg","pdf","png"]:
    fig.savefig(OUT/f"Figure1_study_design.{ext}",dpi=450,facecolor="white",bbox_inches="tight",pad_inches=.02)
plt.close(fig)
print(OUT/"Figure1_study_design.svg")
