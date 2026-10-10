"""Publication figures from frozen predictions and aggregate result files.

Patient-level predictions stay in data_private; exported graphics show only
aggregate estimates. Run after the one-shot external evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figures"
OUT.mkdir(exist_ok=True)
R=json.loads((ROOT/"results"/"model_performance.json").read_text(encoding="utf-8"))
O=json.loads((ROOT/"results"/"observation_analysis.json").read_text(encoding="utf-8"))
C=json.loads((ROOT/"results"/"result_contrasts.json").read_text(encoding="utf-8"))
P={"internal":pd.read_csv(ROOT/"data_private"/"internal_predictions.csv"),
   "external":pd.read_csv(ROOT/"data_private"/"external_predictions.csv")}

NAVY="#123047"; BLUE="#1A67B5"; TEAL="#008C8C"; VIOLET="#8055B7"
CORAL="#D84B68"; GRAY="#647480"; GRID="#E6EDF1"; LIGHT="#F3F7FA"
mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"svg.fonttype":"none",
    "pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False,
    "axes.labelcolor":NAVY,"text.color":NAVY,"xtick.color":GRAY,"ytick.color":GRAY,
    "axes.edgecolor":"#AAB9C3","figure.facecolor":"white","savefig.facecolor":"white"})


def save(fig,name):
    for ext in ("pdf","svg","png"):
        fig.savefig(OUT/f"{name}.{ext}",dpi=450,bbox_inches="tight",pad_inches=.045)
    plt.close(fig)


def panel(ax,letter,title):
    ax.set_title(f"{letter}  {title}",loc="left",fontsize=9,fontweight="bold",pad=10)
    ax.grid(axis="x",color=GRID,lw=.55,zorder=0)


def performance():
    models=[("Baseline phosphate","baseline_phosphate",BLUE),
            ("Multivariable","multivariable",TEAL),
            ("Gradient boosting","exploratory_gradient_boosting",VIOLET)]
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.25),gridspec_kw={"wspace":.32})
    for ax,metric,letter,title,xlim in [
        (axs[0],"auroc","A","Discrimination",(.54,.81)),
        (axs[1],"auprc","B","Precision and recall",(.08,.43))]:
        panel(ax,letter,title)
        for i,(label,key,color) in enumerate(models):
            for cohort,offset,marker in [("internal",.10,"o"),("external",-.10,"s")]:
                m=R["models"][key][cohort][metric]
                ci=R["models"][key]["internal_ci" if cohort=="internal" else "external_ci_hospital"][metric]
                y=2-i+offset
                ax.plot(ci,[y,y],color=color,lw=1.6,zorder=2)
                ax.scatter([m],[y],s=33,marker=marker,color=color,edgecolor="white",lw=.55,zorder=3)
        ax.set(yticks=[2,1,0],yticklabels=[m[0] for m in models],xlim=xlim)
        ax.tick_params(axis="y",length=0)
        ax.set_xlabel("Area under curve")
        if metric=="auprc":
            ax.axvline(R["models"]["multivariable"]["internal"]["event_rate"],
                       color=BLUE,ls=":",lw=1)
            ax.axvline(R["models"]["multivariable"]["external"]["event_rate"],
                       color=CORAL,ls=":",lw=1)
    fig.text(.5,.015,"Circles: MIMIC internal; squares: eICU external. Lines: 95% patient or hospital-cluster bootstrap intervals.",
             ha="center",fontsize=7,color=GRAY)
    fig.subplots_adjust(bottom=.19,left=.22,right=.97,top=.86)
    save(fig,"Figure2_model_performance")


def smooth_calibration(y,p,grid):
    p=np.clip(np.asarray(p),1e-5,1-1e-5)
    lo=np.log(p/(1-p)).reshape(-1,1)
    g=np.log(np.clip(grid,1e-5,1-1e-5)/(1-np.clip(grid,1e-5,1-1e-5))).reshape(-1,1)
    spline=SplineTransformer(n_knots=5,degree=3,knots="quantile",include_bias=False)
    X=spline.fit_transform(lo)
    fit=LogisticRegression(C=100,max_iter=3000).fit(X,y)
    return fit.predict_proba(spline.transform(g))[:,1]


def calibration():
    fig,axs=plt.subplots(1,2,figsize=(7.1,3.45),gridspec_kw={"wspace":.37})
    ax=axs[0]
    panel(ax,"A","Probability calibration")
    ax.plot([0,.6],[0,.6],color="#8497A3",ls="--",lw=1,label="Ideal")
    for cohort,color,label in [("internal",BLUE,"MIMIC internal"),("external",CORAL,"eICU external")]:
        df=P[cohort]
        p=df.multivariable.to_numpy(); y=df.event.to_numpy()
        grid=np.linspace(float(np.quantile(p,.01)),float(np.quantile(p,.99)),120)
        curve=smooth_calibration(y,p,grid)
        ax.plot(grid,curve,color=color,lw=2,label=label)
        bins=pd.qcut(p,10,labels=False,duplicates="drop")
        points=pd.DataFrame({"bin":bins,"p":p,"y":y}).groupby("bin").agg(pred=("p","mean"),obs=("y","mean"))
        ax.scatter(points.pred,points.obs,color=color,s=15,alpha=.7,edgecolor="white",lw=.3,zorder=3)
    ax.set(xlim=(0,.6),ylim=(0,.6),xlabel="Predicted probability",ylabel="Observed event fraction")
    ax.set_aspect("equal",adjustable="box")
    ax.legend(frameon=False,loc="upper left",fontsize=7)
    ax=axs[1]
    panel(ax,"B","Between-hospital calibration")
    sites=C["external_hospital_summary"]["anonymous_sites_sorted_by_testing_rate"]
    rate=np.array([s["testing_rate"] for s in sites])
    diff=np.array([s["observed_minus_predicted"] for s in sites])
    sizes=np.array([s["n_tested"] for s in sites])
    ax.axhline(0,color="#8497A3",ls="--",lw=1)
    ax.scatter(rate,diff,s=np.clip(sizes/8,18,120),color=VIOLET,alpha=.65,edgecolor="white",lw=.6)
    ax.set(xlabel="Hospital repeat-testing fraction",ylabel="Observed minus predicted risk",
           xlim=(.4,.91))
    ax.axhspan(-.02,.02,color=LIGHT,zorder=-2)
    fig.text(.5,.01,"Calibration curves use logistic splines; points show decile means. Hospitals: ≥50 eligible, ≥50 tested, ≥10 events.",
             ha="center",fontsize=7,color=GRAY)
    fig.subplots_adjust(bottom=.22,left=.12,right=.98,top=.86)
    save(fig,"Figure3_calibration_hospitals")


def observation():
    fig,axs=plt.subplots(1,3,figsize=(7.1,3.25),gridspec_kw={"wspace":.42,"width_ratios":[1,1,1.18]})
    a=axs[0]; panel(a,"A","Repeat testing")
    names=["MIMIC-IV","eICU"]
    rates=[O["mimic"]["testing_rate"],O["eicu"]["testing_rate"]]
    a.barh([1,0],rates,color=TEAL,height=.48,label="Retested")
    a.barh([1,0],[1-x for x in rates],left=rates,color="#DCE5E9",height=.48,label="No repeat test")
    for y,rate in zip([1,0],rates): a.text(rate-.025,y,f"{rate:.1%}",ha="right",va="center",color="white",fontweight="bold",fontsize=7)
    a.set(xlim=(0,1),yticks=[1,0],yticklabels=names,xlabel="Fraction of eligible patients")
    a.tick_params(axis="y",length=0)
    b=axs[1]; panel(b,"B","Hospital variation")
    source=load_for_hospitals()
    b.hist(source,bins=np.arange(0,1.01,.1),color=VIOLET,alpha=.82,edgecolor="white")
    b.axvline(.7,color=CORAL,ls="--",lw=1.2)
    b.set(xlim=(0,1),xlabel="eICU hospital testing fraction",ylabel="Hospitals")
    c=axs[2]; panel(c,"C","Unknown-label bounds")
    for y,name,color in [(1,"MIMIC-IV",BLUE),(0,"eICU",CORAL)]:
        o=O["mimic" if name=="MIMIC-IV" else "eicu"]
        low=o["full_cohort_event_risk_lower_bound"];high=o["full_cohort_event_risk_upper_bound"]
        observed=o["documented_event_rate_among_tested"]
        c.plot([low,high],[y,y],color=color,lw=4,solid_capstyle="round")
        c.scatter([observed],[y],color=NAVY,s=30,zorder=3,marker="D",edgecolor="white",lw=.4)
    c.set(xlim=(0,.5),yticks=[1,0],yticklabels=["MIMIC-IV","eICU"],xlabel="Event risk")
    c.tick_params(axis="y",length=0)
    fig.text(.5,.015,"Ranges assign unknown labels among untested patients to 0 or 1; they do not identify biological incidence. Diamonds: tested-patient rates.",
             ha="center",fontsize=7,color=GRAY)
    fig.subplots_adjust(bottom=.23,left=.12,right=.97,top=.84)
    save(fig,"Figure4_observation_process")


def load_for_hospitals():
    d=pd.read_csv(ROOT/"data_private"/"eicu_feasibility.csv",usecols=["hospital","followup_count"])
    h=d.groupby("hospital").agg(n=("hospital","size"),n_tested=("followup_count",lambda x:int((x>0).sum())))
    return (h.loc[h.n>=50,"n_tested"]/h.loc[h.n>=50,"n"]).to_numpy()


if __name__=="__main__":
    performance();calibration();observation()
    print("Exported Figures 2–4 as editable PDF/SVG plus PNG previews")
