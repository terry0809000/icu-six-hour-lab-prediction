"""Re-export existing data-faithful figures at BMC's 170-mm full width."""
from pathlib import Path
import runpy, re
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'deliverables/work_in_progress/BMC_MIDM_20260926/figures'
OUT.mkdir(parents=True,exist_ok=True)
# Run the original generator in memory while redirecting saves. No frozen files change.
def render(script):
    text=script.read_text(encoding='utf-8')
    text=text.replace('OUT=ROOT/"figures"',f'OUT=Path({str(OUT)!r})')
    if script.name=='figure1_design.py':
        text=text.replace('fig=plt.figure(figsize=(7.1,5.6)', 'fig=plt.figure(figsize=(170/25.4,5.1)')
        text=text.replace('ylim=(0,1)', 'ylim=(0,.88)')
        text='\n'.join(l for l in text.splitlines() if not any(x in l for x in ['label(ax,.04,.955','label(ax,.04,.91','ax.plot([.04,.96],[.885']))
        text=text.replace('"48-hour horizon"','"Discharge or 48 h"')
        text=text.replace('Baseline P ≥ 2.0 mg/dL','Baseline P ≥ 2.0 mg/dL').replace('Does early low-phosphate risk transport when the outcome is only observed after repeat testing?', 'Does prediction of a recorded low result transport across different repeat-testing practices?')
        text=text.replace('Report testing rates, bounds, weighting assumptions','Report testing, label completions, weighting assumptions')
    text=text.replace('figsize=(7.1,','figsize=(170/25.4,')
    text=text.replace('if metric=="auprc":\n', 'if metric=="auprc":\n            ax.set_yticklabels([])\n')
    # Preserve coordinates and export an editable SVG plus embedded-font PDF and PNG.
    # Enforce exact physical width at export without using raster resampling.
    text=text.replace('bbox_inches="tight",pad_inches=.045','bbox_inches=None,pad_inches=0')
    text=text.replace('bbox_inches="tight",pad_inches=.02','bbox_inches=None,pad_inches=0')
    text=text.replace('Unknown-label bounds','Label completion')
    text=text.replace('xlabel="Event risk"','xlabel="Completed-label fraction"')
    text=text.replace('Ranges assign unknown labels among untested patients to 0 or 1; they do not identify biological incidence. Diamonds: tested-patient rates.', 'Ranges: hypothetical label completions. Diamonds: event fractions among retested patients.')
    text=text.replace('Circles: MIMIC internal; squares: eICU external. Lines: 95% patient or hospital-cluster bootstrap intervals.', 'Circles: MIMIC internal; squares: eICU external. Lines: 95% bootstrap intervals.')
    text=text.replace('Calibration curves use logistic splines; points show decile means. Hospitals: ≥50 eligible, ≥50 tested, ≥10 events.', 'Curves: logistic splines; points: decile means. Hospitals: ≥50 eligible, ≥50 tested, ≥10 events.')
    exec(compile(text,str(script),'exec'),{'__file__':str(script),'__name__':'__main__'})
render(ROOT/'code/figure1_design.py')
render(ROOT/'code/figures_results.py')
print(OUT)
