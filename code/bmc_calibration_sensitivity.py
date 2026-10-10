"""Post hoc aggregate calibration assessment of existing frozen predictions.

Reads no patient identifiers and never refits or updates a prediction model.
Hospital labels are used only in memory for the original cluster resampling.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PRIVATE = ROOT / "data_private"
SEED = 240924
B = 600


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def offset_intercept(y, logits):
    """Solve the intercept-only binomial score with fixed unit offset slope."""
    observed = float(np.mean(y))
    if not 0 < observed < 1:
        raise ValueError("Finite offset intercept requires both outcome classes")
    score = lambda alpha: float(np.mean(expit(alpha + logits)) - observed)
    lower = -float(np.max(logits)) - 40.0
    upper = -float(np.min(logits)) + 40.0
    alpha = float(brentq(score, lower, upper, xtol=1e-12, rtol=1e-14))
    residual = abs(score(alpha))
    if residual > 1e-10:
        raise ArithmeticError("Offset calibration score equation failed")
    return alpha, residual


def calculate(y, p, logits):
    alpha, residual = offset_intercept(y, logits)
    observed, expected = float(np.mean(y)), float(np.mean(p))
    return {
        "calibration_in_the_large": alpha,
        "mean_predicted_probability": expected,
        "observed_event_fraction": observed,
        "observed_minus_predicted_probability": observed - expected,
        "observed_expected_ratio": observed / expected,
    }, residual


def analyze(label, original):
    # usecols prevents identifiers and unused clinical fields from being loaded.
    frame = pd.read_csv(PRIVATE / f"{label}_predictions.csv",
                        usecols=["event", "multivariable", "hospital"])
    y = frame.event.to_numpy(dtype=float)
    p = frame.multivariable.to_numpy(dtype=float)
    assert np.isfinite(y).all() and np.isin(y, [0, 1]).all()
    assert np.isfinite(p).all() and ((p > 0) & (p < 1)).all()
    assert frame.hospital.notna().all()
    reference = original["models"]["multivariable"][label]
    assert len(y) == reference["n"] == original["split"][f"{label}_n"]
    assert int(y.sum()) == reference["events"] == original["split"][f"{label}_events"]
    assert abs(float(y.mean()) - reference["event_rate"]) < 1e-14
    assert abs(float(np.mean((y - p) ** 2)) - reference["brier"]) < 1e-14
    logits = logit(p)  # Exact frozen probabilities: no clipping or updating.
    point, point_residual = calculate(y, p, logits)
    clustered = label == "external"
    groups = None
    if clustered:
        h = frame.hospital.astype(str).to_numpy()
        unique = np.unique(h)
        assert len(unique) == original["split"]["external_tested_hospitals"]
        groups = {key: np.flatnonzero(h == key) for key in unique}
    rng = np.random.default_rng(SEED + int(clustered))
    samples = {key: [] for key in point}
    max_residual = point_residual
    skipped = 0
    for _ in range(B):
        if groups is None:
            idx = rng.integers(len(y), size=len(y))
        else:
            idx = np.concatenate([groups[key] for key in
                                  rng.choice(unique, size=len(unique), replace=True)])
        if np.unique(y[idx]).size < 2:
            skipped += 1
            continue
        values, residual = calculate(y[idx], p[idx], logits[idx])
        max_residual = max(max_residual, residual)
        for key, value in values.items():
            samples[key].append(value)
    assert B - skipped > 0
    return {
        "n": int(len(y)), "events": int(y.sum()),
        "hospital_clusters": int(len(unique)) if clustered else None,
        "bootstrap_unit": "hospital" if clustered else "patient",
        "seed": SEED + int(clustered), "bootstrap_requested_replicates": B,
        "bootstrap_valid_replicates": B - skipped,
        "bootstrap_single_class_replicates_skipped": skipped,
        "metrics": {key: {"estimate": value,
            "ci95_percentile": [float(x) for x in np.quantile(samples[key], [.025, .975])]}
            for key, value in point.items()},
        "verification": {
            "counts_event_fraction_and_brier_match_original": True,
            "point_absolute_mean_score_residual": point_residual,
            "maximum_absolute_mean_score_residual_including_bootstraps": max_residual,
            "original_jointly_fitted_calibration_intercept": reference["calibration_intercept"],
            "original_jointly_fitted_calibration_slope": reference["calibration_slope"],
        },
    }


def main():
    source_paths = [PRIVATE / "internal_predictions.csv", PRIVATE / "external_predictions.csv",
                    RESULTS / "model_performance.json", RESULTS / "frozen_model_manifest.json",
                    ROOT / "code" / "model_and_evaluate.py",
                    ROOT / "code" / "evaluate_frozen_models.py",
                    ROOT / "code" / "result_contrasts.py"]
    before = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
              for path in source_paths}
    original = json.loads((RESULTS / "model_performance.json").read_text(encoding="utf-8"))
    # Analytic sanity check: a constant prediction of .2 and event rate of .3.
    test_y = np.array([1.] * 3 + [0.] * 7)
    alpha, residual = offset_intercept(test_y, np.full(10, logit(.2)))
    expected_alpha = float(logit(.3) - logit(.2))
    assert abs(alpha - expected_alpha) < 1e-10
    result = {
        "analysis": "Post hoc BMC supplemental calibration assessment",
        "date": "2026-09-26", "model": "multivariable", "post_hoc": True,
        "prediction_model_refitted_or_updated": False,
        "method": {
            "citl": "Unpenalized intercept-only logistic regression: logit(P(Y=1)) = alpha + offset(logit(p)); offset slope fixed at 1. Solve mean(expit(alpha+logit(p))) = mean(y).",
            "observed_expected_ratio": "sum(y)/sum(p), calculated from unchanged frozen probabilities",
            "confidence_intervals": "2.5th and 97.5th percentiles of 600 bootstrap draws per evaluation set",
            "internal_resampling": "Sample n individual patient rows with replacement; source cohort contains one stay per person.",
            "external_resampling": "Convert original hospital column to strings; sample 173 unique hospital labels with replacement; concatenate all rows of each selected hospital, preserving repeat selections. Same schema and seeds as result_contrasts.py.",
            "seeds": {"internal": SEED, "external": SEED + 1},
            "probability_clipping": "None; require all frozen probabilities strictly between zero and one.",
            "interpretation": "Positive CITL or observed/expected above 1 indicates overall underprediction. CITL is distinct from the original intercept jointly fitted with a free calibration slope.",
            "limitations": "Post hoc descriptive assessment among patients with follow-up phosphate testing. Uncertainty conditions on the fitted frozen model and realized evaluation cohort; bootstrap does not include model-development uncertainty or eliminate informative testing, residual confounding, or hospital selection. No recalibrated predictions or externally tuned model are created.",
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "analytic_solver_check": {"absolute_intercept_error": abs(alpha - expected_alpha),
                                  "absolute_mean_score_residual": residual},
        "internal": analyze("internal", original),
        "external": analyze("external", original),
        "source_sha256": before,
        "analysis_code_sha256": sha(Path(__file__)),
    }
    after = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
             for path in source_paths}
    assert before == after, "An original source file changed during analysis"
    result["source_hashes_unchanged_after_analysis"] = True
    out = RESULTS / "bmc_calibration_sensitivity.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Post hoc calibration supplement", "", "Generated 26 September 2026 from existing frozen internal and external multivariable-model predictions. This analysis was added after the original evaluation; it was not prespecified. No prediction model was fitted, updated, or externally tuned.", "", "## Method", "", result["method"]["citl"], "", "The originally reported calibration intercept was jointly estimated with a free slope and is not calibration-in-the-large (CITL). Mean predictions, observed event fractions, their difference, and observed/expected (O/E) ratios use the unchanged frozen predictions.", "", "Internal uncertainty uses 600 patient bootstrap draws (seed 240924). External uncertainty uses 600 hospital cluster draws (seed 240925): sample all 173 hospital labels with replacement and retain all patients from each sampled hospital, including repeated clusters. This reproduces the resampling schema and seeds in code/result_contrasts.py. All intervals are percentile 95% confidence intervals; no probability clipping was needed.", "", "## Aggregate results", "", "| Metric | Internal | External |", "|---|---:|---:|"]
    labels = {"calibration_in_the_large": "CITL (log-odds)",
              "mean_predicted_probability": "Mean predicted probability",
              "observed_event_fraction": "Observed event fraction",
              "observed_minus_predicted_probability": "Observed minus predicted probability",
              "observed_expected_ratio": "Observed/expected ratio"}
    lines += [f"| Patients / events | {result['internal']['n']} / {result['internal']['events']} | {result['external']['n']} / {result['external']['events']} |"]
    for key, title in labels.items():
        cells = []
        for label in ["internal", "external"]:
            item = result[label]["metrics"][key]
            lo, hi = item["ci95_percentile"]
            cells.append(f"{item['estimate']:.4f} ({lo:.4f} to {hi:.4f})")
        lines.append(f"| {title} | {' | '.join(cells)} |")
    lines += ["", "Values in parentheses are 95% percentile bootstrap intervals. Positive CITL and O/E above 1 indicate overall underprediction.", "", "## Verification and limitations", "", "Patient/event counts, event fractions, and Brier scores agree with the original model_performance.json; the external set includes the original 173 hospitals. Source SHA-256 hashes were identical before and after analysis. The script loads no patient identifiers; hospital labels stay in memory and are not exported. Outputs contain aggregate statistics only.", ""]
    for label in ["internal", "external"]:
        entry = result[label]
        residual = entry["verification"]["maximum_absolute_mean_score_residual_including_bootstraps"]
        lines.append(f"- {label.capitalize()}: {entry['bootstrap_valid_replicates']}/{B} finite bootstrap estimates; maximum absolute mean offset-score residual {residual:.3g}.")
    lines += [f"- Analytic constant-risk solver check: absolute intercept error {result['analytic_solver_check']['absolute_intercept_error']:.3g}.", "", result["method"]["limitations"], "", "Reproduction: run code/bmc_calibration_sensitivity.py with NumPy, pandas and SciPy. Exact software versions, numerical checks, seeds and source hashes are recorded in results/bmc_calibration_sensitivity.json.", ""]
    supplement = ROOT / "review_packages" / "BMC_20260926" / "CALIBRATION_SUPPLEMENT.md"
    supplement.parent.mkdir(parents=True, exist_ok=True)
    supplement.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["internal", "external", "analytic_solver_check", "source_hashes_unchanged_after_analysis"]}, indent=2))


if __name__ == "__main__":
    main()
