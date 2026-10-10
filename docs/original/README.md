# Additional file 2 — Code and aggregate evidence

Target: BMC Medical Informatics and Decision Making. Author: Tianyi Yu.

## Contents and access

Only source code, aggregate JSON results and protocol/mapping documents are included. No participant rows, patient identifiers, individual predictions, model objects or unredacted interaction transcripts are distributed. Obtain MIMIC-IV 3.1 and eICU-CRD 2.0 through PhysioNet using your own credentials and data-use agreements. The public files support code review and numerical tracing, but cannot directly instantiate the original fitted score.

## Recompute in a new credentialed workspace

Preserve the supplied `results` directory as reference evidence. In a separate copy of this archive, rename it `reference_results` and create fresh `results` and `data_private` directories. Do not execute the refitting sequence inside the original study workspace. Point extraction constants to your authorized source ZIPs. Code uses pathlib and Python; required libraries include NumPy, pandas, SciPy, scikit-learn, joblib and matplotlib. The original software versions were not fully recorded at freeze; supplemental JSONs report the versions for their execution. BMC preparation used Python 3.12.14/scikit-learn 1.9.1/pandas 3.0.6/NumPy 2.5.3/SciPy 1.18.1. These are not asserted to be the exact original fitting versions.

From the new workspace, execute in order:

1. `python code/extract_feasibility.py`
2. `python code/augment_censoring.py`
3. `python code/fit_frozen_models.py`
4. `python code/evaluate_frozen_models.py`
5. `python code/observation_analysis.py`
6. `python code/result_contrasts.py`
7. `python code/result_availability_sensitivity.py`
8. `python code/bmc_calibration_sensitivity.py`
9. `python code/figure1_design.py` and `python code/figures_results.py`
10. `python code/prepare_bmc_figures.py` for journal-sized exports.

The shared module `code/model_and_evaluate.py` is required by the frozen-stage scripts and is included. The freeze script deliberately refuses to overwrite an existing manifest. Local path changes and software changes may change fingerprints or numerical outputs; supplied reference aggregates remain the historical record.

The dated exploratory tuning scripts under `experiments/post_release_tuning_20260924` may then be run separately. Their selection is confined to original MIMIC development folds and their internal comparison reuses the previously evaluated holdout. They do not read eICU patient rows. Do not reinterpret them as prespecified model selection or a new independent validation. Calibration-in-the-large is a post hoc description of frozen predictions, not externally fitted replacement predictions.

## Interpretation of historical result keys

Some frozen JSON keys use `full_cohort_event_risk_lower_bound` and `upper_bound`. The BMC manuscript corrects this terminology: these numerical endpoints are hypothetical binary assay-label completions with observed labels fixed. The actual full-cohort recorded-event fraction is known; neither these completions nor that fraction identify biological incidence or standardized-testing risk. Original keys are preserved to maintain traceability.

Individual metric intervals use 400 bootstrap replicates. Paired model contrasts, tuning contrasts and the additional overall calibration assessment use 600. AUPRC is average precision. Follow-up eligibility uses specimen collection time, not a reporting deadline. Cohort eligibility can depend on baseline specimens reported after the landmark. These limitations must accompany any reuse.
