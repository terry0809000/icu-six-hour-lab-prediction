# Execution interfaces — Six-hour ICU laboratory prediction

This guide describes the preserved source interfaces by reading the code. No research module was imported, no command was run, and no performance or portability validation is implied. Use the original project instructions for the scientific run order.

## Data → analysis → outputs

**Input contract:** Credentialed MIMIC-IV 3.1 and eICU 2.0; extraction paths must point to your authorized archives.

1. Acquire inputs from their original providers and configure the paths defined in the source.
2. Install the recorded dependencies in an isolated environment.
3. Preserve grouping, seeds, development/evaluation separation and any protocol/hash gates.
4. Run acquisition/extraction and preprocessing before model development. Run evaluation only after the selection/freeze required by the original implementation.
5. Keep raw data, row-level derivatives, predictions and fitted models outside the public repository.

## Script reference

### `code/augment_censoring.py`

[Open source](code/augment_censoring.py) · Executable entry point

Append early ICU death ascertainment before the model/data freeze. This uses only small stay/discharge tables. It does not alter eligibility, predictors, observed event labels, or the modelling sample.

### `code/bmc_calibration_sensitivity.py`

[Open source](code/bmc_calibration_sensitivity.py) · Executable entry point

Post hoc aggregate calibration assessment of existing frozen predictions. Reads no patient identifiers and never refits or updates a prediction model. Hospital labels are used only in memory for the original cluster resampling.

### `code/build_multimedia_appendix.py`

[Open source](code/build_multimedia_appendix.py) · Supporting module

Build a nonidentifying supplementary methods/results appendix from frozen aggregates.

### `code/evaluate_frozen_models.py`

[Open source](code/evaluate_frozen_models.py) · Executable entry point

Stage 2: use verified frozen development artifacts for internal/external tests.

### `code/extract_feasibility.py`

[Open source](code/extract_feasibility.py) · Executable entry point

Stream authorized ICU source ZIPs; keep patient-level extracts private. The protocol was locked before this script was run. No model is fitted here.

### `code/figure1_design.py`

[Open source](code/figure1_design.py) · Supporting module

Scientific Figure 1: editable, data-true study and ascertainment schematic.

### `code/figures_results.py`

[Open source](code/figures_results.py) · Executable entry point

Publication figures from frozen predictions and aggregate result files. Patient-level predictions stay in data_private; exported graphics show only aggregate estimates. Run after the one-shot external evaluation.

### `code/fit_frozen_models.py`

[Open source](code/fit_frozen_models.py) · Executable entry point

Stage 1: train/select on MIMIC development only, then freeze artifacts.

### `code/model_and_evaluate.py`

[Open source](code/model_and_evaluate.py) · Executable entry point

Frozen development-only selection and held-out evaluation for protocol v1.

### `code/observation_analysis.py`

[Open source](code/observation_analysis.py) · Executable entry point

Prespecified observation-process and censoring sensitivity analyses.

### `code/prepare_bmc_figures.py`

[Open source](code/prepare_bmc_figures.py) · Supporting module

Re-export existing data-faithful figures at BMC's 170-mm full width.

### `code/result_availability_sensitivity.py`

[Open source](code/result_availability_sensitivity.py) · Supporting module

Post-freeze sensitivity restricted to fully reported baseline phosphate results. The primary retrospective cohort excludes low phosphate specimens collected in the first six hours even if their final results arrived later. This restriction checks the frozen model only in patients whose every baseline phosphate result was available by the landmark. It cannot add excluded patients back and is not a redefinition of the primary cohort.

### `code/result_contrasts.py`

[Open source](code/result_contrasts.py) · Executable entry point

Post-freeze paired contrasts and anonymous hospital heterogeneity summary.

### `experiments/post_release_tuning_20260924/run_dev_only_sweep.py`

[Open source](experiments/post_release_tuning_20260924/run_dev_only_sweep.py) · Executable entry point

Post-release exploratory tuning on MIMIC development only. Never reads eICU or writes patient-level data. Existing frozen artifacts and the submission archive remain unchanged. The results cannot be treated as a new independent external validation.

### `experiments/post_release_tuning_20260924/run_logistic_c_extension.py`

[Open source](experiments/post_release_tuning_20260924/run_logistic_c_extension.py) · Supporting module

Development-only check beyond the original logistic C-grid upper boundary.

## Reproduction limits

Absolute paths, frozen-file references and platform-specific assumptions are preserved. Configure a matching local workspace rather than bypassing scientific integrity guards. A repository containing existing scripts is not evidence that every data or environment prerequisite is satisfied. No original clinical results or completed-run artifacts are redistributed.
