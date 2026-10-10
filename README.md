# Six-hour ICU laboratory prediction

**Health Informatics · Research pipeline · Source-code release**

This project provides the author's existing analysis implementation for six-hour icu laboratory prediction. Scientific source files are preserved from the local research product. This upload organizes code and documentation; it does not represent new experiments, model retraining or an independently verified reproduction.

## Pipeline

1. **Acquire inputs.** Credentialed MIMIC-IV 3.1 and eICU 2.0; extraction paths must point to your authorized archives.
2. **Prepare the environment.** Use the project-specific dependency files listed below in an isolated environment. Preserve the versions and seeds recorded in the source release.
3. **Prepare data and run the model or analysis.** Follow the original execution notes and entry points below, preserving time order, participant grouping and fixed evaluation boundaries.
4. **Evaluate and summarize.** Use the included evaluation/analysis scripts. Write generated data, predictions, fitted objects and caches to a private working copy.

This is a source-only repository. Raw data, individual-level derivatives and fitted checkpoints are not bundled. Some saved-evidence reproduction commands in the original notes require artifacts from the original source release; obtain those independently where authorized.

## Environment

The preserved release does not include a complete dependency lock. Refer to original execution notes and imports; no new environment compatibility claim is made.

## Execution

See the [pipeline and command reference](PIPELINE.md) for entry points, required arguments and the source-specific input contract.

Run commands from the repository root unless the original notes specify another working directory. Use a fresh copy and configure local data paths before running.

### Original execution notes

- [README.md](docs/original/README.md)

### Scientific source inventory

<details>
<summary>Browse the preserved scientific implementation</summary>

- [`code/augment_censoring.py`](code/augment_censoring.py)
- [`code/bmc_calibration_sensitivity.py`](code/bmc_calibration_sensitivity.py)
- [`code/build_multimedia_appendix.py`](code/build_multimedia_appendix.py)
- [`code/evaluate_frozen_models.py`](code/evaluate_frozen_models.py)
- [`code/extract_feasibility.py`](code/extract_feasibility.py)
- [`code/figure1_design.py`](code/figure1_design.py)
- [`code/figures_results.py`](code/figures_results.py)
- [`code/fit_frozen_models.py`](code/fit_frozen_models.py)
- [`code/model_and_evaluate.py`](code/model_and_evaluate.py)
- [`code/observation_analysis.py`](code/observation_analysis.py)
- [`code/prepare_bmc_figures.py`](code/prepare_bmc_figures.py)
- [`code/result_availability_sensitivity.py`](code/result_availability_sensitivity.py)
- [`code/result_contrasts.py`](code/result_contrasts.py)
- [`experiments/post_release_tuning_20260924/run_dev_only_sweep.py`](experiments/post_release_tuning_20260924/run_dev_only_sweep.py)
- [`experiments/post_release_tuning_20260924/run_logistic_c_extension.py`](experiments/post_release_tuning_20260924/run_logistic_c_extension.py)

</details>

## Repository contents and rights

Only scientific code, dependencies, configuration and execution documentation are included. Manuscripts, submission documents, raw participant records, trained models, credentials, caches and software environments are excluded. Notebook cell outputs and execution counters are cleared for publication; code cells are retained.

Existing source and third-party license notices govern their respective files. No blanket license is invented for material whose original release did not specify one. Original dataset licenses and access requirements continue to apply.

Research software is not a clinical deployment claim. Negative findings and the source release's interpretation limits remain applicable. No tests or experiments were run during this upload, as requested by the author.
