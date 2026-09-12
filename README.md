# WB-PIML for Cf/SiC fatigue

Physics-based lifetime prediction with statistical residual correction for
multi-source Cf/SiC fatigue data containing exact fractures and right-censored
runouts. This **0912 repository release** supports the revised manuscript and uses
the supplied 0909 canonical analysis, dataset and results. It evaluates both point
and probabilistic predictions, including two independently selected simple-trunk
controls.

## Data and method

`data.xlsx`, worksheet `Plot_Data_Verified`, contains **223 retained records:
158 exact fractures and 65 runouts, from 18 sources and 17 reviewed campaigns**.
The workbook includes provenance, labels and campaign assignments. Distinct
record-equivalence IDs are a structural check, not proof that all cross-publication
experiments are independent; remaining provenance qualifications are retained.

The physical trunk combines Walker–Basquin mechanical damage and a time-dependent
environmental term, fitted with joint exact-density/right-censored-survival
likelihood. A residual learner uses exact fractures only; its contribution is
weighted by temperature support from the training data (250 °C temperature scale).
Runouts also inform training-only scale calibration and model selection, but are
never treated as exact lives in residual regression or point-error metrics.

Primary evaluation uses five campaign-disjoint outer folds and campaign-disjoint
development folds. Source and campaign identifiers are not predictive inputs.
Candidates within each model pool are selected on development data: retain
campaign-balanced RMSE within 0.020 of the pool minimum, then minimize calibrated
runout survival NLL. The pools are independently selected and have different sizes:

| Model | Candidates | Role |
| --- | ---: | --- |
| WB-PIML | 54 | Competing-damage trunk plus residual correction |
| WB-Residual | 18 | Simple Walker–Basquin trunk plus residual correction |
| Basquin-Residual | 6 | Simple trunk with Walker gamma fixed to zero |
| ML-Strong | 64 | Selected control pool, including nine physics-feature candidates |

Basquin-Residual retains stress-ratio-related inputs in its residual learner.
These comparisons evaluate complete selected prediction pipelines, not the
isolated causal contribution of each physical term. Fixed-configuration repeated
validation and retrained ablation remain distinct analyses.

## Installation and execution

The reference run records Python **3.14.0**, NumPy 2.3.5, pandas 3.0.1, SciPy
1.17.1 and scikit-learn 1.8.0. The remaining pins in `requirements.txt` come from
the local packaging environment; the original protocol did not record their versions.

```bash
python -m pip install -r requirements.txt
python verify_release.py
python run.py
```

`run.py` keeps the previous entry point and writes new output to `results/`.
It invokes the short `WB-PIML.py` compatibility loader, which verifies and executes
the four files in `wb_piml_parts/` in their original order and in one shared
namespace. Concatenating those four files reconstructs the supplied 6368-line
analysis source byte-for-byte (SHA-256
`ec5e39f4ff50ef12c942e3a88f992a48a678fc30a0c9eaa5b37e32f2a193f793`).

| Source file | Contents |
| --- | --- |
| `wb_piml_parts/part_01_core.py` | Configuration, data loading, physical/residual models and primary metrics |
| `wb_piml_parts/part_02_validation_reporting.py` | Sensitivity, validation, figures and result export |
| `wb_piml_parts/part_03_revision_models.py` | Revision-stage audits, probability records and outer-fold tasks |
| `wb_piml_parts/part_04_workflow_probability.py` | Execution workflow, probability scoring, simple-trunk controls and entry point |

The fragments intentionally share a namespace and are not independent modules.
`wb_piml_parts/source_manifest.json` records their line ranges and hashes. This
byte-preserving layout avoids numerical or control-flow changes while keeping each
GitHub source file reviewable.

To specify the complete reference settings explicitly:

```bash
python run.py --data data.xlsx --out results --trees 400 --audit-trees 160 --repeats 10 --bootstrap 5000 --jobs 8 --tree-jobs 1 --blas-threads 1
```

The seed is fixed in the source at `20260824`. Full nested selection, factor
holdouts and sensitivity analyses can take substantial time; reduced settings
will not reproduce the reference run. Use `python run.py --help` for options and
`--export-csv` for optional table exports. Direct `python WB-PIML.py` execution
instead defaults to `WB_PIML_results/`. Neither default overwrites `reference_results/`.

## Reference results

The complete supplied run is in [`reference_results/`](reference_results/):

- [`WB_PIML_results.xlsx`](reference_results/WB_PIML_results.xlsx): **57 worksheets**,
  including predictions, folds, selection traces, probability calibration and audits.
- [`protocol.json`](reference_results/protocol.json): executed settings and data hash.
- [`README_results.md`](reference_results/README_results.md): detailed method and results.
- Ten composite PNGs, numbered 01–10. Figure 09 covers probability comparisons;
  figure 10 covers simple-trunk comparisons.
- `WB_PIML_figures.pptx`: the original **29-slide presentation for figures 01–08**.
  The new figures 09–10 are separate PNGs and are not included in that presentation.

The original source-generated README retains an inherited eight-figure inventory;
that sentence is corrected in the packaged reference README only. No numerical
results or analysis code were changed during repository preparation. PNG sequence
numbers are program-output identifiers, not the revised manuscript figure numbers.

Useful worksheet groups:

| Question | Worksheets |
| --- | --- |
| Primary point predictions and comparisons | `Exact_Outer_Pred`, `Runout_Outer_Pred`, `Primary_Campaign_Metrics`, `Paired_Bootstrap` |
| Distribution predictions and complete censored NLL | `Prob_Predictions`, `Prob_Metrics`, `Prob_Paired_CI`, `Prob_Calibration`, `Prob_Method` |
| Simple-trunk comparisons | `Trunk_Metrics`, `Trunk_Paired_CI`, `Trunk_Empirical_PI`, `Trunk_Method`, `Trunk_External_Candidates` |
| Empirical versus distribution intervals | `Empirical_PI_Summary`, `Empirical_PI_Rows`, `Prob_Intervals` |
| Splits, provenance and limitations | `Outer_Fold_Assignment`, `Split_Audit`, `Campaign_Map`, `Duplicate_Candidates`, `Claim_Gate` |
| Factor-held-out stress tests | `External_Summary`, `External_Paired_CI`, `Prob_Metrics` (scenario rows) |

## Findings and limits

Primary campaign-balanced results (`Trunk_Metrics`, scenario `Primary`; lower
RMSE and NLL are better):

| Model | Exact-fracture RMSE on log10 life | Complete censored NLL |
| --- | ---: | ---: |
| WB-PIML | 1.105287 | 1.560045 |
| WB-Residual | 1.143115 | 1.655097 |
| Basquin-Residual | 1.110583 | 1.651634 |

The paired RMSE gain over ML-Ens is approximately 0.067, with a 95% campaign-bootstrap
interval of [-0.065, 0.196]. Point-prediction improvement is modest and uncertain.
The complete censored NLL improves against both simple-trunk controls: paired
95% gain intervals are [0.036, 0.150] and [0.030, 0.147], respectively. This does
not establish superiority over every comparator: WB-CD has a lower primary NLL
point estimate, and the WB-PIML-Ox ablation has a slightly lower RMSE point estimate.

Performance is not uniformly better across holdouts. For example, temperature
holdout (LOTO) improves point RMSE relative to the simple trunks but gives worse
complete censored NLL (1.824 versus approximately 1.657 and 1.651).

Empirical residual intervals and parametric distribution intervals are different
outputs. Coverage is assessed on observed exact failures only; it is not unconditional
coverage of latent lifetimes under censoring. The fitted scale combines specimen
scatter and predictive error and is constant per trained model, rather than an
identified material-specific scatter law. A runout prediction below its stopping
cycle is a diagnostic, not a prediction-accuracy percentage.

LOSO/LOAO/LOTO/LOEO are factor-held-out stress tests on the existing compilation,
not new independent external datasets. Their splits do not enforce campaign
separation. The simple-trunk comparisons were added during revision after earlier
results were inspected; bootstrap intervals condition on the studied campaigns.
These results do not guarantee absence of overfitting, identify separate physical
damage mechanisms, or demonstrate universal model superiority.

## Release verification

[`release_manifest.json`](release_manifest.json) records SHA-256 hashes of packaged
files, original artifact hashes and the sole reference-README correction.
`verify_release.py` verifies every packaged file, reconstructs and hashes the
canonical source from the four fragments, loads the data through the split entry
point, checks model candidate counts and primary fold assignments, and recomputes
primary point metrics and distribution scores from saved predictions. These are
deterministic equivalence and saved-result consistency checks, not a fresh full
training run or independent validation. See [`CHANGELOG.md`](CHANGELOG.md) for the
release history.
