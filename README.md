# WB-PIML for Cf/SiC fatigue

This **9 October 2026 revision release** accompanies *Physics-informed machine
learning for fatigue life prediction of Cf/SiC ceramic matrix composites*. It uses
the completed `full_20261008_182317_470433` run and the October 9 IJF revision.
The analysis combines Walker–Basquin mechanical and environmental damage,
right-censored likelihood and environment-shared residual learning.

## Data and method

`data.xlsx`, sheet `Plot_Data_Verified`, contains **224 records: 158 fractures and
66 runouts, from 18 sources and 17 campaigns**. The Min 2013 runout (`row_id=315`)
is restored. Source traceability, duplicate groups and campaign labels are retained.

Runouts contribute lower life bounds to physical fitting and probability
calibration; exact-life residual regression and point-error metrics use fractures.
Five campaign-disjoint outer folds evaluate the primary model. Selection and
calibration use development data. Source/campaign IDs are not predictive inputs.

The physical trunk estimates the water/oxygen factor `10**beta`, `beta >= 0`,
inside training partitions. Every hybrid residual family excludes `env_class`
and environment-derived features. Identical non-environmental conditions receive
identical residual corrections across environments, preserving the trunk's
water/air ordering in complete predictions. Direct ML controls retain environment
inputs. Temperature attenuation retains the 250 °C support scale.

| Model pool | Candidates |
| --- | ---: |
| WB-PIML | 54 |
| WB-Residual | 18 |
| Basquin-Residual | 6 |
| ML-Strong | 64 |

Pools are independently selected: retain campaign-balanced fracture RMSE within
0.020 of the pool minimum, then minimize calibrated runout NLL. Simple-trunk
controls, retrained ablations, fixed-configuration repeated validation and
conditional sensitivity are distinct analyses.

## Install and run

The completed run used **Python 3.14.0**. `requirements.txt` pins the actual
numerical/export packages, including NumPy 2.3.5, pandas 3.0.1, SciPy 1.17.1 and
scikit-learn 1.8.0. Other platform/library builds can introduce numeric differences.

```bash
python -m pip install -r requirements.txt
python run.py --check --jobs 2
python verify_release.py
```

Full workflow, using a new, empty output directory:

```bash
python run.py --jobs 16 --out runs/reproduction_1009
```

Without `--jobs`, the original worker-budget rule uses physical cores and available
memory. Without `--out`, outputs go to `runs/full_TIMESTAMP`. Defaults remain seed
20260824, 400 trees, 160 audit trees, 10 repeats and 5,000 bootstrap draws. A full
run includes primary analysis, all 34 sensitivity variants, strict holdouts,
verification and a frozen prospective model.

`WB-PIML.py` is an equivalent entry point; `run.load_analysis()` permits inspection.
Twelve short files execute in order in a shared namespace. Use a launcher, not an
individual part. Hash-verified reconstruction in the ignored `.wb_piml_runtime/`
cache preserves definition overrides, source checks and multiprocessing behavior.
Only default data/output paths move to the repository. The directory must be writable.

Supplemental rerun against this release's primary workbook:

```bash
python run.py --reviewer-only --baseline reference_results/primary/WB_PIML_results.xlsx --out runs/reviewer_recheck
```

## Source files

| File in `wb_piml_parts/` | Contents |
| --- | --- |
| `part_01_data_physics.py` | Configuration, data and physical fitting |
| `part_02_residual_metrics.py` | Residual models and primary scores |
| `part_03_validation.py` | Sensitivity and validation |
| `part_04_reporting.py` | Figures, workbook and result notes |
| `part_05_training_selection.py` | Training policies and censored controls |
| `part_06_claim_provenance.py` | Claim checks, provenance and probability records |
| `part_07_primary_workflow.py` | Outer-fold tasks and primary workflow |
| `part_08_probability_controls.py` | Distribution scoring and simple trunks |
| `part_09_environment_refits.py` | Environmental fitting and reviewer refits |
| `part_10_execution_checks.py` | Verification, checks and orchestration |
| `part_11_frozen_prediction.py` | Frozen inference and prospective evaluation |
| `part_12_water_strict_holdouts.py` | Water identifiability and strict holdouts |

The working script is distributed through these twelve files with minimal
comments. Its equations, candidates, seeds, fitting rules and validation settings
are preserved. Removed comments/documentation change the source hash; this release
records both original and public hashes. Executable algorithm syntax trees were
compared before packaging. Only documentation, integrity bookkeeping and usage
instructions differ. `source_equivalence.json` records this comparison. No
single-file working script is tracked in the repository.

## Reference results and verification

`reference_results/` preserves **all 290 original run artifacts**:

- `primary/`: 57-sheet workbook, protocol, ten generated PNGs and slide deck.
- `reviewer/`: sensitivity, selection, environmental-order, censoring and strict audits.
- `prospective/`: frozen model and future-experiment templates.
- Original workflow metadata, analysis plans and `run.log`.

The large strict-calibration table is stored losslessly as
`reviewer/strict_holdout/calibration.csv.gz`; its decompressed hash is verified.
The other 289 run files keep their original bytes. Historical paths and source
hashes describe the original run; readers do not need those local paths.

`verify_release.py` checks release hashes, data counts, campaign partitions, saved
point/distribution scores, 34 sensitivity variants and 170 checkpoints, strict
isolation, water/air ordering and frozen inference. Archived sensitivity source
fingerprints are adapted only in a disposable audit copy to the public source.
Archived files and numeric values stay unchanged.

```bash
python verify_release.py --report runs/verification.json
python verify_release.py --refit-selected --report runs/selected_refits.json
```

The default verifier does not train models. `--refit-selected` additionally fits
the five selected pipelines and their calibration folds, plus the selected final
model. `verification_report.json` records release checks;
`full_reproduction_report.json` records a separate complete rerun and its numeric
comparison with the supplied run. Comparison tolerance is `rtol=1e-10, atol=1e-10`.
Paths, source fingerprints, timestamps, logs, images and serialized bytes may
change when rerunning even when numeric outputs agree.

After a full rerun, compare its numeric outputs directly:

```bash
python compare_results.py runs/reproduction_1009 --report runs/comparison.json
```

PNG prefixes identify program outputs. Revised manuscript Fig. 5 uses
`primary/10_trunk_comparison.png`; Fig. 10 uses
`primary/08_validation_scope_comparison.png`. Origin typesetting and submission
documents are maintained separately from this computational release.

## Current results

Primary campaign-balanced WB-PIML scores on `log10(N)`:

| Metric | Estimate |
| --- | ---: |
| Fracture RMSE | 1.0461412097 |
| Fracture MAE | 0.7944498935 |
| Fracture R² | 0.3079423971 |
| Fraction within a factor of five | 0.5476498155 |
| Auxiliary C* | 0.4043395633 |
| Full censored NLL | 1.5305880910 |

Campaign-bootstrap 95% intervals: RMSE [0.8295786902, 1.2377426939],
R² [0.0188805985, 0.4750503260]. These intervals condition on saved predictions
and do not include repeated selection uncertainty.

All five primary outer folds have water training records and estimate its factor
at 1. All 224 synthetic same-condition water/air comparisons have zero reversals
and equal predicted lives in this supplied run. Boundary estimates do not establish
equal physical damage: water records come from two campaigns and remain confounded
with temperature/material support.

Strict factor holdouts contain 30 tasks: 17 reused already-disjoint tasks,
11 refitted, two not estimable, zero failed fits. The 34 sensitivity variants
retain selected primary residual settings and do not select a replacement model.
All 66 runout stopping-rule evidence entries remain unresolved. No new independent
experiments were evaluated; predicted intervals are not engineering design-life bounds.

## Frozen prediction and later evaluation

Fill copies of the templates in `reference_results/prospective/`, then run:

```bash
python run.py --predict new_conditions.csv --model reference_results/prospective/frozen_model.joblib --out runs/prospective_prediction
python run.py --evaluate new_outcomes.csv --predictions runs/prospective_prediction/predictions.csv --out runs/prospective_evaluation
```

Future responses stay out of fitting. Consult the supplied experiment guide for
fields. Load the supplied hash-verified joblib artifact or a model you trust.

## 中文使用说明

本版对应2026年10月9日IJF修改稿，采用10月8日环境共享残差模型的完整结果。
数据集为224条记录（158条断裂、66条runout）。十二个短源码文件按原顺序执行，
公式、候选池、随机种子和验证参数保持一致；原工作脚本不作为单独文件发布。

先运行 `python run.py --check --jobs 2`，再运行 `python verify_release.py`。
完整重跑可用 `python run.py --jobs 16 --out runs/reproduction_1009`，目录须为空。
全部290个原运行文件保留，大型严格校准表采用无损gzip压缩。
`source_equivalence.json`、`verification_report.json` 和
`full_reproduction_report.json` 分别记录代码、归档结果与完整重跑的核对证据。
本次同条件水氧/空气预测无逆转；预测相等不能解释为实测环境损伤相同。

## License

MIT; see `LICENSE`.
