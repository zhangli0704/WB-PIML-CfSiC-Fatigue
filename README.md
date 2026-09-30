# WB-PIML: fatigue life of Cf/SiC composites

This is the **30 September 2026 revision release**, synchronized with the completed
`0928_shuiyangxin/runs/full_20260928_160433_681221` analysis and the September 30
revised manuscript. It replaces the September 12 four-fragment release.

The maintained analysis is split into **three source fragments**. Their ordered
byte concatenation reproduces the exact 8,835-line source used for the supplied
run. No equations, candidates, seeds, fitting rules or numerical results were
changed during packaging.

## Files

| File or directory | Purpose |
| --- | --- |
| `wb_piml_parts/part_01_core_validation.py` | Data loading, physical models, validation, metrics and figure/export helpers; original lines 1–2867 |
| `wb_piml_parts/part_02_primary_workflow.py` | Campaign audits, nested selection, primary workflow, probabilistic evaluation and trunk controls; lines 2868–6372 |
| `wb_piml_parts/part_03_revision_diagnostics.py` | Revision overrides, learned water factor, sensitivity diagnostics, strict holdouts, frozen inference and unified entry point; lines 6373–8835 |
| `run.py`, `WB-PIML.py` | Equivalent launchers; `run.py` also exposes `load_analysis()` for inspection |
| `data.xlsx` | Original input workbook: 223 records, 158 fractures and 65 runouts, 18 sources and 17 campaigns |
| `reference_results/primary/` | Main workbook, protocol, ten generated PNGs and presentation |
| `reference_results/reviewer/` | All 34 sensitivity variants, checkpoints, water-factor diagnostics, selection/censoring audits and strict holdouts |
| `reference_results/prospective/` | Frozen model, manifest, selected development OOF predictions and empty future-experiment templates |
| `reference_results/workflow_status.json`, `run.log` | Original completed-run metadata and log |
| `verify_release.py` | Integrity, arithmetic, partition and optional selected-refit checks |
| `verification_report.json` | Recorded packaging verification, including five selected outer-fold refits and the selected frozen-model refit |
| `release_manifest.json` | SHA-256 hashes for release files and all 289 original run artifacts |

All 289 original run artifacts are preserved, including generated notes and
historical local paths: 288 files are unchanged, and the 87 MB strict-calibration
CSV is stored as `reviewer/strict_holdout/calibration.csv.gz` using lossless gzip.
The verifier checks its decompressed SHA-256 and automatically expands it only
in a temporary audit directory. It has no truncated rows or modified values.
Those historical paths record provenance; no original Desktop path is needed
to run this release. The uncompressed original run is approximately 136 MB.

The PNG prefixes are **program output identifiers**, not the revised manuscript
figure numbers. In particular, manuscript Fig. 5 corresponds to
`primary/10_trunk_comparison.png`, while manuscript Fig. 10 corresponds to
`primary/08_validation_scope_comparison.png`. Manuscript typesetting and Origin
project files are not part of this computational release.

## Install and run

Use **Python 3.14.0**, the Python version recorded by the completed Windows run.
`requirements.txt` pins its numerical and export packages. The unchanged source
also contains older Python 3.12 reference-environment notes; the actual executed
versions are in `reference_results/workflow_status.json` and take precedence for
reproducing this supplied run. Different platforms or numerical-library builds
may introduce floating-point differences.

```bash
python -m pip install -r requirements.txt
python run.py --check --jobs 2
python verify_release.py
```

Run the complete workflow into a **new, empty** output directory:

```bash
python run.py --jobs 16 --out runs/reproduction_0930
```

Without `--jobs`, the original workflow selects a worker budget from CPU and
available memory. Without `--out`, the launcher creates `runs/full_TIMESTAMP`.
The complete run includes the revised primary analysis, 34 conditional sensitivity
variants, verification and the final frozen model. Its original settings are
seed 20260824, 400 trees, 160 audit trees, 10 repeated splits and 5,000 bootstrap
draws. These are not exposed as new launcher options. Model fitting can be costly.

The fragments are sequential source sections, **not three independent programs**.
Use either launcher instead of running an individual fragment. The launcher
validates the fragments and reconstructs the original source in the ignored
`.wb_piml_runtime/` cache. This preserves the original `__file__`-based source
checks and multiprocessing behavior. Only default data/output locations are
redirected to this repository. The cache is generated, not a fourth maintained
analysis file. The repository directory must be writable for the cache.

For an existing **revised learned-water** primary result:

```bash
python run.py --reviewer-only --baseline reference_results/primary/WB_PIML_results.xlsx --out runs/reviewer_recheck
```

This reruns supplemental analyses and is more expensive than `verify_release.py`.
Do not supply the old fixed-water September 12 workbook as the revised baseline.

## Reproduction checks

```bash
python verify_release.py --refit-selected --report runs/verification.json
```

The default verifier checks all release hashes, reconstructs the canonical source,
checks the input counts and five campaign-disjoint folds, recomputes primary and
distribution scores from saved predictions, and reruns the original 34-variant
and strict-holdout audits on a temporary copy. It also loads the supplied frozen
model and checks inference on all 223 conditions. It does not train models.

`--refit-selected` additionally refits each of the five selected primary pipelines
and its four inner calibration fits, then refits the selected final model and
compares with the frozen model. Comparisons use `rtol=1e-10, atol=1e-10`.
These checks passed during packaging in the recorded environment. They do **not**
constitute a fresh full nested candidate search or retraining of every sensitivity
variant. Original artifact bytes match the completed run after decompression;
timestamps, paths, logs, image metadata and serialized-file bytes are not promised
to be identical when the complete workflow is rerun.

Canonical source SHA-256:
`e059f66d17042c9c5b6ae60272956602a0d523791d486f1935aac2df6622c4c1`.

Input data SHA-256:
`8047f89eaee10db4bbb7b7919c31105886e59204202f52586133e4b3b18b9935`.

## Revised results and interpretation

Primary campaign-balanced WB-PIML scores, on the `log10(N)` coordinate:

| Metric | Estimate |
| --- | ---: |
| Fracture RMSE | 1.0826975319 |
| Fracture MAE | 0.8638195371 |
| Fracture R² | 0.2587308652 |
| Fraction within a factor of five | 0.4515560655 |
| Auxiliary C* | 0.4827268313 |
| Full censored NLL | 1.5498855565 |

The RMSE 95% campaign-cluster interval is [0.8925552431, 1.2541838451]; the R²
interval is [-0.0635816093, 0.4268388053]. Bootstrap intervals are conditional on
saved predictions and do not include repeated model selection uncertainty.

The revision learns the water/oxygen damage multiplier `10**beta`, with
`beta >= 0`, within each training partition. Four primary outer fits reach the
lower bound 1; the remaining fold has no water training records and uses a
disclosed fixed reference value of 1. Neither case proves equal environmental
damage. The physical trunk ordering is constrained, but the residual-corrected
complete model does not guarantee global water-versus-air monotonicity.

The 34 sensitivity variants retain the selected primary residual configuration
while refitting the specified assumptions and calibration. Their scores are not
used to select a replacement primary model. The strict campaign-separated factor
audit has 30 tasks: 17 already-disjoint tasks reused, 11 refitted, 2 not estimable
and 0 failed fits. Unavailable tasks are explicitly disclosed, not filled with
predictions. Strict holdouts remain analyses of this assembled database.

There are 65 unresolved runout stopping-rule evidence entries in the supplied run;
conditional independent censoring is not established by passing numerical checks.
The final model is prepared for prospective experiments; no new independent
experiments were evaluated in this run. Its intervals are not engineering design
life bounds. Consult the original reviewer audits for these limitations.

## Frozen prediction and later evaluation

Fill copies of the templates in `reference_results/prospective/`, then use:

```bash
python run.py --predict new_conditions.csv --model reference_results/prospective/frozen_model.joblib --out runs/prospective_prediction
python run.py --evaluate new_outcomes.csv --predictions runs/prospective_prediction/predictions.csv --out runs/prospective_evaluation
```

This pathway keeps future experiment responses out of model fitting. The model
file uses joblib serialization; load only the supplied hash-verified artifact or
another artifact you trust. See the accompanying Chinese experiment guide for
required columns and interpretation.

## 中文使用说明

本包按 `0928_shuiyangxin` 已完成结果更新，核心代码严格拆成三个分片；
`run.py` 和 `WB-PIML.py` 是运行入口。请先执行 `python run.py --check --jobs 2`，
再执行 `python verify_release.py`。完整重跑可使用
`python run.py --jobs 16 --out runs/reproduction_0930`，输出目录须为空。

发布核验已重新拟合五折已选模型及其校准折，并核对最终冻结模型；未重新执行
所有候选搜索和全部敏感性训练。原始结果文件没有改数、删行或重新生成。
其中严格校准表以 `.csv.gz` 无损压缩保存，核验时自动解压；解压后的文件
哈希与原运行完全一致，其余 288 个原运行文件保持原样。
三个分片按顺序还原的代码与用户原单文件完全一致。
