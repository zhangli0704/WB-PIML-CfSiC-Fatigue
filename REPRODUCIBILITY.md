# Reproducibility

The input workbook and the three files in `reference_results/` were copied without
modification. Their SHA-256 checksums are:

| File | SHA-256 |
| --- | --- |
| `data_v23.xlsx` | `e7cb77a6a2b655a8bf263ca4e937368e18d984fb092d3741ba39664bd5496094` |
| `reference_results/PIML_v23_final_results.xlsx` | `726f722a572d2b92d6441fc9737d5166bace76127cbbcf1c9bf7a58070c6bb4b` |
| `reference_results/protocol.json` | `93c7714f2796ec6fbe81bf4d50be3b216f1155b87da3e1248466cb876685d018` |
| `reference_results/README_results.md` | `92146a6e4b5dc811c3e9bb3d441504473534779ecd84f3a26f9acf8f2fade55b` |

The source used for this reorganization was `PIML_v23_final(8).py`, SHA-256
`a116ad07d907f83d8cc1e8445cb5da32fb8e1c6d0fa0532ce8d7c53657eb9a2e`.

The reorganization retains the numerical expressions, candidate order, seeds,
training weights, validation partitions, optimization settings, and table export
order. Figure/PPT generation and its dependencies have been removed. The original
descriptions of those outputs remain in the result metadata for compatibility.

An AST comparison against the source passed for 77 unchanged top-level
functions/classes and all 37 constants/state initializers. The comparison expands
the extracted `run_primary_folds()` helper and checks its arguments, initialization,
loop body, and returned values. Only documentation, progress output, package paths,
module-qualified audit state, and the removed presentation code are normalized.
All modules import successfully; the loaded 222-row input DataFrame also matches
the original loader exactly, including dtypes and row/column order.

Five metric/bootstrap functions share campaign preparation and aggregation helpers.
Each bootstrap samples the same indices in the same order and concatenates the
same complete row-level arrays before applying the original numerical reductions.
Only repeated, read-only preparation is cached within a single function call;
neither model fits nor reference results are cached. Independent checks against
the original functions also cover repeated campaigns, duplicate record IDs,
unequal group sizes, missing interval values, and constant target values.
The four bootstrap functions also passed strict DataFrame comparisons with 100
draws each, and `sb_metrics` passed eight additional group/target cases. Results
and the tested source hashes are recorded in `checks/bootstrap_regression.json`
and `checks/refactor_audit.json`.

The reference workbook contains 45 worksheets, 204,721 cells, and 73,299 numeric
cells. It contains no formulas or date-valued cells. The comparison tool checks
every sheet, including audit, sensitivity, interval, and selection tables.

Exact archival preservation and exact retraining are different checks. The original
implementation uses `n_jobs=-1` for tree models. In the original environment, two
repeated fits of its first-fold 400-tree ExtraTrees model produced differences of
up to `1.7763568394002505e-15` in log-life predictions. The comparison tool therefore
reports strict equality separately from any explicitly requested tolerance. It
does not round, replace, or copy reference values into recomputed results.

## Complete recomputation check

A complete run with 400 trees, 160 sensitivity-audit trees, 10 repeats, and 5,000
bootstrap draws finished successfully in 3,643 seconds. All 45 worksheets were
compared against the unchanged reference workbook.

- Sheet order, dimensions, cell types, text, boolean values, protocol JSON, and
  results README: exactly equal.
- Of 73,299 numeric cells, 70,315 are exactly equal and 2,984 differ.
- 535 differences exceed the default `1e-12` comparison tolerance. They occur
  only in AFT scale, likelihood, and the derived censoring-margin fields.
- Maximum absolute difference: `6.390894413677017e-08` in
  `External_Selection!AP80` (`inner_runout_survival_NLL`).
- Point predictions differ by at most `4.440892098500626e-15`; 80%/90% interval
  endpoints differ by at most `2.6645352591003757e-15`. Primary and external
  performance metrics and their confidence intervals agree within `1e-12`.
- All 5,602 checked selection-setting/index cells and all 1,017 checked selection
  boolean flags are exactly equal.
- All 26,788 boolean cells and the nine physical-coefficient columns in
  `Physical_Parameters` are exactly equal. `Empirical_PI_Summary` is also exactly equal.

This run does **not** pass strict numeric equality or the default `1e-12` check
for the entire workbook. The exact published data remain in `reference_results/`.
See `checks/full_results.json` and `checks/numerical_differences.json` for the
complete comparison. No tolerance was enlarged to label this run a strict pass.

## Local numerical sensitivity check

The original source was also used to rebuild the first outer fold's ExtraTrees
out-of-fold predictions three times with the same selected configuration, four
inner folds, seeds, 300 trees, and `n_jobs=-1`. Predictions differed by at most
`5.773159728050814e-15`, while the original AFT scale calibration produced scale
differences of about `2.13e-8` and runout NLL differences of about `1.66e-8`.
The joint NLL differed by only about `9.73e-14`.

A separate, explicitly synthetic experiment applied approximately `1e-15`
perturbations to the same predictions and produced scale and runout NLL changes
of up to about `9.40e-8` and `7.31e-8`, respectively. These checks show that the
observed full-run differences are compatible with the original optimizer's
numerical sensitivity; they do not establish the unique cause of every difference.
Neither check is a complete rerun of the original pipeline. See
`checks/aft_numerical_sensitivity.json` for inputs and both experiments.
