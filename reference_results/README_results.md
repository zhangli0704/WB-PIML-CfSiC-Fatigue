# WB-PIML v23 censor-aware, temperature-robust competing-damage results

## Analysis protocol

- Data: `data_v23.xlsx` / `Plot_Data_Verified` only.
- The verified worksheet contains 160 exact fractures and 62 right-censored runouts; all 222 unique records enter the joint AFT likelihood.
- Every record has `sample_weight=1`; row numbers, sources, campaigns and equivalence IDs never change fitting importance.
- Exact fractures contribute density terms. Runouts contribute survival terms and are never treated as failures at their stopping cycles.
- No code-side label override is applied; workbook labels and campaign IDs are used as reviewed.
- 17 publication sources are assigned to 18 human-reviewed campaigns; no campaign is inferred from row counts alone.
- `Plot_Data_Verified` is physically deduplicated: 222 unique records and no repeated equivalence ID. Duplicate copies remain traceable in `Plot_Data` and `Excluded_Records` but are never read by V23.
- Five fixed campaign-disjoint outer folds; every record appears in one outer test fold and no equivalence cluster crosses folds.
- The physical branch combines Walker--Basquin mechanical damage with a time-dependent environmental rate proportional to `E(env)*Arrhenius(T)/f`.
- The machine-learning branch predicts only `y-mu_WB-CD`; the physical trunk has unit weight in every fold and cannot fall back to a purely data-driven model.
- If the current development set has no exact fracture at the requested temperature, the learned point residual is set to zero and prediction uses the competing-damage physical trunk; this support check never reads validation responses.
- Walker gamma, residual shrinkage eta and tree regularization are selected exclusively inside campaign-disjoint development folds. No affine output calibration is used.
- Exact-only campaign-balanced RMSE remains the point-performance endpoint; joint censored NLL is the within-tolerance selection tie-break.
- ET-Walker retains the v22 joint-data inner-fold budget; v23 adds only four predeclared robust residual candidates to WB-PIML.
- Runouts affect the physical trunk and AFT scale only through `-log P(Nf>N_stop|x)`; point RMSE/MAE/R2/F5/C* remain exact-only.
- There is no neural network, source intercept, arbitrary oxidation threshold, or outer-test tuning.
- Repeated record-, series-, source-, and campaign-disjoint validation separates interpolation from generalization.
- LOSO/LOAO/LOTO/LOEO are factor-held-out stress tests. WB-PIML, ExtraTrees and ET-Walker are re-selected only in the remaining records with campaign-disjoint inner folds. These outer factor splits are not campaign-disjoint by design, and the actual campaign overlap is reported for every holdout.

## Performance summary

- Best pre-specified model in the primary campaign-disjoint analysis: **WB-PIML = 1.023** campaign-balanced RMSE.
- WB-PIML campaign-balanced RMSE: **1.023**.
- Selected competing-damage residual branch RMSE: **1.023**; reported-model consistency difference: **0.000**.
- Fixed physical-trunk weights by fold: **[1.00, 1.00, 1.00, 1.00, 1.00]**; every value must equal 1.00.
- Paired RMSE gain over ML-Ens: **0.140**, 95% campaign-bootstrap CI **[0.060, 0.225]**.
- Paired RMSE gain over plain ExtraTrees: **0.157**, 95% campaign-bootstrap CI **[0.055, 0.269]**.
- Paired RMSE gain over equally tuned ET-Walker: **0.148**, 95% campaign-bootstrap CI **[0.041, 0.261]**.

The `SB_*` metrics are campaign-balanced. The `Claim_Gate` worksheet summarizes the statistical and validation checks.

## Output files

- The default tabular deliverable is one workbook: `PIML_v23_final_results.xlsx`.
- `Outer_Fold_Assignment` is included in that workbook for reproducibility.
- CSV files are disabled by default; pass `--export-csv` only when legacy machine-readable files are explicitly needed.
- The workbook contains the data for the manuscript figures.
- `protocol.json` and this README remain as non-tabular audit outputs.
