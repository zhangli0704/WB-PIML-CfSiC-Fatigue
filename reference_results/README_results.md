# WB-PIML results

## Analysis

- Data: `data.xlsx` / `Plot_Data_Verified` only: 158 exact fractures and 65 right-censored runouts.
- All 223 retained records have unit likelihood weight. Exact fractures contribute density terms; runouts contribute survival terms and never enter point-residual regression or point-life metrics as exact failures.
- Labels, evidence fields and campaign IDs are read from the workbook. The retained table has 223 distinct equivalence IDs. ID uniqueness is a structural check and does not establish independence of every cross-publication experiment; unresolved source provenance remains qualified in the workbook.
- 18 sources belong to 17 reviewed campaigns. Primary validation uses five campaign-disjoint outer folds and campaign-disjoint development folds; no campaign crosses a primary train/test split.
- The five-parameter physical trunk combines Walker--Basquin mechanical damage and time-dependent environmental damage. It is fitted by joint censored AFT likelihood with mechanical-coefficient regularization and a nonnegative penalty on the training mean squared environmental log-life decrement. The environmental-penalty strength is selected within development folds.
- The hybrid candidates retain the physical prediction with unit weight and add an exact-fracture residual. The three residual families are ExtraTrees, ExtraTrees with training-fitted physical features, and Matérn kernel ridge. Matérn uses nu=1.5, length scale=2 and ridge alpha=1, with zero residual prior.
- All candidates share the original observable numeric inputs, coarse architecture, the detailed architecture descriptor, and environment. The physics-feature family adds mechanical predicted life, environmental life decrement and Walker stress, computed from a physical model fitted only on the corresponding development subset. Sources, campaigns and record IDs are not predictive features.
- Every hybrid uses soft temperature support `exp[-(nearest exact-development temperature distance / 250 C)^2]`. Support depends on development covariates only. The same-temperature flag reports availability and does not imply zero residual at unseen temperatures.
- WB-PIML evaluates 54 candidates: three residual families, gamma in {0.75, 0.85, 0.95}, environmental strength in {0, 0.1, 1}, and eta in {0.75, 1}. ML-Strong separately selects among 64 candidates: 18 ExtraTrees, 18 gradient boosting, 18 Matérn, one fixed ML ensemble and nine direct ExtraTrees controls with the same training-fitted physical features. ML-Strong is a selected predictive control, and its full pool includes physics-informed candidates.
- Each pool independently retains candidates with inner campaign-balanced exact RMSE within 0.020 of its own minimum, then minimizes calibrated inner runout survival NLL; RMSE and candidate index resolve ties. All candidates in a pool are evaluated on the same development folds. Total search budgets are 54 and 64, not equal. ET-Walker retains its separate gamma/tree search.
- The WB-CD reference survival comparison is a diagnostic, not a selection filter. No affine output calibration is applied. LOTO is a separate factor-held-out stress test and does not choose the hybrid configuration.
- Repeated record-, series-, source- and campaign-disjoint validation uses a fixed secondary ExtraTrees residual: gamma=0.85, environmental strength=0, eta=0.75, leaf=3, max_features=0.85, and soft support at 250 C. Duplicate/label and S-star sensitivity use the same fixed configuration; these results are distinct from the nested-selected primary model.
- LOSO/LOAO/LOTO/LOEO re-select WB-PIML, ML-Strong, ExtraTrees and ET-Walker using only the remaining records. Factor-held-out splits do not impose campaign separation; actual overlap is reported per holdout. LOAO retains coarse architecture groups while detailed architecture remains a predictive descriptor.
- The 80% and 90% prediction intervals use campaign-balanced empirical absolute errors from development OOF predictions. They are not formal finite-sample conformal or design-life guarantees.

## Results

- Lowest campaign-balanced RMSE among the listed physical baselines, predictive controls and proposed model: **WB-PIML = 1.105**.
- WB-PIML campaign-balanced RMSE: **1.105**.
- Selected hybrid branch RMSE: **1.105**; reported-model difference: **0.000**.
- Physical-trunk weights by fold: **[1.00, 1.00, 1.00, 1.00, 1.00]**; every value is 1.00.
- Paired RMSE gain over ML-Ens: **0.067**, 95% campaign-bootstrap CI **[-0.065, 0.196]**.
- Paired RMSE gain over ExtraTrees: **0.119**, 95% campaign-bootstrap CI **[-0.020, 0.261]**.
- Paired RMSE gain over ET-Walker: **0.108**, 95% campaign-bootstrap CI **[-0.048, 0.259]**.
- Paired RMSE gain over ML-Strong: **0.119**, 95% campaign-bootstrap CI **[-0.052, 0.275]**.

`SB_*` denotes campaign-balanced metrics. `Claim_Gate` reports computed criteria, including comparisons with ML-Strong; a lower point estimate alone does not establish superiority or independent external validation.

## Files

- One results workbook: `WB_PIML_results.xlsx`, including selection traces, fold assignments, source audits and all validation summaries.
- Ten composite PNGs. `WB_PIML_figures.pptx` retains the original 29-slide presentation for figures 01-08; the added probability and trunk comparisons are supplied separately as PNGs 09 and 10.
- `protocol.json` and this README describe the executed method. CSV exports are optional through `--export-csv`.


## Conditional lifetime distributions

Point predictions and their original comparisons are retained. `Prob_Metrics` adds complete censored negative log-likelihood alongside RMSE, MAE and F5. `Prob_Intervals` reports central 80% and 90% distribution intervals, with exact-failure-only coverage and width over all test conditions. `Prob_Predictions` contains the sample-level distributions and quantiles; `Prob_Calibration` records training-only calibration predictions. `Prob_Paired_CI`, `Prob_Campaigns` and `Prob_Method` document the comparison and its definitions.

Ordinary regressors receive a Normal distribution on log10 life with a model-specific scale estimated from development predictions under their selected configuration. Existing primary WB-PIML, physical and selected-control scales and the native AFT distributions are retained. Factor-holdout WB receives a new development OOF scale; factor WB-PIML/ML-Strong use their selected development OOF predictions. Weibull location is not its median; the original point prediction remains unchanged. The six existing factor-holdout models are compared probabilistically, while the two native AFT models are compared in primary validation.

Coverage is observed only for exact failures. Censoring selection prevents treating this as unconditional coverage of latent lifetimes; runout stopping times are evaluated through survival probabilities. The calibrated scale combines specimen scatter and prediction error and does not separate aleatoric from epistemic uncertainty. Probability scores supplement the point results.

`09_probability_comparison.png` summarizes the primary probability comparison. The existing eight figures and their presentation retain their original roles.


## Simple-trunk comparisons

WB-Residual uses the classic Walker-Basquin trunk (18 configurations); Basquin-Residual fixes gamma=0 (6 configurations). Each independently selects from the same three residual families and two gains on the original development campaign folds. The environmental penalty is absent. Basquin-Residual retains R and the other original residual inputs; it only removes Walker correction from the trunk. Both use development OOF censored-scale calibration and empirical interval calibration.

The two controls enter primary validation and all four factor holdouts, with point, probability, and interval results in the existing tables. Trunk_Metrics, Trunk_Paired_CI, Trunk_Empirical_PI and Trunk_Method collect the comparisons; 10_trunk_comparison.png summarizes them. Trunk_External_Candidates retains all added factor-holdout candidate scores. Original fixed-configuration repeated validation is unchanged, so its new-control console entries are blank.

Positive gains in Trunk_Paired_CI favor the model named proposed, including the WB-Residual versus Basquin-Residual comparison. These are revision-stage comparisons on previously studied data. They do not select a new primary model automatically.

## Repository packaging note

The file-count sentence above was corrected for the 0909 package. Numerical results, code, data, protocol, figures and the presentation are copied unchanged. The unchanged analysis source still generates the inherited eight-figure sentence; see the repository README for the complete output inventory.
