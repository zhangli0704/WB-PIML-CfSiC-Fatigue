"""Repeated partitions and nested factor-held-out validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baselines import (
    apply_safe_fusion, external_predictions, fit_censored_comparators_outer,
    nested_select_et_walker, nested_select_extra_trees, select_safe_fusion,
)
from .config import (
    CLASSIC_WB_SPEC, COMPETING_WB_SPEC, DAMAGE_CAP, GAMMA, HYBRID_SPECS, LOG5, PHYSICAL_SPECS,
    PRIMARY_GROUP_COLUMN, PROPOSED_MODEL, REPEATED_VALIDATION_SEEDS, SEED,
)
from .data import grouped_folds, grouped_folds_by, record_folds, require_exact_training
from .metrics import (
    censored_hinge_rmse, empirical_interval_halfwidths, sb_metrics, validation_metrics,
)
from .physics import (
    calibrate_aft_scale, fit_physical, inner_oof, physical_parameter_values, predict_physical,
)
from .residual import (
    exact_temperature_residual_gate, fit_hybrid_feature_model, nested_select_hybrid,
    summarize_temperature_gate,
)


def run_primary_folds(df: pd.DataFrame, outer_splits, trees: int):
    """Fit the primary models in the fixed campaign-disjoint outer folds."""
    prediction_parts, parameter_rows, audit_rows = [], [], []
    nested_selection_rows, nested_candidate_parts = [], []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_splits, 1):
        print(f"Primary outer fold {outer_fold}/5", flush=True)
        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        train_sources, test_sources = set(train["source_id"]), set(test["source_id"])
        train_campaigns, test_campaigns = set(train["campaign_id"]), set(test["campaign_id"])
        if train_campaigns & test_campaigns:
            raise AssertionError("Outer campaign leakage")
        require_exact_training(train, f"outer fold {outer_fold} development")
        require_exact_training(test, f"outer fold {outer_fold} joint-likelihood test")
        audit_rows.append({"outer_fold": outer_fold, "n_train": len(train), "n_test": len(test),
                           "n_train_exact": int(train["is_exact"].sum()), "n_test_exact": int(test["is_exact"].sum()),
                            "n_train_runout": int(train["is_runout"].sum()),
                            "n_test_runout": int(test["is_runout"].sum()),
                            "train_sources": "|".join(sorted(train_sources)), "test_sources": "|".join(sorted(test_sources)),
                            "train_campaigns": "|".join(sorted(train_campaigns)),
                            "test_campaigns": "|".join(sorted(test_campaigns)),
                            "source_overlap": bool(train_sources & test_sources),
                            "campaign_overlap": False})

        selection_seed = SEED + 10000 * outer_fold + 500
        hybrid_choice, physical_oof, hybrid_candidates = nested_select_hybrid(
            train, selection_seed, trees
        )
        walker_choice, walker_candidates = nested_select_et_walker(
            train, selection_seed, trees
        )
        et_choice, et_oof, et_candidates = nested_select_extra_trees(
            train, selection_seed, trees
        )
        fusion_choice, proposed_oof, fusion_candidates = select_safe_fusion(
            physical_oof, et_oof
        )
        nested_selection_rows.extend([
            {"outer_fold": outer_fold, "model": "WB-PIML-Anchor", **hybrid_choice},
            {"outer_fold": outer_fold, "model": "WB-PIML", **fusion_choice},
            {"outer_fold": outer_fold, "model": "ET-Walker", **walker_choice},
            {"outer_fold": outer_fold, "model": "ExtraTrees", **et_choice},
        ])
        hybrid_candidates.insert(0, "model", "WB-PIML-Anchor")
        hybrid_candidates.insert(0, "outer_fold", outer_fold)
        fusion_candidates.insert(0, "model", "WB-PIML")
        fusion_candidates.insert(0, "outer_fold", outer_fold)
        et_candidates.insert(0, "model", "ExtraTrees")
        et_candidates.insert(0, "outer_fold", outer_fold)
        walker_candidates.insert(0, "model", "ET-Walker")
        walker_candidates.insert(0, "outer_fold", outer_fold)
        nested_candidate_parts.extend([
            hybrid_candidates, fusion_candidates, walker_candidates, et_candidates,
        ])

        for spec_index, physical_spec in enumerate(PHYSICAL_SPECS):
            oof = inner_oof(train, physical_spec, SEED + 10000 * outer_fold + 100 * spec_index)
            scale = calibrate_aft_scale(oof)
            coef = fit_physical(train, physical_spec)
            mu = predict_physical(test, physical_spec, coef)
            half80, half90 = empirical_interval_halfwidths(oof)
            part = test[["row_id", "source_id", "campaign_id", "record_equivalence_id",
                         "cross_source_duplicate_candidate", "label_conflict_flag",
                         "logN", "is_exact", "is_runout", "outer_fold"]].copy()
            part["model"] = physical_spec.key
            part["pred_logN"] = mu
            part["sigma"] = scale["sigma"]
            for level, halfwidth in [(80, half80), (90, half90)]:
                part[f"lower{level}"] = mu - halfwidth
                part[f"upper{level}"] = mu + halfwidth
            prediction_parts.append(part)
            row = {"outer_fold": outer_fold, "model": physical_spec.key, "gamma": GAMMA,
                   "damage_cap": DAMAGE_CAP, "n_location_parameters": len(coef),
                   "n_physics_parameters": len(coef), "learner": "none",
                   "eta": 0.0, "active_runout_count": np.nan,
                   "active_runout_rate": np.nan, "residual_correction_rms": 0.0,
                   "empirical_halfwidth_80": half80, "empirical_halfwidth_90": half90,
                   **physical_parameter_values(physical_spec, coef), **scale}
            parameter_rows.append(row)

        physical_outer_prediction = None
        physical_outer_coef = None
        physical_outer_meta = None
        for hybrid_index, (hybrid_name, trunk_spec) in enumerate(HYBRID_SPECS):
            outer_seed = SEED + 10000 * outer_fold
            gamma = hybrid_choice["gamma"]
            min_leaf = hybrid_choice["min_samples_leaf"]
            max_features = hybrid_choice["max_features"]
            eta = hybrid_choice["eta"]
            residual_mode = str(hybrid_choice["residual_mode"])
            ridge_alpha = float(hybrid_choice["ridge_alpha"])
            temperature_scale = float(hybrid_choice["temperature_scale"])
            mu, coef, residual_meta = fit_hybrid_feature_model(
                train, test, trunk_spec, outer_seed + 1, trees,
                gamma, min_leaf, max_features, eta,
                residual_mode, ridge_alpha, temperature_scale,
            )
            scale = {"sigma": np.nan, "calibration_objective": np.nan,
                     "exact_density_NLL": np.nan, "runout_survival_NLL": np.nan}
            half80 = half90 = np.nan
            part = test[["row_id", "source_id", "campaign_id", "record_equivalence_id",
                         "cross_source_duplicate_candidate", "label_conflict_flag",
                         "logN", "is_exact", "is_runout", "outer_fold"]].copy()
            part["model"] = hybrid_name
            part["pred_logN"] = mu
            part["sigma"] = scale["sigma"]
            for level, halfwidth in [(80, half80), (90, half90)]:
                part[f"lower{level}"] = mu - halfwidth if np.isfinite(halfwidth) else np.nan
                part[f"upper{level}"] = mu + halfwidth if np.isfinite(halfwidth) else np.nan
            prediction_parts.append(part)
            row = {
                "outer_fold": outer_fold, "model": hybrid_name, "gamma": gamma,
                "damage_cap": DAMAGE_CAP, "n_location_parameters": np.nan,
                "n_physics_parameters": len(coef),
                "learner": (
                    f"{residual_mode} residual ({trees} trees, min_leaf={min_leaf}, "
                    f"max_features={max_features}, ridge_alpha={ridge_alpha}, "
                    f"temperature_scale={temperature_scale})"
                ),
                "eta": eta,
                "residual_mode": residual_mode,
                "ridge_alpha": ridge_alpha,
                "temperature_scale": temperature_scale,
                "active_runout_count": residual_meta["active_runout_count"],
                "active_runout_rate": residual_meta["active_runout_rate"],
                "residual_correction_rms": residual_meta["residual_correction_rms"],
                "exact_temperature_support_rate": residual_meta["exact_temperature_support_rate"],
                "n_exact_temperature_unsupported": residual_meta["n_exact_temperature_unsupported"],
                "all_test_temperature_support_rate": residual_meta["all_test_temperature_support_rate"],
                "n_all_test_temperature_unsupported": residual_meta["n_all_test_temperature_unsupported"],
                "n_residual_temperature_unsupported": residual_meta["n_residual_temperature_unsupported"],
                "empirical_halfwidth_80": half80, "empirical_halfwidth_90": half90,
                **physical_parameter_values(trunk_spec, coef),
                **scale,
            }
            parameter_rows.append(row)
            if hybrid_name == "WB-PIML-Anchor":
                physical_outer_prediction = np.asarray(mu, float)
                physical_outer_coef = np.asarray(coef, float)
                physical_outer_meta = residual_meta

        external = external_predictions(
            train, test, SEED + 10000 * outer_fold, trees,
            et_choice["min_samples_leaf"], et_choice["max_features"],
            walker_choice["gamma"], walker_choice["min_samples_leaf"],
            walker_choice["max_features"],
        )
        censored_external, censored_selections, censored_candidate_tables = (
            fit_censored_comparators_outer(train, test, selection_seed)
        )
        for selection in censored_selections:
            nested_selection_rows.append({"outer_fold": outer_fold, **selection})
        for table in censored_candidate_tables:
            table.insert(0, "outer_fold", outer_fold)
            nested_candidate_parts.append(table)

        if physical_outer_prediction is None or physical_outer_coef is None or physical_outer_meta is None:
            raise AssertionError("The selected physical branch was not fitted")
        proposed_mu = apply_safe_fusion(
            external["ExtraTrees"], physical_outer_prediction,
            float(fusion_choice["physics_mix"]),
            float(fusion_choice["calibration_intercept"]),
            float(fusion_choice["calibration_slope"]),
        )
        proposed_scale = calibrate_aft_scale(proposed_oof)
        proposed_half80, proposed_half90 = empirical_interval_halfwidths(proposed_oof)
        proposed_part = test[[
            "row_id", "source_id", "campaign_id", "record_equivalence_id",
            "cross_source_duplicate_candidate", "label_conflict_flag",
            "logN", "is_exact", "is_runout", "outer_fold",
        ]].copy()
        proposed_part["model"] = PROPOSED_MODEL
        proposed_part["pred_logN"] = proposed_mu
        proposed_part["sigma"] = proposed_scale["sigma"]
        proposed_part["residual_temperature_supported"] = (
            exact_temperature_residual_gate(train, test).astype(bool)
        )
        for level, halfwidth in [(80, proposed_half80), (90, proposed_half90)]:
            proposed_part[f"lower{level}"] = proposed_mu - halfwidth
            proposed_part[f"upper{level}"] = proposed_mu + halfwidth
        prediction_parts.append(proposed_part)
        parameter_rows.append({
            "outer_fold": outer_fold,
            "model": PROPOSED_MODEL,
            "gamma": hybrid_choice["gamma"],
            "damage_cap": DAMAGE_CAP,
            "n_location_parameters": np.nan,
            "n_physics_parameters": len(physical_outer_coef),
            "learner": "competing-damage trunk + OOF-selected v22/robust-support residual",
            "eta": hybrid_choice["eta"],
            "residual_mode": hybrid_choice["residual_mode"],
            "ridge_alpha": hybrid_choice["ridge_alpha"],
            "temperature_scale": hybrid_choice["temperature_scale"],
            "physics_mix": fusion_choice["physics_mix"],
            "calibration_mode": fusion_choice["calibration_mode"],
            "calibration_intercept": fusion_choice["calibration_intercept"],
            "calibration_slope": fusion_choice["calibration_slope"],
            "active_runout_count": physical_outer_meta["active_runout_count"],
            "active_runout_rate": physical_outer_meta["active_runout_rate"],
            "residual_correction_rms": physical_outer_meta["residual_correction_rms"],
            "exact_temperature_support_rate": physical_outer_meta["exact_temperature_support_rate"],
            "n_exact_temperature_unsupported": physical_outer_meta["n_exact_temperature_unsupported"],
            "all_test_temperature_support_rate": physical_outer_meta["all_test_temperature_support_rate"],
            "n_all_test_temperature_unsupported": physical_outer_meta["n_all_test_temperature_unsupported"],
            "n_residual_temperature_unsupported": physical_outer_meta["n_residual_temperature_unsupported"],
            "empirical_halfwidth_80": proposed_half80,
            "empirical_halfwidth_90": proposed_half90,
            **physical_parameter_values(COMPETING_WB_SPEC, physical_outer_coef),
            **proposed_scale,
        })
        for model, mu in external.items():
            part = test[["row_id", "source_id", "campaign_id", "record_equivalence_id",
                         "cross_source_duplicate_candidate", "label_conflict_flag",
                         "logN", "is_exact", "is_runout", "outer_fold"]].copy()
            part["model"] = model
            part["pred_logN"] = mu
            part["sigma"] = np.nan
            for level in [80, 90]:
                part[f"lower{level}"] = np.nan
                part[f"upper{level}"] = np.nan
            prediction_parts.append(part)
        for model, result in censored_external.items():
            mu = np.asarray(result["mu"], float)
            sigma = float(result["sigma"])
            part = test[["row_id", "source_id", "campaign_id", "record_equivalence_id",
                         "cross_source_duplicate_candidate", "label_conflict_flag",
                         "logN", "is_exact", "is_runout", "outer_fold"]].copy()
            part["model"] = model
            part["pred_logN"] = mu
            part["sigma"] = sigma
            for level in [80, 90]:
                part[f"lower{level}"] = np.nan
                part[f"upper{level}"] = np.nan
            prediction_parts.append(part)
            parameter_rows.append({
                "outer_fold": outer_fold, "model": model,
                "learner": f"{result['family']} observable-covariate right-censored AFT",
                "aft_family": result["family"], "aft_alpha": result["alpha"],
                "sigma": sigma, "n_physics_parameters": 0,
                "active_runout_count": int(train["is_runout"].sum()),
                "active_runout_rate": float(train["is_runout"].mean()),
            })

    return (
        prediction_parts, parameter_rows, audit_rows,
        nested_selection_rows, nested_candidate_parts,
    )


def fixed_safe_prediction(train: pd.DataFrame, test: pd.DataFrame, seed: int,
                          trees: int, residual_mode: str = "legacy_et"
                          ) -> tuple[np.ndarray, dict[str, float | str]]:
    """Apply the frozen v23 competing-damage residual in secondary validation."""
    inner_splits, _ = grouped_folds(
        train, min(4, train[PRIMARY_GROUP_COLUMN].nunique()), seed + 333
    )
    physical_parts = []
    inner_trees = min(trees, 120)
    for inner_fold, (fit_idx, valid_idx) in enumerate(inner_splits, 1):
        fit = train.iloc[fit_idx].reset_index(drop=True)
        valid = train.iloc[valid_idx].reset_index(drop=True)
        physical, _, _ = fit_hybrid_feature_model(
            fit, valid, COMPETING_WB_SPEC,
            seed + 1000 * inner_fold + 1, inner_trees,
            GAMMA, 3, 0.85, 0.75,
            residual_mode, 3.0, 200.0,
        )
        metadata = valid[[
            "row_id", "source_id", "campaign_id", "record_equivalence_id",
            "logN", "is_exact", "is_runout",
        ]].copy()
        metadata["inner_fold"] = inner_fold
        physical_part = metadata.copy()
        physical_part["mu"] = physical
        physical_parts.append(physical_part)
    selected_oof = pd.concat(physical_parts, ignore_index=True)
    exact_oof = selected_oof.loc[selected_oof["is_exact"]].copy()
    exact_oof["pred_logN"] = exact_oof["mu"]
    inner_rmse = sb_metrics([
        group.reset_index(drop=True)
        for _, group in exact_oof.groupby(PRIMARY_GROUP_COLUMN)
    ])["SB_RMSE"]
    scale = calibrate_aft_scale(selected_oof)
    choice: dict[str, float | str] = {
        "physics_mix": 1.0,
        "calibration_mode": "none",
        "calibration_intercept": 0.0,
        "calibration_slope": 1.0,
        "inner_SB_RMSE": float(inner_rmse),
        "inner_runout_hinge_RMSE": float(censored_hinge_rmse(selected_oof)),
        "inner_joint_censored_NLL": float(scale["joint_censored_NLL"]),
    }
    halfwidth80, halfwidth90 = empirical_interval_halfwidths(selected_oof)
    choice["empirical_halfwidth_80"] = float(halfwidth80)
    choice["empirical_halfwidth_90"] = float(halfwidth90)
    temperature_gate = exact_temperature_residual_gate(train, test)
    choice.update(summarize_temperature_gate(test, temperature_gate))
    physical_test, _, _ = fit_hybrid_feature_model(
        train, test, COMPETING_WB_SPEC, seed + 1, trees,
        GAMMA, 3, 0.85, 0.75,
        residual_mode, 3.0, 200.0,
    )
    return np.asarray(physical_test, float), choice


def fixed_split_predictions(df: pd.DataFrame, splits, seed: int, trees: int,
                            strategy: str, repeat: int) -> pd.DataFrame:
    pieces = []
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        fold_seed = seed + 10000 * fold
        residual_mode = "legacy_et"
        if strategy == "LOTO" and exact_temperature_residual_gate(train, test).any():
            raise AssertionError("LOTO rows must have no same-temperature exact support")
        hybrid, hybrid_choice = fixed_safe_prediction(
            train, test, fold_seed + 1, trees, residual_mode=residual_mode
        )
        external = external_predictions(train, test, fold_seed, trees, 3, 0.85)
        coef = fit_physical(train, CLASSIC_WB_SPEC, GAMMA)
        physical = predict_physical(test, CLASSIC_WB_SPEC, coef, GAMMA)
        model_predictions = {"WB-PIML": hybrid, "WB": physical,
                             "ExtraTrees": external["ExtraTrees"],
                             "ET-Walker": external["ET-Walker"],
                             "ML-Ens": external["ML-Ens"]}
        residual_temperature_supported = exact_temperature_residual_gate(
            train, test
        ).astype(bool)
        for model, prediction in model_predictions.items():
            part = test[["row_id", "source_id", "campaign_id", "record_equivalence_id",
                         "series_group_id", "T_C", "logN",
                         "is_exact", "is_runout"]].copy()
            part["strategy"] = strategy
            part["repeat"] = repeat
            part["fold"] = fold
            part["model"] = model
            part["pred_logN"] = prediction
            part["residual_temperature_supported"] = (
                residual_temperature_supported
                if model == PROPOSED_MODEL else np.nan
            )
            for level in (80, 90):
                halfwidth = float(hybrid_choice[f"empirical_halfwidth_{level}"])
                if model == PROPOSED_MODEL:
                    part[f"lower{level}"] = np.asarray(prediction, float) - halfwidth
                    part[f"upper{level}"] = np.asarray(prediction, float) + halfwidth
                else:
                    part[f"lower{level}"] = np.nan
                    part[f"upper{level}"] = np.nan
            pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def nested_external_split_predictions(
    df: pd.DataFrame,
    splits,
    labels,
    holdout_column: str,
    seed: int,
    trees: int,
    strategy: str,
    repeat: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit each external holdout after training-only nested model selection."""
    if len(splits) != len(labels):
        raise ValueError("External holdout splits and labels must have equal length")
    if holdout_column not in df.columns:
        raise KeyError(f"Unknown external holdout column: {holdout_column}")

    pieces: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    expected_models = {"WB-PIML", "WB", "ExtraTrees", "ET-Walker", "ML-Ens"}

    for fold, ((train_idx, test_idx), holdout_group) in enumerate(
        zip(splits, labels), 1
    ):
        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        if set(train["row_id"]) & set(test["row_id"]):
            raise AssertionError(f"{strategy} fold {fold}: train/test row overlap")
        holdout_text = str(holdout_group)
        train_level = train[holdout_column].astype(str)
        test_level = test[holdout_column].astype(str)
        if train_level.eq(holdout_text).any() or not test_level.eq(holdout_text).all():
            raise AssertionError(
                f"{strategy} fold {fold}: invalid holdout membership for {holdout_text}"
            )
        require_exact_training(train, f"{strategy} fold {fold} development")
        require_exact_training(test, f"{strategy} fold {fold} holdout")

        train_campaigns = set(train[PRIMARY_GROUP_COLUMN].astype(str))
        test_campaigns = set(test[PRIMARY_GROUP_COLUMN].astype(str))
        overlapping_campaigns = train_campaigns & test_campaigns
        fold_seed = seed + 10000 * fold
        selection_seed = fold_seed + 500

        hybrid_choice, physical_oof, hybrid_candidates = nested_select_hybrid(
            train, selection_seed, trees
        )
        et_choice, et_oof, et_candidates = nested_select_extra_trees(
            train, selection_seed, trees
        )
        walker_choice, walker_candidates = nested_select_et_walker(
            train, selection_seed, trees
        )
        fusion_choice, proposed_oof, fusion_candidates = select_safe_fusion(
            physical_oof, et_oof
        )

        hybrid, _, _ = fit_hybrid_feature_model(
            train, test, COMPETING_WB_SPEC, fold_seed + 1, trees,
            float(hybrid_choice["gamma"]),
            int(hybrid_choice["min_samples_leaf"]),
            float(hybrid_choice["max_features"]),
            float(hybrid_choice["eta"]),
            str(hybrid_choice["residual_mode"]),
            float(hybrid_choice["ridge_alpha"]),
            float(hybrid_choice["temperature_scale"]),
        )
        external = external_predictions(
            train, test, fold_seed, trees,
            int(et_choice["min_samples_leaf"]),
            float(et_choice["max_features"]),
            float(walker_choice["gamma"]),
            int(walker_choice["min_samples_leaf"]),
            float(walker_choice["max_features"]),
        )
        proposed = apply_safe_fusion(
            external["ExtraTrees"], hybrid,
            float(fusion_choice["physics_mix"]),
            float(fusion_choice["calibration_intercept"]),
            float(fusion_choice["calibration_slope"]),
        )
        coef = fit_physical(train, CLASSIC_WB_SPEC, GAMMA)
        physical = predict_physical(test, CLASSIC_WB_SPEC, coef, GAMMA)
        halfwidth80, halfwidth90 = empirical_interval_halfwidths(proposed_oof)

        if strategy == "LOTO" and exact_temperature_residual_gate(train, test).any():
            raise AssertionError("LOTO rows must have no same-temperature exact support")
        residual_temperature_supported = exact_temperature_residual_gate(
            train, test
        ).astype(bool)
        model_predictions = {
            "WB-PIML": proposed,
            "WB": physical,
            "ExtraTrees": external["ExtraTrees"],
            "ET-Walker": external["ET-Walker"],
            "ML-Ens": external["ML-Ens"],
        }
        for model, prediction in model_predictions.items():
            part = test[[
                "row_id", "source_id", "campaign_id", "record_equivalence_id",
                "series_group_id", "T_C", "logN", "is_exact", "is_runout",
            ]].copy()
            part["strategy"] = strategy
            part["repeat"] = repeat
            part["fold"] = fold
            part["holdout_group"] = holdout_text
            part["holdout_column"] = holdout_column
            part["model"] = model
            part["pred_logN"] = np.asarray(prediction, float)
            part["residual_temperature_supported"] = (
                residual_temperature_supported
                if model == PROPOSED_MODEL else np.nan
            )
            for level, halfwidth in ((80, halfwidth80), (90, halfwidth90)):
                if model == PROPOSED_MODEL:
                    part[f"lower{level}"] = np.asarray(prediction, float) - halfwidth
                    part[f"upper{level}"] = np.asarray(prediction, float) + halfwidth
                else:
                    part[f"lower{level}"] = np.nan
                    part[f"upper{level}"] = np.nan
            pieces.append(part)

        common: dict[str, object] = {
            "scenario": strategy,
            "holdout_column": holdout_column,
            "holdout_group": holdout_text,
            "fold": fold,
            "repeat": repeat,
            "selection_seed": selection_seed,
            "inner_group": PRIMARY_GROUP_COLUMN,
            "n_inner_folds": min(4, int(train[PRIMARY_GROUP_COLUMN].nunique())),
            "n_train": len(train),
            "n_train_exact": int(train["is_exact"].sum()),
            "n_train_runout": int(train["is_runout"].sum()),
            "n_train_campaigns": int(train[PRIMARY_GROUP_COLUMN].nunique()),
            "n_test": len(test),
            "n_test_exact": int(test["is_exact"].sum()),
            "n_test_runout": int(test["is_runout"].sum()),
            "validation_scope": "domain-held-out stress test",
            "campaign_disjoint_by_design": False,
            "campaign_overlap": bool(overlapping_campaigns),
            "holdout_campaign_disjoint": bool(not overlapping_campaigns),
            "n_overlapping_campaigns": len(overlapping_campaigns),
            "selected_in_holdout_training_only": True,
            "outer_test_used_for_selection": False,
        }
        fusion_audit = {
            f"fusion_{key}": value for key, value in fusion_choice.items()
        }
        selection_rows.extend([
            {
                **common,
                "model": PROPOSED_MODEL,
                "n_candidates_evaluated": len(hybrid_candidates),
                "n_fusion_candidates_evaluated": len(fusion_candidates),
                "empirical_halfwidth_80": float(halfwidth80),
                "empirical_halfwidth_90": float(halfwidth90),
                "n_interval_calibration_exact": int(
                    proposed_oof["is_exact"].sum()
                ),
                "interval_calibration": (
                    "development campaign-disjoint OOF absolute residuals"
                ),
                "outer_test_used_for_interval_calibration": False,
                **hybrid_choice,
                **fusion_audit,
            },
            {
                **common,
                "model": "ExtraTrees",
                "n_candidates_evaluated": len(et_candidates),
                **et_choice,
            },
            {
                **common,
                "model": "ET-Walker",
                "n_candidates_evaluated": len(walker_candidates),
                **walker_choice,
            },
        ])

        fold_predictions = pd.concat(
            pieces[-len(expected_models):], ignore_index=True
        )
        for model in expected_models:
            model_rows = fold_predictions.loc[fold_predictions["model"].eq(model)]
            if len(model_rows) != len(test) or model_rows["row_id"].duplicated().any():
                raise AssertionError(
                    f"{strategy} fold {fold}: {model} must predict each holdout row once"
                )

    return (
        pd.concat(pieces, ignore_index=True),
        pd.DataFrame(selection_rows),
    )


def repeated_validation(df: pd.DataFrame, trees: int, repeats: int
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    assignment_rows = []
    for strategy in ["record_interpolation", "series_disjoint", "source_disjoint", "campaign_disjoint"]:
        print(f"Repeated validation: {strategy}", flush=True)
        for repeat, seed in enumerate(REPEATED_VALIDATION_SEEDS[:repeats], 1):
            if strategy == "record_interpolation":
                splits, fold_id = record_folds(df, 5, seed)
                split_group = "row_id"
            elif strategy == "series_disjoint":
                splits, fold_id = grouped_folds_by(df, "series_group_id", 5, seed)
                split_group = "series_group_id"
            elif strategy == "source_disjoint":
                splits, fold_id = grouped_folds_by(df, "source_id", 5, seed)
                split_group = "source_id"
            else:
                splits, fold_id = grouped_folds_by(df, "campaign_id", 5, seed)
                split_group = "campaign_id"
            predictions.append(fixed_split_predictions(
                df, splits, seed, min(trees, 300), strategy, repeat
            ))
            assignment = df[["row_id", "source_id", "campaign_id",
                             "record_equivalence_id", "series_group_id"]].copy()
            assignment["strategy"] = strategy
            assignment["repeat"] = repeat
            assignment["fold"] = fold_id
            assignment["split_group"] = split_group
            assignment_rows.append(assignment)
    prediction_table = pd.concat(predictions, ignore_index=True)
    metrics = validation_metrics(prediction_table)
    summary = metrics.groupby(["strategy", "model"]).agg(
        repeats=("repeat", "nunique"),
        SB_RMSE_mean=("SB_RMSE", "mean"), SB_RMSE_sd=("SB_RMSE", "std"),
        SB_R2_mean=("SB_R2", "mean"), SB_R2_sd=("SB_R2", "std"),
        SB_F5_mean=("SB_F5", "mean"), SB_F5_sd=("SB_F5", "std"),
        SB_C_star_mean=("SB_C_star", "mean"), SB_C_star_sd=("SB_C_star", "std"),
    ).reset_index().sort_values(["strategy", "SB_RMSE_mean"])
    return metrics, summary, pd.concat(assignment_rows, ignore_index=True)


def categorical_holdout_splits(df: pd.DataFrame, column: str):
    splits = []
    labels = []
    for value in sorted(df[column].astype(str).unique()):
        valid = np.flatnonzero(df[column].astype(str).eq(value).to_numpy())
        train = np.flatnonzero(~df[column].astype(str).eq(value).to_numpy())
        if len(train) and len(valid) and df.iloc[train]["is_exact"].any() and df.iloc[valid]["is_exact"].any():
            splits.append((train, valid))
            labels.append(value)
    return splits, labels


def external_holdout_validation(
    df: pd.DataFrame, trees: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = []
    source_splits = []
    source_labels = []
    for value in sorted(df["source_id"].unique()):
        valid = np.flatnonzero(df["source_id"].eq(value).to_numpy())
        train = np.flatnonzero(~df["source_id"].eq(value).to_numpy())
        if df.iloc[train]["is_exact"].any() and df.iloc[valid]["is_exact"].any():
            source_splits.append((train, valid))
            source_labels.append(value)
    scenarios.append(("LOSO", "source_id", source_splits, source_labels))
    for name, column in [("LOAO", "architecture"), ("LOTO", "temp_bin"), ("LOEO", "env_class")]:
        splits, labels = categorical_holdout_splits(df, column)
        scenarios.append((name, column, splits, labels))

    prediction_parts = []
    selection_parts = []
    group_rows = []
    for scenario_index, (scenario, column, splits, labels) in enumerate(scenarios):
        print(f"Nested factor holdouts: {scenario}", flush=True)
        pred, selection = nested_external_split_predictions(
            df, splits, labels, column,
            SEED + 700000 + 1000000 * scenario_index,
            min(trees, 400), scenario, 1,
        )
        prediction_parts.append(pred)
        selection_parts.append(selection)
        expected_row_ids = set(
            df.iloc[np.concatenate([test_idx for _, test_idx in splits])]["row_id"]
        )
        for model in ["WB-PIML", "WB", "ExtraTrees", "ET-Walker", "ML-Ens"]:
            model_rows = pred.loc[pred["model"].eq(model)]
            if (
                set(model_rows["row_id"]) != expected_row_ids
                or model_rows["row_id"].duplicated().any()
            ):
                raise AssertionError(
                    f"{scenario}: {model} must predict each eligible holdout row once"
                )
        for (holdout_group, model), group in pred.loc[pred["is_exact"]].groupby(
            ["holdout_group", "model"]
        ):
            error = group["pred_logN"].to_numpy(float) - group["logN"].to_numpy(float)
            group_rows.append({"scenario": scenario, "holdout_group": holdout_group,
                               "model": model, "n_exact": len(group),
                               "n_sources": group["source_id"].nunique(),
                               "RMSE": float(np.sqrt(np.mean(error ** 2))),
                               "MAE": float(np.mean(np.abs(error))),
                               "F5": float(np.mean(np.abs(error) <= LOG5))})
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions["T_C"] = predictions["row_id"].map(
        df.set_index("row_id")["T_C"]
    ).to_numpy(float)
    summary_rows = []
    for (scenario, model), group in predictions.loc[predictions["is_exact"]].groupby(
        ["strategy", "model"]
    ):
        source_groups = [g.reset_index(drop=True) for _, g in group.groupby(PRIMARY_GROUP_COLUMN)]
        summary_rows.append({"scenario": scenario, "model": model,
                             "n_exact": len(group), "n_campaigns": group["campaign_id"].nunique(),
                             "n_sources": group["source_id"].nunique(),
                             **sb_metrics(source_groups)})
    return (
        pd.DataFrame(summary_rows).sort_values(["scenario", "SB_RMSE"]),
        pd.DataFrame(group_rows),
        predictions,
        pd.concat(selection_parts, ignore_index=True, sort=False),
    )


def sstar_exclusion_sensitivity(df: pd.DataFrame, trees: int) -> pd.DataFrame:
    rows = []
    for label, analysis_df in [("all_verified_exact", df.copy()),
                               ("exclude_Sstar_gt_1", df.loc[df["S"] <= 1.0].reset_index(drop=True))]:
        splits, _ = grouped_folds_by(analysis_df, "campaign_id", 5, SEED)
        pred = fixed_split_predictions(
            analysis_df, splits, SEED + 880000, min(trees, 400), label, 1
        )
        metrics = validation_metrics(pred)
        for _, metric in metrics.iterrows():
            rows.append({"dataset": label, "n_rows": len(analysis_df),
                         "n_exact": int(analysis_df["is_exact"].sum()),
                         "model": metric["model"],
                         "SB_RMSE": metric["SB_RMSE"], "SB_R2": metric["SB_R2"],
                         "SB_F5": metric["SB_F5"], "SB_C_star": metric["SB_C_star"]})
    return pd.DataFrame(rows).sort_values(["model", "dataset"])
