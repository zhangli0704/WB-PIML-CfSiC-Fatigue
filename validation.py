"""Nested validation, factor holdouts and sensitivity analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import replace
import json
import math
from models import (
    apply_safe_fusion,
    calibrate_aft_scale,
    exact_temperature_residual_gate,
    external_predictions,
    fit_censored_comparators_outer,
    fit_hybrid_feature_model,
    fit_physical,
    inner_oof,
    nested_select_et_walker,
    nested_select_extra_trees,
    nested_select_hybrid,
    physical_parameter_values,
    predict_physical,
    select_safe_fusion,
    summarize_temperature_gate,
)
from data import (
    ARRHENIUS_REFERENCE_C,
    ARRHENIUS_SCALE,
    CLASSIC_WB_SPEC,
    COMPETING_WB_SPEC,
    DAMAGE_CAP,
    DEFAULT_COMPETING_CONFIG,
    GAMMA,
    HYBRID_SPECS,
    LOG5,
    PHYSICAL_SPECS,
    PRIMARY_GROUP_COLUMN,
    PROPOSED_MODEL,
    REPEATED_VALIDATION_SEEDS,
    SEED,
    apply_physics_drivers,
    censored_hinge_rmse,
    empirical_interval_halfwidths,
    grouped_folds,
    grouped_folds_by,
    parse_bool,
    record_folds,
    require_exact_training,
    sb_metrics,
    validation_metrics,
)
import data


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


def sensitivity_audit(df: pd.DataFrame, outer_splits, trees: int,
                      nested_selection: pd.DataFrame) -> pd.DataFrame:
    """Outer-fold, response-blind sensitivity audit of major V23 assumptions."""
    selected = nested_selection.loc[
        nested_selection["model"].eq("WB-PIML-Anchor")
    ].set_index("outer_fold")
    if set(selected.index.astype(int)) != set(range(1, len(outer_splits) + 1)):
        raise AssertionError("Sensitivity audit requires one selected hybrid per outer fold")

    default = DEFAULT_COMPETING_CONFIG
    variants: list[dict[str, object]] = [
        {"variant": "selected_pipeline"},
        {"variant": "no_runout_likelihood_training", "use_runouts": False},
        {"variant": "no_residual", "eta": 0.0},
        {"variant": "classic_WB_trunk", "trunk_spec": CLASSIC_WB_SPEC},
        {"variant": "no_exact_temperature_gate", "exact_temperature_gate_enabled": False},
        {"variant": "gamma_low", "gamma": 0.75},
        {"variant": "gamma_high", "gamma": 0.95},
        {"variant": "leaf_2", "min_samples_leaf": 2},
        {"variant": "leaf_5", "min_samples_leaf": 5},
        {"variant": "max_features_low", "max_features": 0.65},
        {"variant": "max_features_full", "max_features": 1.00},
        {"variant": "eta_low", "eta": 0.50},
        {"variant": "eta_full", "eta": 1.00},
        {"variant": "water_factor_1_0", "water_factor": 1.0},
        {"variant": "water_factor_2_0", "water_factor": 2.0},
        {"variant": "arrhenius_reference_700C", "arrhenius_reference_c": 700.0},
        {"variant": "arrhenius_reference_900C", "arrhenius_reference_c": 900.0},
        {"variant": "arrhenius_scale_750", "arrhenius_scale": 750.0},
        {"variant": "arrhenius_scale_1250", "arrhenius_scale": 1250.0},
        {"variant": "ridge_low", "competing_config": replace(default, ridge_strength=0.003)},
        {"variant": "ridge_high", "competing_config": replace(default, ridge_strength=0.03)},
        {"variant": "physics_bounds_wide", "competing_config": replace(
            default, intercept_half_width=1.25, walker_half_width=5.0,
            environment_ratio_min=-10.0, environment_ratio_max=6.0,
            environment_stress_max=14.0, arrhenius_max=14.0,
        )},
        {"variant": "relaxed_sign_constraints", "competing_config": replace(
            default, walker_min=-20.0, environment_stress_min=-5.0,
            arrhenius_min=-5.0,
        )},
        {"variant": "dual_support_equal_mix", "residual_mode": "dual_support",
         "robust_tree_weight": 0.50},
        {"variant": "dual_support_tree_heavy", "residual_mode": "dual_support",
         "robust_tree_weight": 0.80},
        {"variant": "unsupported_residual_zero", "residual_mode": "dual_support",
         "unsupported_smooth_shrink": 0.0},
        {"variant": "unsupported_residual_half", "residual_mode": "dual_support",
         "unsupported_smooth_shrink": 0.50},
        {"variant": "unseen_domain_shrink_half", "residual_mode": "dual_support",
         "unseen_arch_factor": 0.50, "unseen_env_factor": 0.50},
        {"variant": "unseen_domain_no_shrink", "residual_mode": "dual_support",
         "unseen_arch_factor": 1.00, "unseen_env_factor": 1.00},
    ]

    rows: list[dict[str, object]] = []
    for variant_index, variant in enumerate(variants):
        name = str(variant["variant"])
        pieces: list[pd.DataFrame] = []
        parameter_records: list[dict[str, float]] = []
        active_rates: list[float] = []
        correction_rms: list[float] = []
        fold_settings: list[str] = []
        failure_reason = ""
        for fold, (train_idx, test_idx) in enumerate(outer_splits, 1):
            base = selected.loc[fold]
            train = df.iloc[train_idx].reset_index(drop=True)
            test = df.iloc[test_idx].reset_index(drop=True)
            water_factor = float(variant.get("water_factor", 1.5))
            arrhenius_reference_c = float(
                variant.get("arrhenius_reference_c", ARRHENIUS_REFERENCE_C)
            )
            arrhenius_scale = float(variant.get("arrhenius_scale", ARRHENIUS_SCALE))
            train = apply_physics_drivers(
                train, water_factor, arrhenius_reference_c, arrhenius_scale
            )
            test = apply_physics_drivers(
                test, water_factor, arrhenius_reference_c, arrhenius_scale
            )
            use_runouts = bool(variant.get("use_runouts", True))
            fit_frame = (
                train if use_runouts
                else train.loc[train["is_exact"]].reset_index(drop=True)
            )
            settings = {
                "gamma": float(variant.get("gamma", base["gamma"])),
                "min_samples_leaf": int(variant.get(
                    "min_samples_leaf", base["min_samples_leaf"]
                )),
                "max_features": float(variant.get(
                    "max_features", base["max_features"]
                )),
                "eta": float(variant.get("eta", base["eta"])),
                "residual_mode": str(variant.get(
                    "residual_mode", base["residual_mode"]
                )),
                "ridge_alpha": float(variant.get(
                    "ridge_alpha", base["ridge_alpha"]
                )),
                "temperature_scale": float(variant.get(
                    "temperature_scale", base["temperature_scale"]
                )),
            }
            fold_settings.append(json.dumps(settings, sort_keys=True))
            trunk_spec = variant.get("trunk_spec", COMPETING_WB_SPEC)
            try:
                prediction, coef, metadata = fit_hybrid_feature_model(
                    fit_frame, test, trunk_spec,
                    SEED + 100000 * variant_index + 10000 * fold + 1,
                    trees, **settings,
                    competing_config=variant.get("competing_config", default),
                    exact_temperature_gate_enabled=bool(
                        variant.get("exact_temperature_gate_enabled", True)
                    ),
                    robust_tree_weight=float(variant.get("robust_tree_weight", 0.65)),
                    unsupported_smooth_shrink=float(
                        variant.get("unsupported_smooth_shrink", 0.25)
                    ),
                    unseen_arch_factor=float(variant.get("unseen_arch_factor", 0.70)),
                    unseen_env_factor=float(variant.get("unseen_env_factor", 0.70)),
                )
            except Exception as exc:  # Audit failures are reported, never hidden.
                failure_reason = f"outer_fold={fold}: {type(exc).__name__}: {exc}"
                break
            part = test.loc[test["is_exact"], [
                "source_id", "campaign_id", "record_equivalence_id", "logN"
            ]].copy()
            part["pred_logN"] = prediction[test["is_exact"].to_numpy(bool)]
            pieces.append(part)
            parameter_records.append(physical_parameter_values(trunk_spec, coef))
            active_rates.append(float(metadata["active_runout_rate"]))
            correction_rms.append(float(metadata["residual_correction_rms"]))

        common = {
            "variant": name,
            "fit_status": "failed" if failure_reason else "completed",
            "failure_reason": failure_reason,
            "outer_folds_completed": len(pieces),
            "audit_trees": trees,
            "likelihood_training": (
                "exact_density_plus_runout_survival"
                if bool(variant.get("use_runouts", True))
                else "exact_density_only_ablation"
            ),
            "water_factor": float(variant.get("water_factor", 1.5)),
            "arrhenius_reference_C": float(
                variant.get("arrhenius_reference_c", ARRHENIUS_REFERENCE_C)
            ),
            "arrhenius_scale": float(variant.get("arrhenius_scale", ARRHENIUS_SCALE)),
            "trunk_model": str(variant.get("trunk_spec", COMPETING_WB_SPEC).key),
            "exact_temperature_gate_enabled": bool(
                variant.get("exact_temperature_gate_enabled", True)
            ),
            "robust_tree_weight": float(variant.get("robust_tree_weight", 0.65)),
            "unsupported_smooth_shrink": float(
                variant.get("unsupported_smooth_shrink", 0.25)
            ),
            "unseen_arch_factor": float(variant.get("unseen_arch_factor", 0.70)),
            "unseen_env_factor": float(variant.get("unseen_env_factor", 0.70)),
            "fold_specific_selected_settings": " | ".join(fold_settings),
            "used_for_selection": False,
        }
        if failure_reason:
            rows.append(common)
            continue
        pooled = pd.concat(pieces, ignore_index=True)
        values = sb_metrics([
            group.reset_index(drop=True)
            for _, group in pooled.groupby(PRIMARY_GROUP_COLUMN)
        ])
        parameters = pd.DataFrame(parameter_records)
        rows.append({
            **common, **values,
            "mean_mechanical_log10_damage_scale": float(
                parameters["mechanical_log10_damage_scale"].mean()
            ),
            "mean_walker_slope": float(parameters["walker_slope"].mean()),
            "mean_environment_log10_damage_ratio": float(
                parameters["environment_log10_damage_ratio"].mean()
            ),
            "mean_environment_stress_exponent": float(
                parameters["environment_stress_exponent"].mean()
            ),
            "mean_arrhenius_temperature_sensitivity": float(
                parameters["arrhenius_temperature_sensitivity"].mean()
            ),
            "mean_active_runout_rate": float(np.mean(active_rates)),
            "mean_residual_correction_rms": float(np.mean(correction_rms)),
        })
    return pd.DataFrame(rows).reset_index(drop=True)


def _raw_duplicate_variant(formal: pd.DataFrame, use_legacy_labels: bool = False
                           ) -> pd.DataFrame:
    raw = data.AUDIT_CONTEXT.get("raw")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise RuntimeError("Plot_Data is unavailable for duplicate sensitivity")
    original = data.AUDIT_CONTEXT.get("original")
    original_by_row = (
        original.set_index("row_id") if use_legacy_labels
        and isinstance(original, pd.DataFrame) and not original.empty else None
    )
    representatives = formal.set_index("record_equivalence_id", drop=False)
    records = []
    for _, raw_row in raw.iterrows():
        equivalence_id = str(raw_row["record_equivalence_id"]).strip()
        if equivalence_id not in representatives.index:
            continue
        item = representatives.loc[equivalence_id]
        if isinstance(item, pd.DataFrame):
            item = item.iloc[0]
        clone_row = item.copy()
        clone_row["row_id"] = int(raw_row["row_id"])
        clone_row["source_id"] = str(raw_row["source_id"])
        for column in ("record_role", "audit_status", "evidence_grade", "duplicate_group_status"):
            if column in raw_row:
                clone_row[column] = raw_row[column]
        clone_row["reviewed_secondary_source"] = "secondary_replot" in str(
            clone_row.get("record_role", "")
        ).lower()
        clone_row["series_group_id"] = "|".join(str(clone_row[c]) for c in [
            "source_id", "architecture_detail", "env_class", "T_C", "R", "UTS_MPa"
        ])
        if original_by_row is not None and int(raw_row["row_id"]) in original_by_row.index:
            legacy = original_by_row.loc[int(raw_row["row_id"])]
            if isinstance(legacy, pd.DataFrame):
                legacy = legacy.iloc[0]
            nf = float(pd.to_numeric(legacy["Nf_cycles"], errors="coerce"))
            if np.isfinite(nf) and nf > 0:
                clone_row["Nf_cycles"] = nf
                clone_row["logN"] = math.log10(nf)
            runout_value = parse_bool(pd.Series([legacy["is_runout"]]), "legacy is_runout").iloc[0]
            clone_row["is_runout"] = bool(runout_value)
            clone_row["is_exact"] = not bool(runout_value)
            clone_row["supplied_is_runout"] = bool(runout_value)
            clone_row["supplied_is_exact_failure"] = not bool(runout_value)
        clone_row["sample_weight"] = 1.0
        clone_row["likelihood_weight"] = 1.0
        clone_row["training_role"] = (
            "right_censored_likelihood_training" if clone_row["is_runout"]
            else "exact_likelihood_training"
        )
        records.append(clone_row)
    result = pd.DataFrame(records).reset_index(drop=True)
    if result["row_id"].duplicated().any():
        raise AssertionError("Raw sensitivity row_id values must remain unique")
    return result


def _fixed_campaign_splits(frame: pd.DataFrame, assignment: dict[str, int]):
    fold_id = frame["campaign_id"].astype(str).map(assignment)
    if fold_id.isna().any():
        missing = sorted(frame.loc[fold_id.isna(), "campaign_id"].astype(str).unique())
        raise ValueError(f"Sensitivity campaign is absent from the fixed map: {missing}")
    fold_id = fold_id.to_numpy(int)
    splits = []
    for fold in sorted(np.unique(fold_id)):
        valid = np.flatnonzero(fold_id == fold)
        train = np.flatnonzero(fold_id != fold)
        if not len(valid) or not frame.iloc[valid]["is_exact"].any() or not frame.iloc[train]["is_exact"].any():
            raise RuntimeError(f"Fixed sensitivity fold {fold} has no exact failures")
        splits.append((train, valid))
    return splits


def duplicate_source_sensitivity(df: pd.DataFrame, trees: int) -> pd.DataFrame:
    """Genuinely different provenance datasets with fixed campaign folds/seeds."""
    formal = df.copy().reset_index(drop=True)
    _, fold_id = grouped_folds_by(formal, "campaign_id", 5, SEED + 910000)
    campaign_assignment = {
        str(campaign): int(fold)
        for campaign, fold in zip(formal["campaign_id"], fold_id)
    }
    raw_equal = _raw_duplicate_variant(formal, use_legacy_labels=False)
    secondary_excluded = formal.loc[
        ~formal["reviewed_secondary_source"].fillna(False)
    ].reset_index(drop=True)
    high_evidence = formal.loc[
        formal["evidence_grade"].astype(str).str.startswith(("A", "B+"))
    ].reset_index(drop=True)
    variants: list[tuple[str, pd.DataFrame, str]] = [
        ("reviewed_deduplicated_sensitivity_reference", formal,
         "reviewed deduplicated data evaluated with a separate fixed campaign split and fixed model settings; this is not the primary nested estimate"),
        ("duplicate_expanded_equal_weight", raw_equal,
         "restores 23 traceability copies to measure duplicate inflation"),
        ("reviewed_excluding_secondary_replots", secondary_excluded,
         "uses workbook record_role; no source name is hard-coded"),
        ("reviewed_high_evidence_only", high_evidence,
         "uses workbook evidence_grade"),
    ]
    try:
        variants.append((
            "duplicate_expanded_original_Nf_and_labels",
            _raw_duplicate_variant(formal, use_legacy_labels=True),
            "duplicate-expanded data; Nf_cycles and exact/runout labels are restored from Plot_Data_Original, while other modelling fields follow the reviewed representative record",
        ))
    except Exception as exc:
        return pd.DataFrame([{
            "dataset": "duplicate_expanded_original_Nf_and_labels", "status": "not_evaluable",
            "reason": str(exc),
        }])
    rows = []
    for label, analysis_df, definition in variants:
        try:
            splits = _fixed_campaign_splits(analysis_df, campaign_assignment)
            pred = fixed_split_predictions(
                analysis_df, splits, SEED + 920000, min(trees, 400), label, 1
            )
            metrics = validation_metrics(pred)
            for _, metric in metrics.iterrows():
                rows.append({
                    "dataset": label, "status": "evaluated", "definition": definition,
                    "same_campaign_fold_map": True, "same_model_seed": True,
                    "n_rows": len(analysis_df), "n_exact": int(analysis_df["is_exact"].sum()),
                    "n_runout": int(analysis_df["is_runout"].sum()),
                    "n_sources": int(analysis_df["source_id"].nunique()),
                    "n_campaigns": int(analysis_df["campaign_id"].nunique()),
                    "model": metric["model"], "SB_RMSE": metric["SB_RMSE"],
                    "SB_R2": metric["SB_R2"], "SB_F5": metric["SB_F5"],
                    "SB_C_star": metric["SB_C_star"],
                })
        except Exception as exc:
            rows.append({
                "dataset": label, "status": "not_evaluable", "definition": definition,
                "same_campaign_fold_map": True, "same_model_seed": True,
                "n_rows": len(analysis_df), "n_exact": int(analysis_df["is_exact"].sum()),
                "n_runout": int(analysis_df["is_runout"].sum()), "reason": str(exc),
            })
    return pd.DataFrame(rows).sort_values(
        ["dataset", "model"], na_position="last"
    ).reset_index(drop=True)
