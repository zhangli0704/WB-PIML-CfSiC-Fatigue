def sensitivity_audit(df: pd.DataFrame, outer_splits, trees: int,
                      nested_selection: pd.DataFrame) -> pd.DataFrame:


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
        {"variant": "no_temperature_support", "exact_temperature_gate_enabled": False},
        {"variant": "gamma_low", "gamma": 0.75},
        {"variant": "gamma_high", "gamma": 0.95},
        {"variant": "leaf_2", "min_samples_leaf": 2},
        {"variant": "leaf_5", "min_samples_leaf": 5},
        {"variant": "max_features_low", "max_features": 0.65},
        {"variant": "max_features_full", "max_features": 1.00},
        {"variant": "eta_low", "eta": 0.50},
        {"variant": "eta_full", "eta": 1.00},
        {"variant": "fixed_water_factor_1_0", "water_factor": 1.0},
        {"variant": "legacy_water_factor_1_5", "water_factor": 1.5},
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
        {"variant": "environment_regularization_zero", "strength": 0.0},
        {"variant": "environment_regularization_low", "strength": 0.1},
        {"variant": "environment_regularization_high", "strength": 1.0},
        {"variant": "residual_ET", "family": "ET_residual"},
        {"variant": "residual_physics_features_ET", "family": "physics_feature_ET_residual"},
        {"variant": "residual_Matern", "family": "Matern_residual"},
    ]

    rows: list[dict[str, object]] = []
    for variant in variants:
        name = str(variant["variant"])
        pieces: list[pd.DataFrame] = []
        parameter_records: list[dict[str, float]] = []
        active_rates: list[float] = []
        correction_rms: list[float] = []
        fold_settings: list[str] = []
        exact_gate_flags: list[bool] = []
        failure_reason = ""
        for fold, (train_idx, test_idx) in enumerate(outer_splits, 1):
            base = selected.loc[fold]
            train = df.iloc[train_idx].reset_index(drop=True)
            test = df.iloc[test_idx].reset_index(drop=True)
            water_factor = variant.get("water_factor", None)
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
                "family": str(variant.get("family", base["family"])),
                "strength": float(variant.get("strength", base["strength"])),
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
                "kernel_length_scale": float(variant.get(
                    "kernel_length_scale", base.get("kernel_length_scale", 2.0)
                )),
                "temperature_scale": float(variant.get(
                    "temperature_scale", base["temperature_scale"]
                )),
                "temperature_policy": str(variant.get("temperature_policy", base["temperature_policy"])),
            }
            fold_settings.append(json.dumps(settings, sort_keys=True))
            trunk_spec = variant.get("trunk_spec", COMPETING_WB_SPEC)
            try:
                prediction, coef, metadata = fit_hybrid_policy(
                    fit_frame, test, trunk_spec,
                    SEED + 10000 * fold + 1,
                    trees, **settings,
                    competing_config=variant.get("competing_config", default),
                    exact_temperature_gate_enabled=bool(
                        variant.get("exact_temperature_gate_enabled", True)
                    ),
                )
            except Exception as exc:
                failure_reason = f"outer_fold={fold}: {type(exc).__name__}: {exc}"
                break
            exact_gate_flags.append(bool(metadata["exact_temperature_gate_enabled"]))
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
            "water_factor": variant.get("water_factor", "learned_training_only"),
            "arrhenius_reference_C": float(
                variant.get("arrhenius_reference_c", ARRHENIUS_REFERENCE_C)
            ),
            "arrhenius_scale": float(variant.get("arrhenius_scale", ARRHENIUS_SCALE)),
            "trunk_model": str(variant.get("trunk_spec", COMPETING_WB_SPEC).key),
            "exact_temperature_gate_enabled": (
                exact_gate_flags[0] if exact_gate_flags and len(set(exact_gate_flags)) == 1
                else "fold_specific"
            ),
            "temperature_policy": str(variant.get("temperature_policy", "fold_specific_selected")),
            "paired_variant_seeds": True,
            "parameter_applicable_folds": (
                int(selected["family"].isin(
                    ["ET_residual", "physics_feature_ET_residual"]
                ).sum())
                if name in ("leaf_2", "leaf_5", "max_features_low", "max_features_full")
                else len(outer_splits)
            ),
            "parameter_total_folds": len(outer_splits),
            "parameter_applicability": (
                "tree residual folds only"
                if name in ("leaf_2", "leaf_5", "max_features_low", "max_features_full")
                else "all outer folds"
            ),
            "parameter_applicability_note": (
                "Leaf size and max_features are inactive for Matern_residual folds. "
                "Reported metrics still include all outer folds."
                if name in ("leaf_2", "leaf_5", "max_features_low", "max_features_full")
                else "The variant applies to each selected outer-fold pipeline."
            ),
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


def component_ablation_summary(sensitivity: pd.DataFrame) -> pd.DataFrame:

    component_variants = [
        "selected_pipeline",
        "classic_WB_trunk",
        "no_residual",
        "no_runout_likelihood_training",
        "no_temperature_support",
        "relaxed_sign_constraints",
        "environment_regularization_zero",
    ]
    result = sensitivity.loc[
        sensitivity["variant"].isin(component_variants)
    ].copy()
    order = {name: index for index, name in enumerate(component_variants)}
    result["component_order"] = result["variant"].map(order)
    reference = result.loc[
        result["variant"].eq("selected_pipeline"), "SB_RMSE"
    ]
    reference_rmse = float(reference.iloc[0]) if len(reference) else np.nan
    result["RMSE_increase_vs_selected"] = result["SB_RMSE"] - reference_rmse
    result["interpretation"] = np.where(
        result["variant"].eq("selected_pipeline"),
        "reference: fold-specific development-selected WB-PIML settings",
        "positive RMSE increase means the removed/relaxed component helped the selected pipeline",
    )
    return result.sort_values("component_order").drop(columns="component_order")


def selection_rule_sensitivity(
    nested_candidates: pd.DataFrame,
    nested_selection: pd.DataFrame | None = None,
) -> pd.DataFrame:

    required = {
        "model", "outer_fold", "candidate_index", "inner_SB_RMSE",
        "inner_runout_survival_NLL", "reference_WB_CD_runout_NLL",
        "runout_noninferiority_margin", "censor_feasible", "is_reference",
        "basis_id", "min_samples_leaf",
    }
    if not required.issubset(nested_candidates.columns):
        raise KeyError(f"Selection sensitivity missing columns: {sorted(required - set(nested_candidates.columns))}")
    candidates = nested_candidates.loc[nested_candidates["model"].isin(("WB-PIML-Anchor", "ML-Strong"))].copy()
    if candidates.empty or candidates.duplicated(["model", "outer_fold", "candidate_index"]).any():
        raise ValueError("Need unique hybrid candidates in each outer fold")
    expected = {}
    if nested_selection is not None:
        formal = nested_selection.loc[nested_selection["model"].isin(("WB-PIML-Anchor", "ML-Strong"))]
        if formal.duplicated(["model", "outer_fold"]).any():
            raise ValueError("Need one formal hybrid choice per outer fold")
        expected = {(str(row.model), int(row.outer_fold)): int(row.candidate_index) for row in formal.itertuples()}
        if set(expected) != set(zip(candidates["model"].astype(str), candidates["outer_fold"].astype(int))):
            raise ValueError("Candidate folds and formal-choice folds differ")
    rows = []
    for (model, fold), table in candidates.groupby(["model", "outer_fold"], sort=True):
        for tolerance in dict.fromkeys((0.0, 0.01, NEAR_TIE_RMSE, 0.04)):
            chosen, replay = select_budgeted_candidate(table, float(tolerance))
            near = replay.loc[replay["within_RMSE_tolerance"]]
            for margin_multiplier in (0.0, 1.0, 2.0):
                feasible = near["inner_runout_survival_NLL"].le(
                    near["reference_WB_CD_runout_NLL"]
                    + margin_multiplier * near["runout_noninferiority_margin"] + 1e-12
                )
                selected_feasible = bool(
                    chosen["inner_runout_survival_NLL"]
                    <= chosen["reference_WB_CD_runout_NLL"]
                    + margin_multiplier * chosen["runout_noninferiority_margin"] + 1e-12
                )
                is_default = tolerance == NEAR_TIE_RMSE and margin_multiplier == 1.0
                matches_formal = pd.NA
                if is_default and expected:
                    matches_formal = chosen["candidate_index"] == expected[(model, int(fold))]
                    if not matches_formal:
                        raise AssertionError(f"Default budgeted selection replay differs in fold {fold}")
                rows.append({
                    "model": model, "outer_fold": int(fold), "RMSE_near_tie_tolerance": float(tolerance),
                    "censor_margin_multiplier": float(margin_multiplier),
                    "censor_margin_role": "diagnostic_only_does_not_change_selection",
                    "is_default_rule": bool(is_default), "n_near_tie_candidates": len(near),
                    "n_censor_feasible_candidates": int(feasible.sum()),
                    "selected_censor_feasible": selected_feasible,
                    "censor_constraint_relaxed": not selected_feasible,
                    "censor_guard_used_for_selection": False,
                    "selection_rule": chosen["selection_rule"],
                    "LOTO_shortlist_limit": 0, "LOTO_shortlist_candidate_indices": "",
                    "n_missing_LOTO_scores": 0, "missing_LOTO_candidate_indices": "",
                    "audit_status": "complete_replay", "LOTO_used_for_selection": False,
                    "selected_candidate_index": chosen["candidate_index"],
                    "selected_inner_SB_RMSE": chosen["inner_SB_RMSE"],
                    "selected_inner_runout_survival_NLL": chosen["inner_runout_survival_NLL"],
                    "selected_inner_LOTO_RMSE": np.nan,
                    "formal_selected_candidate_index": expected.get((model, int(fold)), pd.NA),
                    "default_matches_formal_choice": matches_formal,
                    "development_only_audit": True, "used_to_change_reported_model": False,
                })
    result = pd.DataFrame(rows)
    for column in ("selected_candidate_index", "formal_selected_candidate_index"):
        result[column] = result[column].astype("Int64")
    result["default_matches_formal_choice"] = result["default_matches_formal_choice"].astype("boolean")
    return result

def selection_protocol_fields():
    return {
        "hybrid_RMSE_near_tie_tolerance": NEAR_TIE_RMSE,
        "ET_Walker_RMSE_near_tie_tolerance": RMSE_TIE_TOLERANCE,
        "hybrid_LOTO_shortlist_limit": 0,
        "hybrid_selection_order": ["within pool inner campaign-balanced exact RMSE <= minimum + 0.02", "minimum inner runout survival NLL, then RMSE, then candidate_id"],
        "hybrid_censor_guard_role": "diagnostic only; not an enforced noninferiority constraint",
        "hybrid_temperature_policies": list(TEMPERATURE_POLICIES),
        "hybrid_candidate_count": len(HYBRID_CANDIDATES),
        "strong_control_candidate_count": len(CONTROL_CANDIDATES),
        "hybrid_residual_basis_count": len(HYBRID_BASES),
        "hybrid_inner_seed": "selection_seed + 100*inner_fold + 1; identical across both pools, gains and families",
        "environment_regularization_grid": list(ENVIRONMENT_STRENGTH_GRID),
        "hybrid_families": list(HYBRID_FAMILIES),
        "strong_control_selection": "independent 64-candidate pool, same folds and RMSE/NLL rule; includes nine physics-feature ExtraTrees candidates",
        "Matern_residual": "nu=1.5, length_scale=2, alpha=1, zero residual prior",
        "features": "seven numeric variables; coarse architecture, detailed architecture and environment; training-only transformations",
        "selection_sensitivity_replay": "same within-pool RMSE-budget/NLL rule; censor margin is diagnostic only",
        "secondary_fixed_temperature_policy": "soft250",
        "secondary_fixed_family": "ET_residual", "secondary_fixed_strength": 0.0,
    }


def grouped_folds_by(df: pd.DataFrame, group_column: str, n_splits: int, seed: int):
    require_exact_training(df, f"grouped_folds_by[{group_column}]")
    summary = df.groupby(group_column).agg(n=("row_id", "size")).reset_index()
    n_splits = min(n_splits, len(summary))
    rng = np.random.default_rng(seed)
    summary["tie"] = rng.random(len(summary))
    summary = summary.sort_values(["n", "tie"], ascending=[False, True])
    target = float(summary["n"].sum()) / n_splits
    load = np.zeros(n_splits)
    assignment: dict[str, int] = {}
    for _, row in summary.iterrows():
        score = load / max(target, 1.0)
        fold = int(np.argmin(score))
        load[fold] += float(row["n"])
        assignment[str(row[group_column])] = fold
    fold_id = df[group_column].astype(str).map(assignment).to_numpy(int)
    splits = []
    for fold in range(n_splits):
        valid = np.flatnonzero(fold_id == fold)
        train = np.flatnonzero(fold_id != fold)
        if not df.iloc[train]["is_exact"].any() or not df.iloc[valid]["is_exact"].any():
            raise RuntimeError(f"{group_column}: every fold must contain exact failures")
        if set(df.iloc[train][group_column].astype(str)) & set(df.iloc[valid][group_column].astype(str)):
            raise AssertionError(f"{group_column} leakage")
        splits.append((train, valid))
    return splits, fold_id + 1


def record_folds(df: pd.DataFrame, n_splits: int, seed: int):
    require_exact_training(df, "record_folds")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df), dtype=int)
    rng.shuffle(indices)
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for index, row_index in enumerate(indices):
        buckets[index % n_splits].append(int(row_index))
    fold_id = np.zeros(len(df), int)
    splits = []
    for fold, bucket in enumerate(buckets):
        valid = np.asarray(sorted(bucket), int)
        keep = np.ones(len(df), bool)
        keep[valid] = False
        train = np.flatnonzero(keep)
        fold_id[valid] = fold + 1
        splits.append((train, valid))
    return splits, fold_id


def fixed_safe_prediction(train: pd.DataFrame, test: pd.DataFrame, seed: int,
                          trees: int, residual_mode: str = "ET_residual"
                          ) -> tuple[np.ndarray, dict[str, float | str]]:

    inner_splits, _ = grouped_folds(
        train, min(4, train[PRIMARY_GROUP_COLUMN].nunique()), seed + 333
    )
    physical_parts = []
    inner_trees = min(trees, 120)
    for inner_fold, (fit_idx, valid_idx) in enumerate(inner_splits, 1):
        fit = train.iloc[fit_idx].reset_index(drop=True)
        valid = train.iloc[valid_idx].reset_index(drop=True)
        physical, _, _ = fit_hybrid_policy(
            fit, valid, COMPETING_WB_SPEC,
            seed + 1000 * inner_fold + 1, inner_trees,
            GAMMA, 3, 0.85, 0.75,
            "ET_residual", 1.0, 250.0, temperature_policy="soft250", strength=0.0,
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
        "temperature_policy": "soft250", "family": "ET_residual", "strength": 0.0,
        "selection_scope": "predeclared_fixed_secondary_configuration",
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
    physical_test, _, _ = fit_hybrid_policy(
        train, test, COMPETING_WB_SPEC, seed + 1, trees,
        GAMMA, 3, 0.85, 0.75,
        "ET_residual", 1.0, 250.0, temperature_policy="soft250", strength=0.0,
    )
    return np.asarray(physical_test, float), choice


def fixed_split_predictions(df: pd.DataFrame, splits, seed: int, trees: int,
                            strategy: str, repeat: int) -> pd.DataFrame:
    pieces = []
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        train = df.iloc[train_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        fold_seed = seed + 10000 * fold
        residual_mode = "extra_trees"
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


def _external_fold_task(df, fold, train_idx, test_idx, holdout_group, holdout_column, seed, trees, strategy, repeat):
    pieces, selection_rows = [], []
    expected_models = {"WB-PIML", "WB", "ExtraTrees", "ET-Walker", "ML-Ens", "ML-Strong", *SIMPLE_CONTROL_MODELS}
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


    (hybrid_choice, physical_oof, hybrid_candidates,
     control_choice, control_oof, control_candidates) = nested_select_structural(
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

    hybrid, hybrid_coef, hybrid_meta = fit_selected_hybrid(train, test, hybrid_choice, fold_seed + 1, trees)
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
    strong = fit_selected_control(train, test, control_choice, fold_seed + 1, trees)
    strong_scale = calibrate_aft_scale(control_oof)
    strong_half80, strong_half90 = empirical_interval_halfwidths(control_oof)
    model_predictions = {
        "ML-Strong": strong,
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
        part["sigma"] = strong_scale["sigma"] if model == "ML-Strong" else np.nan
        part["temperature_policy"] = hybrid_choice["temperature_policy"] if model == PROPOSED_MODEL else "not_applicable"
        part["residual_temperature_supported"] = (
            residual_temperature_supported
            if model == PROPOSED_MODEL else np.nan
        )
        for level, halfwidth in ((80, halfwidth80), (90, halfwidth90)):
            if model == "ML-Strong":
                half = strong_half80 if level == 80 else strong_half90
                part[f"lower{level}"] = np.asarray(prediction, float) - half
                part[f"upper{level}"] = np.asarray(prediction, float) + half
            elif model == PROPOSED_MODEL:
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
    selection_rows.append({**common, "model": "ML-Strong", "n_candidates_evaluated": len(control_candidates),
        "empirical_halfwidth_80": float(strong_half80), "empirical_halfwidth_90": float(strong_half90),
        "n_interval_calibration_exact": int(control_oof.is_exact.sum()),
        "interval_calibration": "development campaign-disjoint OOF absolute residuals",
        "outer_test_used_for_interval_calibration": False, **control_choice})
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
            **water_fit_metadata(train,hybrid_coef),
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

    simple_results = fit_simple_control_models(
        train, test, selection_seed, fold_seed + 1, trees
    )
    template = pieces[0]
    for name, result in simple_results.items():
        part = simple_control_prediction_part(test, template.columns, name, result)
        for column in ("strategy", "repeat", "fold", "holdout_group", "holdout_column"):
            part[column] = template[column].iloc[0]
        pieces.append(part)
        selection_rows.append({**common, "model": name, **result["choice"],
            "n_candidates_evaluated": len(result["candidates"]),
            "empirical_halfwidth_80": result["half80"],
            "empirical_halfwidth_90": result["half90"],
            "n_interval_calibration_exact": int(result["oof"].is_exact.sum()),
            "interval_calibration": "development campaign-disjoint OOF absolute residuals",
            "outer_test_used_for_interval_calibration": False,
            "candidate_scores_json": result["candidates"].to_json(orient="records"),
            **result["metadata"], **result["scale"]})

    fold_predictions = pd.concat(pieces, ignore_index=True)
    for model in expected_models:
        model_rows = fold_predictions.loc[fold_predictions["model"].eq(model)]
        if len(model_rows) != len(test) or model_rows["row_id"].duplicated().any():
            raise AssertionError(
                f"{strategy} fold {fold}: {model} must predict each holdout row once"
            )

    probability_parts, probability_calibration_parts = factor_probability_records(
        train, pieces, proposed_oof, control_oof, strong_scale, hybrid_choice, control_choice,
        et_choice, walker_choice, selection_seed, trees, strategy, fold, holdout_text, simple_results)
    return pieces, selection_rows, probability_parts, probability_calibration_parts


def nested_external_split_predictions(
    df: pd.DataFrame,
    splits,
    labels,
    holdout_column: str,
    seed: int,
    trees: int,
    strategy: str,
    repeat: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:


    if len(splits) != len(labels):
        raise ValueError("External holdout splits and labels must have equal length")
    if holdout_column not in df.columns:
        raise KeyError(f"Unknown external holdout column: {holdout_column}")

    pieces: list[pd.DataFrame] = []
    probability_parts, probability_calibration_parts = [], []
    selection_rows: list[dict[str, object]] = []
    expected_models = {"WB-PIML", "WB", "ExtraTrees", "ET-Walker", "ML-Ens"}

    tasks = [(df, fold, train_idx, test_idx, holdout_group, holdout_column, seed, trees, strategy, repeat)
             for fold, ((train_idx, test_idx), holdout_group) in enumerate(zip(splits, labels), 1)]
    for prediction_rows, choices, probability_rows, calibration_rows in _parallel_map(_external_fold_task, tasks, strategy):
        pieces.extend(prediction_rows)
        selection_rows.extend(choices)
        probability_parts.extend(probability_rows)
        probability_calibration_parts.extend(calibration_rows)

    return (
        pd.concat(pieces, ignore_index=True),
        pd.DataFrame(selection_rows),
        pd.concat(probability_parts, ignore_index=True, sort=False),
        pd.concat(probability_calibration_parts, ignore_index=True, sort=False),
    )


def validation_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.loc[predictions["is_exact"]].groupby(
        ["strategy", "repeat", "model"]
    ):
        strategy, repeat, model = keys
        source_groups = [g.reset_index(drop=True) for _, g in group.groupby(PRIMARY_GROUP_COLUMN)]
        rows.append({"strategy": strategy, "repeat": repeat, "model": model,
                     "n_exact": len(group), "n_campaigns": group["campaign_id"].nunique(),
                     "n_sources": group["source_id"].nunique(),
                     **sb_metrics(source_groups)})
    return pd.DataFrame(rows)


def _validation_repeat_task(df, trees, strategy, repeat, seed):
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
    prediction = fixed_split_predictions(df, splits, seed, min(trees, 300), strategy, repeat)
    assignment = df[["row_id", "source_id", "campaign_id", "record_equivalence_id", "series_group_id"]].copy()
    assignment["strategy"] = strategy
    assignment["repeat"] = repeat
    assignment["fold"] = fold_id
    assignment["split_group"] = split_group
    return prediction, assignment

def repeated_validation(df: pd.DataFrame, trees: int, repeats: int
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    assignment_rows = []
    tasks = [
        (df, trees, strategy, repeat, seed)
        for strategy in ["record_interpolation", "series_disjoint", "source_disjoint", "campaign_disjoint"]
        for repeat, seed in enumerate(REPEATED_VALIDATION_SEEDS[:repeats], 1)
    ]
    for prediction, assignment in _parallel_map(_validation_repeat_task, tasks, "Repeated validation"):
        predictions.append(prediction)
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    probability_parts, probability_calibration_parts = [], []
    prediction_parts = []
    selection_parts = []
    group_rows = []
    for scenario_index, (scenario, column, splits, labels) in enumerate(scenarios):
        pred, selection, probability, calibration = nested_external_split_predictions(
            df, splits, labels, column,
            SEED + 700000 + 1000000 * scenario_index,
            min(trees, 400), scenario, 1,
        )
        prediction_parts.append(pred)
        probability_parts.append(probability)
        probability_calibration_parts.append(calibration)
        selection_parts.append(selection)
        expected_row_ids = set(
            df.iloc[np.concatenate([test_idx for _, test_idx in splits])]["row_id"]
        )
        for model in ["WB-PIML", "WB", "ExtraTrees", "ET-Walker", "ML-Ens", "ML-Strong", *SIMPLE_CONTROL_MODELS]:
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
        pd.concat(probability_parts, ignore_index=True, sort=False),
        pd.concat(probability_calibration_parts, ignore_index=True, sort=False),
    )


def external_paired_effects(predictions: pd.DataFrame, bootstrap: int,
                            seed: int) -> pd.DataFrame:

    tables = []
    scenarios = sorted(predictions["strategy"].astype(str).unique())
    for scenario_index, scenario in enumerate(scenarios):
        scenario_predictions = predictions.loc[
            predictions["strategy"].astype(str).eq(scenario)
        ].copy()
        for comparator_index, comparator in enumerate(PRIMARY_COMPARATORS + SIMPLE_CONTROL_MODELS):
            table = paired_effects(
                scenario_predictions, bootstrap,
                seed + 10000 * scenario_index + 101 * comparator_index,
                comparator=comparator,
            )
            table.insert(0, "scenario", scenario)
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def external_uq_summary(predictions: pd.DataFrame, bootstrap: int,
                        seed: int) -> pd.DataFrame:

    tables = []
    for scenario_index, scenario in enumerate(
        sorted(predictions["strategy"].astype(str).unique())
    ):
        table = uq_summary(
            predictions.loc[predictions["strategy"].astype(str).eq(scenario)].copy(),
            bootstrap, seed + 10000 * scenario_index,
        )
        table.insert(0, "scenario", scenario)
        table["calibration"] = (
            "per-holdout nested-selected development OOF absolute residuals"
        )
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


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

