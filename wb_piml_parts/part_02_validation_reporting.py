


def sensitivity_audit(df: pd.DataFrame, outer_splits, trees: int,
                      nested_selection: pd.DataFrame) -> pd.DataFrame:
    """Evaluate physical assumptions using each development fold's selected candidate.

    Held-out responses are used only to compute sensitivity metrics.
    """
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
            "water_factor": float(variant.get("water_factor", 1.5)),
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
    """Extract genuine retrained component removals from the sensitivity audit."""
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
    """Replay the development RMSE budget and survival-NLL ranking."""
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
    """Predict secondary validation sets with fixed residual settings and OOF intervals."""
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

    # Select hyperparameters using development data only.
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

    hybrid, _, _ = fit_selected_hybrid(train, test, hybrid_choice, fold_seed + 1, trees)
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
    """Fit each external holdout after training-only nested model selection.

    The held-out responses are not passed to any selector.  WB-PIML,
    ExtraTrees and ET-Walker are re-selected with campaign-disjoint inner
    folds in the remaining development data.  RF, GBDT and SVR retain their
    predeclared settings; ML-Ens uses those three models and the selected
    ExtraTrees component.
    """
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
    """Campaign-bootstrap paired effects within each held-out-domain scenario."""
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
    """Evaluate development-calibrated intervals under LOSO/LOAO/LOTO/LOEO."""
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






def _metric_lookup(metrics: pd.DataFrame, model: str, key: str) -> float:
    return float(metrics.loc[metrics["model"].eq(model), key].iloc[0])


@dataclass(frozen=True)
class FigureSlideImage:
    """One rendered figure or subplot destined for the results deck."""

    title: str
    png_bytes: bytes
    image_kind: str


class FigureDeckCollector:
    """Collect full figures and non-empty panels, then write one 16:9 PPTX."""

    def __init__(self, data_name: str) -> None:
        self.data_name = str(data_name)
        self.images: list[FigureSlideImage] = []

    @staticmethod
    def _panel_png(fig, ax, dpi: int = 240) -> bytes:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tight = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        width_in, height_in = fig.get_size_inches()
        pad_x = max(0.08, tight.width * 0.04)
        pad_y = max(0.08, tight.height * 0.06)
        crop = Bbox.from_extents(
            max(0.0, tight.x0 - pad_x),
            max(0.0, tight.y0 - pad_y),
            min(float(width_in), tight.x1 + pad_x),
            min(float(height_in), tight.y1 + pad_y),
        )
        buffer = BytesIO()
        fig.savefig(
            buffer, format="png", dpi=dpi, bbox_inches=crop,
            facecolor="white", edgecolor="none",
        )
        return buffer.getvalue()

    def add_figure(self, fig, output_path: Path, full_title: str,
                   panel_axes, panel_titles) -> None:
        """Save the composite PNG and retain it plus every named panel for PPTX."""
        panel_axes = list(panel_axes)
        panel_titles = list(panel_titles)
        if len(panel_axes) != len(panel_titles):
            raise ValueError("Each PowerPoint subplot must have exactly one title")
        fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
        self.images.append(FigureSlideImage(full_title, output_path.read_bytes(), "full figure"))
        for ax, title in zip(panel_axes, panel_titles):
            if not ax.get_visible() or not ax.axison:
                continue
            self.images.append(FigureSlideImage(
                str(title), self._panel_png(fig, ax), "individual subplot"
            ))

    @staticmethod
    def _add_title(slide, title: str, top: float = 0.25, size: int = 28) -> None:
        box = slide.shapes.add_textbox(Inches(0.55), Inches(top), Inches(12.25), Inches(0.62))
        frame = box.text_frame
        frame.clear()
        paragraph = frame.paragraphs[0]
        paragraph.text = title
        paragraph.font.name = "Arial"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = True
        paragraph.font.color.rgb = RGBColor(22, 52, 78)
        paragraph.alignment = PP_ALIGN.LEFT

    @staticmethod
    def _add_image_contain(slide, png_bytes: bytes, left: float, top: float,
                           width: float, height: float) -> None:
        with Image.open(BytesIO(png_bytes)) as picture:
            pixel_width, pixel_height = picture.size
        scale = min(width / pixel_width, height / pixel_height)
        draw_width = pixel_width * scale
        draw_height = pixel_height * scale
        draw_left = left + (width - draw_width) / 2
        draw_top = top + (height - draw_height) / 2
        slide.shapes.add_picture(
            BytesIO(png_bytes), Inches(draw_left), Inches(draw_top),
            width=Inches(draw_width), height=Inches(draw_height),
        )

    def write(self, output_path: Path) -> None:
        """Write and ZIP-validate the complete results presentation atomically."""
        if any(value is None for value in (Image, Presentation, RGBColor, PP_ALIGN, Inches, Pt)):
            raise ImportError("PowerPoint export requires Pillow and python-pptx")
        presentation = Presentation()
        presentation.slide_width = Inches(13.333)
        presentation.slide_height = Inches(7.5)
        blank_layout = presentation.slide_layouts[6]

        cover = presentation.slides.add_slide(blank_layout)
        cover.background.fill.solid()
        cover.background.fill.fore_color.rgb = RGBColor(244, 248, 251)
        accent = cover.shapes.add_shape(1, Inches(0), Inches(0), Inches(0.25), Inches(7.5))
        accent.fill.solid(); accent.fill.fore_color.rgb = RGBColor(21, 101, 128)
        accent.line.fill.background()
        self._add_title(cover, "WB-PIML — Figures", top=1.75, size=34)
        subtitle = cover.shapes.add_textbox(Inches(0.62), Inches(2.65), Inches(11.8), Inches(1.6))
        frame = subtitle.text_frame
        frame.clear()
        paragraph = frame.paragraphs[0]
        paragraph.text = (
            f"8 composite figures and every non-empty subplot\n"
            f"Generated from {self.data_name}"
        )
        paragraph.font.name = "Arial"; paragraph.font.size = Pt(21)
        paragraph.font.color.rgb = RGBColor(58, 77, 91)

        total = len(self.images) + 1
        for slide_number, item in enumerate(self.images, 2):
            slide = presentation.slides.add_slide(blank_layout)
            slide.background.fill.solid()
            slide.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
            self._add_title(slide, item.title)
            self._add_image_contain(slide, item.png_bytes, 0.45, 1.0, 12.43, 5.95)
            footer = slide.shapes.add_textbox(Inches(0.55), Inches(7.02), Inches(12.2), Inches(0.25))
            footer_frame = footer.text_frame
            footer_frame.clear()
            footer_p = footer_frame.paragraphs[0]
            footer_p.text = f"{item.image_kind}  |  {self.data_name}  |  {slide_number}/{total}"
            footer_p.font.name = "Arial"; footer_p.font.size = Pt(9)
            footer_p.font.color.rgb = RGBColor(102, 117, 127)
            footer_p.alignment = PP_ALIGN.RIGHT

        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}_", suffix=".pptx", dir=output_path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            presentation.save(temporary_path)
            with zipfile.ZipFile(temporary_path, "r") as archive:
                damaged_member = archive.testzip()
                if damaged_member is not None:
                    raise IOError(f"PowerPoint ZIP validation failed at {damaged_member}")
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def make_figures(out: Path, predictions: pd.DataFrame, metrics: pd.DataFrame,
                 effects: pd.DataFrame, uq: pd.DataFrame, runout: pd.DataFrame,
                 source_gains: pd.DataFrame, deck: FigureDeckCollector) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    exact = predictions.loc[predictions["is_exact"]]
    lim = [float(exact["logN"].min() - 0.3), float(exact["logN"].max() + 0.3)]

    chosen = ["WB", "ML-Ens", "ExtraTrees", "ML-Strong", "WB-PIML"]
    fig = plt.figure(figsize=(15, 9.5))
    grid = fig.add_gridspec(2, 6)
    positions = [
        (0, slice(0, 2)), (0, slice(2, 4)), (0, slice(4, 6)),
        (1, slice(1, 3)), (1, slice(3, 5)),
    ]
    axes = []
    for row, columns in positions:
        if axes:
            axes.append(fig.add_subplot(
                grid[row, columns], sharex=axes[0], sharey=axes[0]
            ))
        else:
            axes.append(fig.add_subplot(grid[row, columns]))
    for ax, model in zip(axes, chosen):
        part = exact.loc[exact["model"].eq(model)]
        ax.fill_between(lim, np.array(lim) - LOG5, np.array(lim) + LOG5, color="#f6c979", alpha=0.18)
        ax.plot(lim, lim, "k-", lw=1.4)
        ax.scatter(part["logN"], part["pred_logN"], s=28, alpha=0.72, edgecolor="white", linewidth=0.3)
        ax.set(xlim=lim, ylim=lim, title=f"{model}\nSB-RMSE={_metric_lookup(metrics, model, 'SB_RMSE'):.3f}")
    fig.supxlabel(r"Observed exact-fracture $\log_{10}(N_f)$")
    fig.supylabel(r"Campaign-disjoint prediction $\log_{10}(N_f)$")
    fig.suptitle("Campaign-disjoint outer-test predictions for exact-fracture records", y=0.99)
    fig.tight_layout()
    deck.add_figure(
        fig, out / "01_exact_failure_predictions.png",
        "Figure 1 — Exact-fracture outer-test predictions",
        axes,
        [f"Figure 1{chr(65 + index)} — {model} exact-fracture predictions"
         for index, model in enumerate(chosen)],
    )
    plt.close(fig)

    keys = [("SB_RMSE", "RMSE ↓"), ("SB_R2", r"$R^2$ ↑"), ("SB_F5", r"$F_5$ ↑"), ("SB_C_star", r"$C^*$ ↓")]
    chosen = ["WB", "ML-Ens", "ExtraTrees", "ET-Walker", "ML-Strong", "WB-PIML"]
    selected = metrics.loc[metrics["model"].isin(chosen)].set_index("model").loc[chosen].reset_index()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    colors = ["#777777", "#4c78a8", "#72b7b2", "#54a24b", "#b279a2", "#f58518"]
    for ax, (key, title) in zip(axes, keys):
        y = selected[key].to_numpy(float)
        lo, hi = selected[f"{key}_lo"].to_numpy(float), selected[f"{key}_hi"].to_numpy(float)
        ax.bar(selected["model"], y, color=colors, edgecolor="white")
        ax.errorbar(np.arange(len(y)), y, yerr=np.vstack([y - lo, hi - y]), fmt="none", ecolor="black", capsize=3)
        ax.set_title(title); ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Campaign-disjoint point-prediction metrics for exact fractures (95% campaign-cluster intervals)")
    fig.tight_layout()
    deck.add_figure(
        fig, out / "02_campaign_balanced_metrics.png",
        "Figure 2 — Campaign-balanced primary metrics",
        axes,
        [f"Figure 2{chr(65 + index)} — {title.replace(' $', ' ')}"
         for index, (_, title) in enumerate(keys)],
    )
    plt.close(fig)

    ablation_models = ["WB", "WB-CD", "WB-M", "WB-Ox", "ExtraTrees", "ET-Walker",
                       "WB-PIML", "WB-PIML-M", "WB-PIML-Ox"]
    physical = metrics.loc[metrics["model"].isin(ablation_models)].set_index("model").loc[ablation_models].reset_index()
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    y = physical["SB_RMSE"].to_numpy(float)
    ax.bar(physical["model"], y, color="#72b7b2", edgecolor="black", alpha=0.85)
    ax.errorbar(np.arange(len(y)), y,
                yerr=np.vstack([y - physical["SB_RMSE_lo"], physical["SB_RMSE_hi"] - y]),
                fmt="none", ecolor="black", capsize=4)
    ax.set_ylabel("Campaign-balanced RMSE ↓")
    ax.set_title("Retrained ablation of the physical trunk and residual correction")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    deck.add_figure(
        fig, out / "03_retrained_ablation.png",
        "Figure 3 — Retrained model ablation",
        [ax], ["Figure 3A — Competing-damage and residual ablation"],
    )
    plt.close(fig)

    paired_comparators = ["ExtraTrees", "ML-Strong", "ML-Ens"]
    paired_colors = {"ExtraTrees": "#72b7b2", "ET-Walker": "#54a24b", "ML-Strong": "#b279a2", "ML-Ens": "#4c78a8"}
    effect_labels = {
        "RMSE_gain": "RMSE gain",
        "MAE_gain": "MAE gain",
        "R2_gain": r"$R^2$ gain",
        "F5_gain": r"$F_5$ gain",
        "C_star_gain": r"$C^*$ gain",
    }
    paired_rows = effects.loc[
        effects["comparator"].isin(paired_comparators)
        & effects["effect"].isin(effect_labels)
    ]
    paired_low = float(min(0.0, paired_rows["CI95_lo"].min()))
    paired_high = float(max(0.0, paired_rows["CI95_hi"].max()))
    paired_pad = 0.05 * max(paired_high - paired_low, 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), sharey=True)
    for ax, comparator in zip(axes, paired_comparators):
        effect_plot = effects.loc[effects["comparator"].eq(comparator)].iloc[::-1].reset_index(drop=True)
        x = effect_plot["estimate"].to_numpy(float)
        ax.errorbar(
            x, np.arange(len(effect_plot)),
            xerr=np.vstack([x - effect_plot["CI95_lo"], effect_plot["CI95_hi"] - x]),
            fmt="o", color=paired_colors[comparator], ecolor=paired_colors[comparator], capsize=4,
        )
        ax.axvline(0, color="black", lw=1.2)
        ax.set_yticks(
            np.arange(len(effect_plot)),
            [effect_labels.get(value, value) for value in effect_plot["effect"]],
        )
        ax.set_xlim(paired_low - paired_pad, paired_high + paired_pad)
        ax.set_xlabel("Paired gain (positive favors WB-PIML)")
        ax.set_title(f"WB-PIML vs {comparator}")
    fig.suptitle("Paired comparison with predictive controls (95% campaign-cluster bootstrap intervals)", y=1.01)
    fig.tight_layout()
    deck.add_figure(
        fig, out / "04_paired_effect_intervals.png",
        "Figure 4 — Paired performance-effect intervals",
        axes,
        [f"Figure 4{chr(65 + index)} — WB-PIML versus {model}"
         for index, model in enumerate(paired_comparators)],
    )
    plt.close(fig)

    proposed = exact.loc[exact["model"].eq("WB-PIML")].sort_values("logN")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    axes[0].errorbar(proposed["logN"], proposed["pred_logN"],
                     yerr=np.vstack([proposed["pred_logN"] - proposed["lower90"], proposed["upper90"] - proposed["pred_logN"]]),
                     fmt="o", ms=3.5, alpha=0.45, ecolor="#90caf9", color="#4c78a8")
    interval_ylim = [
        float(min(lim[0], proposed["lower90"].min()) - 0.2),
        float(max(lim[1], proposed["upper90"].max()) + 0.2),
    ]
    axes[0].plot(lim, lim, "k-", lw=1.2); axes[0].set(xlim=lim, ylim=interval_ylim,
        xlabel=r"Observed exact-fracture $\log_{10}(N_f)$",
        ylabel=r"WB-PIML prediction $\log_{10}(N_f)$",
        title="Outer-test predictions with empirical 90% intervals")
    axes[1].errorbar(uq["nominal"], uq["coverage"],
                     yerr=np.vstack([uq["coverage"] - uq["coverage_lo"], uq["coverage_hi"] - uq["coverage"]]),
                     fmt="o-", capsize=4, color="#f58518", label="WB-PIML")
    axes[1].plot([0.75, 0.95], [0.75, 0.95], "k--", label="ideal")
    axes[1].set(xlim=(0.76, 0.94), ylim=(0.60, 1.01), xlabel="Nominal coverage", ylabel="Campaign-balanced empirical coverage",
                title="Empirical coverage (95% campaign-cluster confidence intervals)"); axes[1].legend()
    fig.tight_layout()
    deck.add_figure(
        fig, out / "05_empirical_prediction_intervals.png",
        "Figure 5 — Empirical prediction intervals",
        axes,
        ["Figure 5A — Outer-test empirical 90% intervals",
         "Figure 5B — Empirical interval coverage"],
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    nll_plot = runout.dropna(subset=["SB_runout_NLL"])
    axes[0].bar(nll_plot["model"], nll_plot["SB_runout_NLL"], color="#72b7b2")
    axes[0].set(title="Runout survival negative log-likelihood", ylabel="Campaign-balanced survival NLL ↓")
    axes[1].bar(runout["model"], runout["SB_point_below_runout_limit_rate"], color="#e45756")
    axes[1].set(title="Point predictions relative to runout limits", ylabel="Fraction below stopping cycles ↓")
    for ax in axes: ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    deck.add_figure(
        fig, out / "06_runout_evaluation.png",
        "Figure 6 — Runout evaluation under right censoring",
        axes,
        ["Figure 6A — Runout survival negative log-likelihood",
         "Figure 6B — Point prediction versus runout limit"],
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    gains = source_gains.sort_values("RMSE_gain").reset_index(drop=True)
    source_labels = [
        f"{row['plot_label']} (n={int(row['n_exact'])})"
        for _, row in gains.iterrows()
    ]
    ax.barh(source_labels, gains["RMSE_gain"],
            color=np.where(gains["RMSE_gain"] >= 0, "#54a24b", "#e45756"))
    ax.axvline(0, color="black", lw=1.2); ax.set_xlabel("Source RMSE gain: ML-Ens − WB-PIML")
    ax.set_title("Variation in RMSE gain across data sources")
    fig.tight_layout()
    deck.add_figure(
        fig, out / "07_source_level_gains.png",
        "Figure 7 — Source-specific RMSE gain",
        [ax], ["Figure 7A — Source-level RMSE gain"],
    )
    plt.close(fig)


def make_validation_figure(out: Path, validation_summary: pd.DataFrame,
                           external_summary: pd.DataFrame,
                           deck: FigureDeckCollector) -> None:
    """Plot fixed secondary validation and nested-selected factor holdouts."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    tier_order = ["record_interpolation", "series_disjoint", "source_disjoint", "campaign_disjoint"]
    tier_labels = ["Record\ninterpolation", "Series\ndisjoint", "Source\ndisjoint", "Campaign\ndisjoint"]
    fixed_models = ["WB", "ExtraTrees", "ET-Walker", "ML-Ens", "WB-PIML"]
    external_models = ["WB", "ExtraTrees", "ET-Walker", "ML-Ens", "ML-Strong", "WB-PIML"]
    colors = {"WB": "#777777", "WB-PIML": "#f58518", "ExtraTrees": "#72b7b2",
              "ET-Walker": "#54a24b", "ML-Ens": "#4c78a8", "ML-Strong": "#b279a2"}
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6), sharey=True)
    x = np.arange(len(tier_order))
    fixed_width = 0.15
    fixed_offsets = (np.arange(len(fixed_models)) - (len(fixed_models) - 1) / 2) * fixed_width
    for offset, model in zip(fixed_offsets, fixed_models):
        values, errors = [], []
        for tier in tier_order:
            selected = validation_summary.loc[
                validation_summary["strategy"].eq(tier) & validation_summary["model"].eq(model)
            ]
            if len(selected) != 1:
                raise AssertionError(f"Expected one fixed validation summary for {tier}/{model}")
            row = selected.iloc[0]
            values.append(float(row["SB_RMSE_mean"]))
            errors.append(float(row["SB_RMSE_sd"]) if np.isfinite(row["SB_RMSE_sd"]) else 0.0)
        axes[0].bar(x + offset, values, fixed_width, yerr=errors, capsize=3,
                    label=model, color=colors[model], edgecolor="white")
    axes[0].set_xticks(x, tier_labels)
    axes[0].set_ylabel(r"Campaign-balanced RMSE in $\log_{10}(N_f)$ ↓")
    axes[0].set_title("Fixed-configuration repeated validation\n(error bars: across-repeat SD)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.20), ncol=3)

    scenario_order = ["LOSO", "LOAO", "LOTO", "LOEO"]
    x = np.arange(len(scenario_order))
    external_width = 0.13
    external_offsets = (np.arange(len(external_models)) - (len(external_models) - 1) / 2) * external_width
    for offset, model in zip(external_offsets, external_models):
        values = []
        for scenario in scenario_order:
            selected = external_summary.loc[
                external_summary["scenario"].eq(scenario) & external_summary["model"].eq(model),
                "SB_RMSE",
            ]
            if len(selected) != 1:
                raise AssertionError(f"Expected one external summary for {scenario}/{model}")
            values.append(float(selected.iloc[0]))
        axes[1].bar(x + offset, values, external_width, label=model,
                    color=colors[model], edgecolor="white")
    axes[1].set_xticks(x, scenario_order)
    axes[1].set_title("Factor-held-out stress tests\n(model selection within each training partition)")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper center",
                   bbox_to_anchor=(0.5, -0.20), ncol=3)
    axes[1].text(
        0.01, -0.43,
        "LOSO: source; LOAO: coarse architecture; LOTO: temperature regime; LOEO: environment\n"
        "Campaign overlap is reported separately for each holdout",
        transform=axes[1].transAxes, fontsize=8.5, va="top",
    )
    fig.suptitle("Prediction errors under different validation schemes", y=1.02)
    fig.tight_layout()
    deck.add_figure(
        fig, out / "08_validation_scope_comparison.png",
        "Figure 8 — Validation-scope comparison", axes,
        ["Figure 8A — Fixed-configuration repeated validation",
         "Figure 8B — Factor-held-out tests including ML-Strong"],
    )
    plt.close(fig)


def write_workbook(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    if Font is None:
        warnings.warn("openpyxl unavailable; workbook was not written")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=".xlsx", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with pd.ExcelWriter(temporary_path, engine="openpyxl") as writer:
            for sheet, table in tables.items():
                table.to_excel(writer, sheet_name=sheet[:31], index=False)
            workbook = writer.book
            for sheet in workbook.worksheets:
                sheet.freeze_panes = "A2"
                sheet.auto_filter.ref = sheet.dimensions
                sheet.sheet_view.showGridLines = False
                for cell in sheet[1]:
                    cell.font = Font(color="FFFFFF", bold=True)
                    cell.fill = PatternFill("solid", fgColor="1F4E78")
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                for column in range(1, sheet.max_column + 1):
                    values = [str(sheet.cell(row, column).value or "") for row in range(1, min(sheet.max_row, 250) + 1)]
                    sheet.column_dimensions[get_column_letter(column)].width = min(max(11, max(map(len, values)) + 2), 48)
                for row in sheet.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(vertical="top", wrap_text=False)
            text_layouts = {
                "README": {"A": 28, "B": 100},
                "Protocol": {"A": 38, "B": 110},
                "Data_Audit": {"A": 48, "B": 80},
                "Metric_Definitions": {"A": 22, "B": 34, "C": 100},
                "Model_Specs": {"A": 24, "B": 44, "D": 110},
                "Validation_Definitions": {"A": 24, "B": 100, "C": 100},
                "Claim_Gate": {"A": 46, "B": 12, "C": 110, "D": 20},
                "References": {"A": 44, "B": 70, "C": 100},
            }
            for sheet_name, widths in text_layouts.items():
                if sheet_name not in workbook.sheetnames:
                    continue
                sheet = workbook[sheet_name]
                for column, width in widths.items():
                    sheet.column_dimensions[column].width = width
                for row_index in range(2, sheet.max_row + 1):
                    sheet.row_dimensions[row_index].height = 32
                    for cell in sheet[row_index]:
                        cell.alignment = Alignment(vertical="top", wrap_text=True)
        with zipfile.ZipFile(temporary_path) as archive:
            damaged_member = archive.testzip()
            if damaged_member is not None:
                raise IOError(f"Workbook ZIP validation failed at {damaged_member}")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_readme(path: Path, data_path: Path, metrics: pd.DataFrame, effects: pd.DataFrame,
                 gate: pd.DataFrame, df: pd.DataFrame, parameters: pd.DataFrame,
                 n_runout: int) -> None:
    proposed = metrics.loc[metrics["model"].eq(PROPOSED_MODEL)].iloc[0]
    anchor = metrics.loc[metrics["model"].eq("WB-PIML-Anchor")].iloc[0]
    proposed_parameters = parameters.loc[parameters["model"].eq(PROPOSED_MODEL)]
    mix_text = ", ".join(f"{value:.2f}" for value in proposed_parameters["physics_mix"])
    comparison_models = ["WB", "WB-CD", "RF", "ExtraTrees", "ET-Walker", "GBDT", "SVR",
                         "ML-Ens", "ML-Strong", *CENSORED_MODELS, PROPOSED_MODEL]
    primary_best = metrics.loc[metrics["model"].isin(comparison_models)].sort_values("SB_RMSE").iloc[0]
    paired_lines = []
    for name in PRIMARY_COMPARATORS:
        selected = effects.loc[effects["comparator"].eq(name) & effects["effect"].eq("RMSE_gain")]
        if len(selected) != 1:
            raise AssertionError(f"README requires one paired RMSE result for {name}")
        result = selected.iloc[0]
        paired_lines.append(
            f"- Paired RMSE gain over {name}: **{result['estimate']:.3f}**, "
            f"95% campaign-bootstrap CI **[{result['CI95_lo']:.3f}, {result['CI95_hi']:.3f}]**."
        )
    n_exact = int(df["is_exact"].sum())
    text = f"""# WB-PIML results

## Analysis

- Data: `{data_path.name}` / `{SHEET}` only: {n_exact} exact fractures and {n_runout} right-censored runouts.
- All {len(df)} retained records have unit likelihood weight. Exact fractures contribute density terms; runouts contribute survival terms and never enter point-residual regression or point-life metrics as exact failures.
- Labels, evidence fields and campaign IDs are read from the workbook. The retained table has {df['record_equivalence_id'].nunique()} distinct equivalence IDs. ID uniqueness is a structural check and does not establish independence of every cross-publication experiment; unresolved source provenance remains qualified in the workbook.
- {df['source_id'].nunique()} sources belong to {df['campaign_id'].nunique()} reviewed campaigns. Primary validation uses five campaign-disjoint outer folds and campaign-disjoint development folds; no campaign crosses a primary train/test split.
- The five-parameter physical trunk combines Walker--Basquin mechanical damage and time-dependent environmental damage. It is fitted by joint censored AFT likelihood with mechanical-coefficient regularization and a nonnegative penalty on the training mean squared environmental log-life decrement. The environmental-penalty strength is selected within development folds.
- The hybrid candidates retain the physical prediction with unit weight and add an exact-fracture residual. The three residual families are ExtraTrees, ExtraTrees with training-fitted physical features, and Matérn kernel ridge. Matérn uses nu=1.5, length scale=2 and ridge alpha=1, with zero residual prior.
- All candidates share the original observable numeric inputs, coarse architecture, the detailed architecture descriptor, and environment. The physics-feature family adds mechanical predicted life, environmental life decrement and Walker stress, computed from a physical model fitted only on the corresponding development subset. Sources, campaigns and record IDs are not predictive features.
- Every hybrid uses soft temperature support `exp[-(nearest exact-development temperature distance / 250 C)^2]`. Support depends on development covariates only. The same-temperature flag reports availability and does not imply zero residual at unseen temperatures.
- WB-PIML evaluates {len(HYBRID_CANDIDATES)} candidates: three residual families, gamma in {{0.75, 0.85, 0.95}}, environmental strength in {{0, 0.1, 1}}, and eta in {{0.75, 1}}. ML-Strong separately selects among {len(CONTROL_CANDIDATES)} candidates: 18 ExtraTrees, 18 gradient boosting, 18 Matérn, one fixed ML ensemble and nine direct ExtraTrees controls with the same training-fitted physical features. ML-Strong is a selected predictive control, and its full pool includes physics-informed candidates.
- Each pool independently retains candidates with inner campaign-balanced exact RMSE within {NEAR_TIE_RMSE:.3f} of its own minimum, then minimizes calibrated inner runout survival NLL; RMSE and candidate index resolve ties. All candidates in a pool are evaluated on the same development folds. Total search budgets are 54 and 64, not equal. ET-Walker retains its separate gamma/tree search.
- The WB-CD reference survival comparison is a diagnostic, not a selection filter. No affine output calibration is applied. LOTO is a separate factor-held-out stress test and does not choose the hybrid configuration.
- Repeated record-, series-, source- and campaign-disjoint validation uses a fixed secondary ExtraTrees residual: gamma={GAMMA}, environmental strength=0, eta=0.75, leaf=3, max_features=0.85, and soft support at 250 C. Duplicate/label and S-star sensitivity use the same fixed configuration; these results are distinct from the nested-selected primary model.
- LOSO/LOAO/LOTO/LOEO re-select WB-PIML, ML-Strong, ExtraTrees and ET-Walker using only the remaining records. Factor-held-out splits do not impose campaign separation; actual overlap is reported per holdout. LOAO retains coarse architecture groups while detailed architecture remains a predictive descriptor.
- The 80% and 90% prediction intervals use campaign-balanced empirical absolute errors from development OOF predictions. They are not formal finite-sample conformal or design-life guarantees.

## Results

- Lowest campaign-balanced RMSE among the listed physical baselines, predictive controls and proposed model: **{primary_best['model']} = {primary_best['SB_RMSE']:.3f}**.
- WB-PIML campaign-balanced RMSE: **{proposed['SB_RMSE']:.3f}**.
- Selected hybrid branch RMSE: **{anchor['SB_RMSE']:.3f}**; reported-model difference: **{anchor['SB_RMSE'] - proposed['SB_RMSE']:.3f}**.
- Physical-trunk weights by fold: **[{mix_text}]**; every value is 1.00.
{chr(10).join(paired_lines)}

`SB_*` denotes campaign-balanced metrics. `Claim_Gate` reports computed criteria, including comparisons with ML-Strong; a lower point estimate alone does not establish superiority or independent external validation.

## Files

- One results workbook: `WB_PIML_results.xlsx`, including selection traces, fold assignments, source audits and all validation summaries.
- Eight composite PNGs and `{PPTX_OUTPUT_NAME}`; the presentation includes each composite and every non-empty subplot.
- `protocol.json` and this README describe the executed method. CSV exports are optional through `--export-csv`.
"""
    path.write_text(text, encoding="utf-8")
