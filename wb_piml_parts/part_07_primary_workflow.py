def _outer_fold_task(df, outer_fold, train_idx, test_idx, trees):
    prediction_parts, parameter_rows, audit_rows = [], [], []
    nested_selection_rows, nested_candidate_parts = [], []
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
    (hybrid_choice, physical_oof, hybrid_candidates,
     control_choice, control_oof, control_candidates) = nested_select_structural(
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
        {"outer_fold": outer_fold, "model": "ML-Strong", **control_choice},
        {"outer_fold": outer_fold, "model": "WB-PIML", **fusion_choice},
        {"outer_fold": outer_fold, "model": "ET-Walker", **walker_choice},
        {"outer_fold": outer_fold, "model": "ExtraTrees", **et_choice},
    ])
    control_candidates.insert(0, "model", "ML-Strong")
    control_candidates.insert(0, "outer_fold", outer_fold)
    hybrid_candidates.insert(0, "model", "WB-PIML-Anchor")
    hybrid_candidates.insert(0, "outer_fold", outer_fold)
    fusion_candidates.insert(0, "model", "WB-PIML")
    fusion_candidates.insert(0, "outer_fold", outer_fold)
    et_candidates.insert(0, "model", "ExtraTrees")
    et_candidates.insert(0, "outer_fold", outer_fold)
    walker_candidates.insert(0, "model", "ET-Walker")
    walker_candidates.insert(0, "outer_fold", outer_fold)
    nested_candidate_parts.extend([
        hybrid_candidates, control_candidates, fusion_candidates, walker_candidates, et_candidates,
    ])

    probability_oof_sources = {}
    for spec_index, physical_spec in enumerate(PHYSICAL_SPECS):
        oof = inner_oof(train, physical_spec, SEED + 10000 * outer_fold + 100 * spec_index)
        scale = calibrate_aft_scale(oof)
        probability_oof_sources[physical_spec.key] = (oof, float(scale["sigma"]),
            SEED + 10000 * outer_fold + 100 * spec_index, {"spec": physical_spec.key, "gamma": GAMMA},
            "deterministic_physical_fit")
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
        kernel_length_scale = float(hybrid_choice["kernel_length_scale"])
        temperature_scale = float(hybrid_choice["temperature_scale"])
        mu, coef, residual_meta = fit_selected_hybrid(
            train, test, hybrid_choice, outer_seed + 1, trees, spec=trunk_spec
        )
        scale = {"sigma": np.nan, "calibration_objective": np.nan,
                 "exact_density_NLL": np.nan, "runout_survival_NLL": np.nan}
        half80 = half90 = np.nan
        part = test[["row_id", "source_id", "campaign_id", "record_equivalence_id",
                     "cross_source_duplicate_candidate", "label_conflict_flag",
                     "logN", "is_exact", "is_runout", "outer_fold"]].copy()
        part["model"] = hybrid_name
        part["temperature_policy"] = str(hybrid_choice["temperature_policy"])
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
                f"max_features={max_features}, ridge_alpha={ridge_alpha}, kernel_length_scale={kernel_length_scale}, "
                f"temperature_scale={temperature_scale})"
            ),
            "eta": eta, "family": hybrid_choice["family"], "strength": residual_meta["strength"],
            "environment_regularization_applicable": residual_meta["environment_regularization_applicable"],
            "residual_mode": residual_mode,
            "ridge_alpha": ridge_alpha, "kernel_length_scale": kernel_length_scale,
            "temperature_scale": temperature_scale,
            "temperature_policy": str(hybrid_choice["temperature_policy"]),
            "active_runout_count": residual_meta["active_runout_count"],
            "active_runout_rate": residual_meta["active_runout_rate"],
            "residual_correction_rms": residual_meta["residual_correction_rms"],
            "exact_temperature_support_rate": residual_meta["exact_temperature_support_rate"],
            "n_exact_temperature_unsupported": residual_meta["n_exact_temperature_unsupported"],
            "all_test_temperature_support_rate": residual_meta["all_test_temperature_support_rate"],
            "n_all_test_temperature_unsupported": residual_meta["n_all_test_temperature_unsupported"],
            "n_residual_temperature_unsupported": residual_meta["n_residual_temperature_unsupported"],
            "empirical_halfwidth_80": half80, "empirical_halfwidth_90": half90,
            **(water_fit_metadata(train,coef) if trunk_spec.mode=="competing_damage" else {}),
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
    proposed_part["temperature_policy"] = str(hybrid_choice["temperature_policy"])
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
        "learner": "regularized competing-damage trunk + OOF-selected residual family and gain; fixed soft250 support",
        "eta": hybrid_choice["eta"], "family": hybrid_choice["family"], "strength": hybrid_choice["strength"],
        "residual_mode": hybrid_choice["residual_mode"],
        "ridge_alpha": hybrid_choice["ridge_alpha"],
        "kernel_length_scale": hybrid_choice["kernel_length_scale"],
        "temperature_scale": hybrid_choice["temperature_scale"],
        "temperature_policy": hybrid_choice["temperature_policy"],
        "censor_feasible": hybrid_choice["censor_feasible"],
        "censor_constraint_relaxed": hybrid_choice["censor_constraint_relaxed"],
        "censor_guard_used_for_selection": False,
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
    strong_mu = fit_selected_control(train, test, control_choice, SEED + 10000 * outer_fold + 1, trees)
    strong_scale = calibrate_aft_scale(control_oof)
    strong_half80, strong_half90 = empirical_interval_halfwidths(control_oof)
    strong_part = test[["row_id", "source_id", "campaign_id", "record_equivalence_id",
        "cross_source_duplicate_candidate", "label_conflict_flag", "logN", "is_exact", "is_runout", "outer_fold"]].copy()
    strong_part["model"], strong_part["pred_logN"], strong_part["sigma"] = "ML-Strong", strong_mu, strong_scale["sigma"]
    for level, half in ((80, strong_half80), (90, strong_half90)):
        strong_part[f"lower{level}"], strong_part[f"upper{level}"] = strong_mu-half, strong_mu+half
    prediction_parts.append(strong_part)
    parameter_rows.append({"outer_fold": outer_fold, "model": "ML-Strong", **control_choice, **strong_scale,
        "learner": control_choice["family"], "active_runout_count": int(train.is_runout.sum()) if control_choice["family"]=="physics_feature_ET" else 0,
        "runouts_used_for_scale_and_selection": True, "n_physics_parameters": 6 if control_choice["family"]=="physics_feature_ET" else 0,
        "empirical_halfwidth_80": strong_half80, "empirical_halfwidth_90": strong_half90})
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

    probability_oof_sources[PROPOSED_MODEL] = (proposed_oof, float(proposed_scale["sigma"]),
        selection_seed, {"hybrid": hybrid_choice, "fusion": fusion_choice}, "split_seed+100*inner_fold+1")
    probability_oof_sources["ML-Strong"] = (control_oof, float(strong_scale["sigma"]),
        selection_seed, control_choice, "split_seed+100*inner_fold+1")
    simple_results = fit_simple_control_models(
        train, test, selection_seed, SEED + 10000 * outer_fold + 1, trees
    )
    for name, result in simple_results.items():
        part = simple_control_prediction_part(test, proposed_part.columns, name, result)
        prediction_parts.append(part)
        nested_selection_rows.append({"outer_fold": outer_fold, "model": name,
                                      **result["choice"]})
        candidate_table = result["candidates"].copy()
        candidate_table.insert(0, "outer_fold", outer_fold)
        nested_candidate_parts.append(candidate_table)
        parameter_rows.append({"outer_fold": outer_fold, "model": name,
            "n_location_parameters": np.nan, "n_physics_parameters": 2,
            "learner": "censored two-parameter stress-life trunk + selected residual",
            **result["choice"], **result["metadata"], **result["scale"],
            **physical_parameter_values(CLASSIC_WB_SPEC, result["coef"]),
            "empirical_halfwidth_80": result["half80"],
            "empirical_halfwidth_90": result["half90"]})
        probability_oof_sources[name] = (
            result["oof"], float(result["scale"]["sigma"]), selection_seed,
            result["choice"], "split_seed+100*inner_fold+1"
        )
    probability_parts, probability_calibration_parts = main_probability_records(
        train, prediction_parts, probability_oof_sources, censored_external,
        et_choice, walker_choice, selection_seed, trees, outer_fold)
    return (prediction_parts, parameter_rows, audit_rows, nested_selection_rows, nested_candidate_parts,
            probability_parts, probability_calibration_parts)


def _execute_task(function, arguments, tree_jobs, blas_threads):
    global WORKERS, TREE_JOBS, BLAS_THREADS
    WORKERS, TREE_JOBS, BLAS_THREADS = 1, tree_jobs, blas_threads
    with threadpool_limits(limits=blas_threads):
        return function(*arguments)


def _parallel_map(function, tasks, label):
    workers = min(WORKERS, len(tasks))
    if not tasks:
        return
    print(f"{label}: {len(tasks)} tasks, {workers} workers", flush=True)
    if workers == 1:
        for index, arguments in enumerate(tasks, 1):
            result = function(*arguments)
            print(f"{label}: {index}/{len(tasks)} complete", flush=True)
            yield result
    else:
        with parallel_config(backend="loky", inner_max_num_threads=BLAS_THREADS):
            results = Parallel(n_jobs=workers, return_as="generator")(
                delayed(_execute_task)(function, arguments, TREE_JOBS, BLAS_THREADS)
                for arguments in tasks
            )
            for index, result in enumerate(results, 1):
                print(f"{label}: {index}/{len(tasks)} complete", flush=True)
                yield result


def environmental_decrement(frame, coefficients, gamma):
    _, _, log_k, stress_exponent, temperature_exponent = np.asarray(coefficients, float)[:5]
    exposure = frame["environment_exposure"].to_numpy(float)
    active = exposure > 0.0
    decrement = np.zeros(len(frame), dtype=float)
    log_sw = walker_log_stress(frame, gamma)
    log_ratio = (
        log_k + np.log10(exposure[active]) + water_log_adjustment(frame,coefficients)[active]
        + temperature_exponent * frame.loc[active, "arrhenius_temperature_drive"].to_numpy(float) / math.log(10.0)
        + stress_exponent * log_sw[active] - frame.loc[active, "logf"].to_numpy(float)
    )
    decrement[active] = np.logaddexp(0.0, math.log(10.0) * log_ratio) / math.log(10.0)
    return decrement


def kernel_predict(x, xv, y, length_scale, alpha, centered):
    mean = float(np.mean(y)) if centered else 0.0
    kernel = Matern(length_scale=length_scale, nu=1.5)
    matrix = kernel(x)
    matrix.flat[::len(x) + 1] += alpha
    weights = cho_solve(cho_factor(matrix, lower=True, check_finite=False), y - mean, check_finite=False)
    return mean + kernel(xv, x) @ weights


def predict_structural_configs(train, valid, seed, configs, trees=400):

    require_exact_training(train, "predict_structural_configs")
    exact = train.loc[train.is_exact].reset_index(drop=True)
    y = exact.logN.to_numpy(float)
    prep = make_preprocessor()
    x = prep.fit_transform(exact)
    xv = prep.transform(valid)
    residual_prep = make_residual_preprocessor()
    xr = residual_prep.fit_transform(exact)
    xrv = residual_prep.transform(valid)
    support = residual_policy_weight(train, valid, "ET_residual", "soft250")
    physics, fitted, out = {}, {}, {}
    for c in configs:
        if "gamma" in c:
            key = (c["gamma"], c["strength"])
            if key not in physics:
                coef = fit_physical(train, COMPETING_WB_SPEC, c["gamma"],
                    replace(DEFAULT_COMPETING_CONFIG, environment_strength=c["strength"]))
                full = pd.concat([exact, valid], ignore_index=True)
                base = predict_physical(full, COMPETING_WB_SPEC, coef, c["gamma"])
                mechanical = coef[0] - coef[1] * walker_log_stress(full, c["gamma"])
                aug = full.copy()
                aug["mechanical_prediction"] = mechanical
                aug["environment_decrement"] = mechanical - base
                aug["walker_stress"] = walker_log_stress(full, c["gamma"])
                aug_prep = make_preprocessor(("mechanical_prediction", "environment_decrement", "walker_stress"))
                xa = aug_prep.fit_transform(aug.iloc[:len(exact)])
                xav = aug_prep.transform(aug.iloc[len(exact):])
                residual_aug_prep = make_residual_preprocessor(RESIDUAL_PHYSICS_NUMERIC)
                xra = residual_aug_prep.fit_transform(aug.iloc[:len(exact)])
                xrav = residual_aug_prep.transform(aug.iloc[len(exact):])
                physics[key] = (base[:len(exact)], base[len(exact):], xa, xav, xra, xrav)
            base_train, base_valid, xa, xav, xra, xrav = physics[key]
        family = c["family"]
        fit_key = tuple(sorted((k, v) for k, v in c.items() if k not in ("candidate_id", "eta", "pool")))
        if fit_key not in fitted:
            if c["pool"] == "hybrid":
                target = y - base_train
                if family == "Matern_residual":
                    correction = kernel_predict(xr, xrv, target,
                        float(c.get("kernel_length_scale", 2.0)), float(c.get("ridge_alpha", 1.0)), False)
                else:
                    xx, xxv = (xra, xrav) if family == "physics_feature_ET_residual" else (xr, xrv)
                    estimator = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=3, max_features=.85, random_state=seed, n_jobs=TREE_JOBS)
                    correction = estimator.fit(xx, target).predict(xxv)
                fitted[fit_key] = correction
            elif family in ("ET", "physics_feature_ET"):
                xx, xxv = (xa, xav) if family == "physics_feature_ET" else (x, xv)
                estimator = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=c.get("leaf", 3), max_features=c.get("max_features", .85), random_state=seed, n_jobs=TREE_JOBS)
                fitted[fit_key] = estimator.fit(xx, y).predict(xxv)
            elif family == "GBDT":
                estimator = GradientBoostingRegressor(n_estimators=300, learning_rate=c["learning_rate"], max_depth=c["depth"], min_samples_leaf=c["leaf"], loss="squared_error", random_state=seed)
                fitted[fit_key] = estimator.fit(x, y).predict(xv)
            elif family == "Matern":
                fitted[fit_key] = kernel_predict(x, xv, y, c["length_scale"], c["alpha"], True)
            else:
                fitted[fit_key] = external_predictions(train, valid, seed, trees)["ML-Ens"]
        prediction = fitted[fit_key]
        if c["pool"] == "hybrid":
            prediction = base_valid + c["eta"] * support * prediction
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f"Nonfinite predictions: candidate {c['candidate_id']}")
        out[c["candidate_id"]] = prediction
    return out


def fit_selected_control(train, valid, choice, seed, trees):
    candidate_id = int(choice["candidate_id"])
    config = next(c for c in CONTROL_CANDIDATES if c["candidate_id"] == candidate_id)
    return predict_structural_configs(train, valid, seed, [config], trees)[candidate_id]


def nested_select_structural(train, seed, trees):
    require_exact_training(train, "nested_select_structural")
    splits, _ = grouped_folds(train, min(4, train[PRIMARY_GROUP_COLUMN].nunique()), seed)
    reference = inner_oof(train, COMPETING_WB_SPEC, seed)
    reference_scale = calibrate_aft_scale(reference)
    reference_runout = float(reference_scale["runout_survival_NLL"])
    reference_fold_loss = _runout_fold_losses(reference, float(reference_scale["sigma"]))
    columns = ["row_id", "source_id", "campaign_id", "record_equivalence_id", "logN", "is_exact", "is_runout", "temp_bin"]
    stored = {c["candidate_id"]: [] for c in STRUCTURAL_CANDIDATES}
    for inner_fold, (fit_idx, valid_idx) in enumerate(splits, 1):
        fit = train.iloc[fit_idx].reset_index(drop=True)
        valid = train.iloc[valid_idx].reset_index(drop=True)
        if set(fit.campaign_id) & set(valid.campaign_id):
            raise AssertionError("Inner campaign leakage")
        predictions = predict_structural_configs(fit, valid, seed + 100 * inner_fold + 1, STRUCTURAL_CANDIDATES, trees)
        for cid, prediction in predictions.items():
            part = valid[columns].copy()
            part["inner_fold"], part["mu"] = inner_fold, prediction
            stored[cid].append(part)
    rows = []
    for c in STRUCTURAL_CANDIDATES:
        cid = c["candidate_id"]
        oof = pd.concat(stored[cid], ignore_index=True)
        stored[cid] = oof
        if len(oof) != len(train) or not oof.row_id.is_unique:
            raise AssertionError("Every training row must have exactly one OOF prediction")
        exact = oof.loc[oof.is_exact].copy()
        exact["pred_logN"] = exact.mu
        score = sb_metrics([g.reset_index(drop=True) for _, g in exact.groupby(PRIMARY_GROUP_COLUMN)])
        scale = calibrate_aft_scale(oof)
        fold_loss = _runout_fold_losses(oof, float(scale["sigma"]))
        paired = pd.concat([fold_loss.rename("candidate"), reference_fold_loss.rename("reference")], axis=1, join="inner").dropna()
        margin = float((paired.candidate-paired.reference).std(ddof=1)/math.sqrt(len(paired))) if len(paired)>1 else 0.0
        runout_loss = float(scale["runout_survival_NLL"])
        feasible = bool(not train.is_runout.any() or runout_loss <= reference_runout + margin + 1e-12)
        rows.append({**c, "candidate_index": cid, "basis_id": cid//2 if c["pool"]=="hybrid" else cid,
            "min_samples_leaf": int(c.get("leaf", 3)), "max_features": float(c.get("max_features", .85)),
            "residual_mode": c["family"],
            "ridge_alpha": float(c.get("ridge_alpha", c.get("alpha", 1.0))),
            "kernel_length_scale": float(c.get("kernel_length_scale", c.get("length_scale", 2.0))),
            "kernel_parameters_applicable": c["family"] in ("Matern_residual", "Matern"),
            "temperature_scale": 250.0,
            "temperature_policy": "soft250" if c["pool"]=="hybrid" else "not_applicable", "is_reference": False,
            "inner_SB_RMSE": score["SB_RMSE"], "inner_runout_hinge_RMSE": censored_hinge_rmse(oof),
            "inner_joint_censored_NLL": float(scale["joint_censored_NLL"]),
            "inner_runout_survival_NLL": runout_loss, "inner_sigma": float(scale["sigma"]),
            "reference_WB_CD_runout_NLL": reference_runout, "runout_noninferiority_margin": margin,
            "censor_feasible": feasible, "inner_basis_seed_offset": 0})
    table = pd.DataFrame(rows)
    outputs = []
    for pool in ("hybrid", "control"):
        choice, candidates = select_budgeted_candidate(table.loc[table.pool.eq(pool)])
        outputs.extend((choice, stored[choice["candidate_index"]], candidates))
    return tuple(outputs)


def _run_analysis(args):
    data_path = resolve_data(args.data)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    full_df = load_data(data_path)
    df = full_df.reset_index(drop=True)
    exact_df = df.loc[df["is_exact"]].reset_index(drop=True)
    runout_df = full_df.loc[full_df["is_runout"]].reset_index(drop=True)
    require_exact_training(df, "main joint-likelihood analysis frame")
    print(f"Records: {len(df)} ({len(exact_df)} failures, {len(runout_df)} runouts); "
          f"workers={WORKERS}, forest threads={TREE_JOBS}, BLAS threads={BLAS_THREADS}", flush=True)
    if (
        len(full_df) != EXPECTED_VERIFIED_ROWS
        or len(exact_df) != EXPECTED_EXACT_ROWS
        or len(runout_df) != EXPECTED_RUNOUT_ROWS
    ):
        raise ValueError(
            f"Plot_Data_Verified must contain {EXPECTED_VERIFIED_ROWS} analysis records: "
            f"{EXPECTED_EXACT_ROWS} exact failures and {EXPECTED_RUNOUT_ROWS} runouts; "
            f"found {len(full_df)} total, {len(exact_df)} exact and {len(runout_df)} runout rows"
        )
    if not np.allclose(training_weights(df), 1.0):
        raise AssertionError("Every unique likelihood record must have unit weight")
    outer_splits, outer_id = grouped_folds(df, 5, SEED)
    df["outer_fold"] = outer_id
    prediction_parts, parameter_rows, fold_metric_rows, audit_rows = [], [], [], []
    nested_selection_rows, nested_candidate_parts = [], []
    probability_parts, probability_calibration_parts = [], []

    tasks = [(df, fold, train_idx, test_idx, args.trees)
             for fold, (train_idx, test_idx) in enumerate(outer_splits, 1)]
    for result in _parallel_map(_outer_fold_task, tasks, "Primary validation"):
        for target, rows in zip((prediction_parts, parameter_rows, audit_rows,
                                 nested_selection_rows, nested_candidate_parts,
                                 probability_parts, probability_calibration_parts), result):
            target.extend(rows)

    probability_predictions = pd.concat(probability_parts, ignore_index=True, sort=False)
    probability_calibration = pd.concat(probability_calibration_parts, ignore_index=True, sort=False)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions["T_C"] = predictions["row_id"].map(
        df.set_index("row_id")["T_C"]
    ).to_numpy(float)
    expected_models = ([s.key for s in PHYSICAL_SPECS]
                       + [name for name, _ in HYBRID_SPECS]
                       + [PROPOSED_MODEL]
                       + list(EXTERNAL_MODELS) + list(CENSORED_MODELS) + list(SIMPLE_CONTROL_MODELS))
    for model in expected_models:
        model_rows = predictions.loc[predictions["model"].eq(model)]
        if len(model_rows) != len(df) or model_rows["row_id"].duplicated().any():
            raise AssertionError(f"{model} must predict every record exactly once")

    temperature_gate_rows = predictions.loc[
        predictions["model"].eq(PROPOSED_MODEL),
        ["outer_fold", "row_id", "source_id", "campaign_id",
         "record_equivalence_id", "T_C", "is_exact", "is_runout",
         "residual_temperature_supported", "logN", "pred_logN"],
    ].copy()
    temperature_gate_summary = (
        temperature_gate_rows.groupby(
            ["outer_fold", "T_C", "is_exact", "is_runout",
             "residual_temperature_supported"],
            dropna=False,
        )
        .size().rename("n_rows").reset_index()
        .sort_values(["outer_fold", "T_C", "is_runout"])
    )

    for (fold, model), group in predictions.loc[predictions["is_exact"]].groupby(["outer_fold", "model"]):
        groups = [g.reset_index(drop=True) for _, g in group.groupby(PRIMARY_GROUP_COLUMN)]
        fold_metric_rows.append({"outer_fold": fold, "model": model, "n_exact": len(group), **sb_metrics(groups)})

    print("Bootstrap confidence intervals", flush=True)
    metrics = metric_summary(predictions, args.bootstrap, SEED + 1)
    effects = pd.concat([
        paired_effects(predictions, args.bootstrap, SEED + 20 + index, comparator=model)
        for index, model in enumerate(EXTERNAL_MODELS + SIMPLE_CONTROL_MODELS)
    ], ignore_index=True)
    uq = uq_summary(predictions, args.bootstrap, SEED + 3)
    runout_predictions = predictions.loc[predictions["is_runout"]].copy()
    runout_predictions["evaluation_scope"] = "campaign_disjoint_outer_test_right_censored"
    runout_predictions["used_for_training"] = runout_predictions["model"].isin(
        [s.key for s in PHYSICAL_SPECS] + [name for name, _ in HYBRID_SPECS]
        + [PROPOSED_MODEL] + list(CENSORED_MODELS) + list(SIMPLE_CONTROL_MODELS)
    )
    runout_predictions["used_for_candidate_selection"] = (
        runout_predictions["model"].isin(
            ["WB-PIML-Anchor", PROPOSED_MODEL, "ML-Strong"] + list(CENSORED_MODELS) + list(SIMPLE_CONTROL_MODELS)
        )
    )
    runout_predictions["used_as_censor_feasibility_reference"] = (
        runout_predictions["model"].eq("WB-CD")
    )
    control_location_folds = {int(row["outer_fold"]): row["family"] == "physics_feature_ET"
                              for row in nested_selection_rows if row["model"] == "ML-Strong"}
    control_rows = runout_predictions["model"].eq("ML-Strong")
    runout_predictions.loc[control_rows, "used_for_training"] = runout_predictions.loc[control_rows, "outer_fold"].map(control_location_folds).astype(bool)
    runout_predictions["used_for_selection"] = (
        runout_predictions["used_for_candidate_selection"]
        | runout_predictions["used_as_censor_feasibility_reference"]
    )
    runout = runout_summary(runout_predictions)
    parameters = pd.DataFrame(parameter_rows)
    parameters["damage_coefficient_sum"] = parameters[["M_coefficient", "Ox_coefficient", "Q_coefficient"]].fillna(0).sum(axis=1)
    parameters["damage_boundary_hit"] = parameters["damage_coefficient_sum"] >= DAMAGE_CAP - 1e-5
    parameters["walker_slope_boundary_hit"] = parameters["walker_slope"] <= 0.05 + 1e-5
    parameters["environment_ratio_boundary_hit"] = (
        np.isclose(pd.to_numeric(parameters["environment_log10_damage_ratio"], errors="coerce"), -8.0, atol=1e-5)
        | np.isclose(pd.to_numeric(parameters["environment_log10_damage_ratio"], errors="coerce"), 4.0, atol=1e-5)
    )
    parameters["environment_stress_boundary_hit"] = (
        np.isclose(pd.to_numeric(parameters["environment_stress_exponent"], errors="coerce"), 0.0, atol=1e-5)
        | np.isclose(pd.to_numeric(parameters["environment_stress_exponent"], errors="coerce"), 10.0, atol=1e-5)
    )
    parameters["arrhenius_boundary_hit"] = (
        np.isclose(pd.to_numeric(parameters["arrhenius_temperature_sensitivity"], errors="coerce"), 0.0, atol=1e-5)
        | np.isclose(pd.to_numeric(parameters["arrhenius_temperature_sensitivity"], errors="coerce"), 10.0, atol=1e-5)
    )
    parameters["competing_damage_boundary_hit"] = parameters[[
        "environment_ratio_boundary_hit", "environment_stress_boundary_hit",
        "arrhenius_boundary_hit",
    ]].any(axis=1)
    audit = pd.DataFrame(audit_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    ablation = ablation_effects(predictions, args.bootstrap, SEED + 4)
    source_gains = source_gain_table(predictions)
    nested_selection = pd.DataFrame(nested_selection_rows)
    nested_candidates = pd.concat(nested_candidate_parts, ignore_index=True, sort=False)
    print("Sensitivity analyses", flush=True)
    sensitivity = sensitivity_audit(
        df, outer_splits, args.audit_trees, nested_selection
    )
    component_ablation = component_ablation_summary(sensitivity)
    selection_sensitivity = selection_rule_sensitivity(nested_candidates, nested_selection)
    repeated_metrics, validation_summary, validation_assignments = repeated_validation(
        df, args.trees, args.repeats
    )
    (
        external_summary,
        external_groups,
        external_holdout_predictions,
        external_selection,
        external_probability_predictions,
        external_probability_calibration,
    ) = external_holdout_validation(df, args.trees)
    external_selection, simple_external_candidates = extract_simple_external_candidates(external_selection)
    external_effects = external_paired_effects(
        external_holdout_predictions, args.bootstrap, SEED + 930000
    )
    external_uq = external_uq_summary(
        external_holdout_predictions, args.bootstrap, SEED + 940000
    )
    external_pi_rows = external_holdout_predictions.loc[
        external_holdout_predictions["model"].eq(PROPOSED_MODEL)
        & external_holdout_predictions["is_exact"]
    ].copy()
    external_temperature_gate_summary = (
        external_holdout_predictions.loc[
            external_holdout_predictions["model"].eq(PROPOSED_MODEL)
        ]
        .groupby(
            ["strategy", "holdout_group", "fold", "T_C", "is_exact",
             "is_runout", "residual_temperature_supported"],
            dropna=False,
        )
        .size().rename("n_rows").reset_index()
        .sort_values(["strategy", "fold", "T_C", "is_runout"])
    )
    gate = claim_gate(
        metrics, effects, parameters, df, nested_selection, nested_candidates,
        runout, external_summary, external_groups, external_effects, external_uq,
        external_selection,
        temperature_gate_rows,
    )
    print("Source and stress sensitivity checks", flush=True)
    sstar_sensitivity = sstar_exclusion_sensitivity(df, args.trees)
    duplicate_sensitivity = duplicate_source_sensitivity(df, args.trees)
    duplicate_clusters, duplicate_candidates, campaign_map = campaign_audit_tables(df)
    label_review = full_df.loc[
        full_df["supplied_label_conflict_flag"] | full_df["label_override_applied"],
        ["row_id", "source_row_no", "source_id", "record_equivalence_id",
         "sigma_max_MPa_before_normalization", "Nf_cycles", "raw_Nf",
         "supplied_is_exact_failure", "supplied_is_runout",
         "reviewed_is_exact_failure", "is_runout", "label_override_applied",
         "label_review_status", "label_review_evidence"],
    ].copy().sort_values(["record_equivalence_id", "source_id", "row_id"])
    label_review["evidence_file"] = (
        f"{data_path.name} / Plot_Data_Verified: evidence_source and evidence_locator"
    )

    qa_flagged = df.loc[df["qa_sstar_gt_1"] | df["qa_exact_round_limit"], [
        "row_id", "source_id", "architecture_detail", "env_class", "T_C", "R",
        "frequency_Hz", "Nf_cycles", "S", "is_exact", "is_runout",
        "qa_sstar_gt_1", "qa_exact_round_limit",
    ]].copy()
    qa_flagged["action"] = np.where(
        qa_flagged["qa_exact_round_limit"],
        "verify against original source; main analysis uses the reviewed label recorded in Label_Review",
        "retained in main analysis; compare with S*>1 exclusion sensitivity",
    )
    validation_definitions = pd.DataFrame([
        ("record_interpolation", "rows stratified across five folds; source/series overlap is disclosed",
         "secondary interpolation diagnostic; not evidence for unseen-source prediction"),
        ("series_disjoint", "series = source + detailed architecture + environment + T + R + UTS; series never overlaps",
         "primary in-domain grouped validation when explicit batch identifiers are unavailable"),
        ("source_disjoint", "publication source never overlaps; ten repeated five-fold assignments",
         "secondary comparison; may split publications from one experimental campaign"),
        ("campaign_disjoint", "human-reviewed source-to-campaign map never overlaps; ten repeated five-fold assignments",
         "fully campaign-disjoint outer validation within the compiled database"),
        ("LOSO", "one complete source held out at a time; another source from the same campaign may remain in development",
         "source-held-out stress test with training-only nested selection; source isolation does not imply campaign isolation, and actual overlap is reported"),
        ("LOAO", "one weaving-architecture group held out at a time",
         "weaving-architecture-held-out stress test with training-only nested selection; campaign isolation is not imposed by this split, and actual overlap is reported"),
        ("LOTO", "one reviewed temperature regime held out at a time",
         "temperature-regime-held-out stress test with training-only nested selection; campaign isolation is not imposed by this split, and actual overlap is reported"),
        ("LOEO", "one environment class held out at a time",
         "environment-held-out stress test with training-only nested selection; campaign isolation is not imposed by this split, and actual overlap is reported"),
    ], columns=["strategy", "split_definition", "interpretation"])

    proposed_exact = predictions.loc[
        predictions["model"].eq("WB-PIML") & predictions["is_exact"]
    ].copy()
    for level in (80, 90):
        proposed_exact[f"covered{level}"] = (
            proposed_exact["logN"].between(proposed_exact[f"lower{level}"], proposed_exact[f"upper{level}"])
        )
        proposed_exact[f"width{level}"] = proposed_exact[f"upper{level}"] - proposed_exact[f"lower{level}"]

    model_specs = pd.DataFrame([
        {"model": "WB", "role": "right-censored lognormal AFT physics baseline", "location_parameters": 2,
         "equation": "mu=a-b*log10(S_Walker); exact density + runout survival", "used_for_selection": False},
        {"model": "WB-CD", "role": 'right-censored competing-damage physical reference', "location_parameters": 6,
         "equation": 'mu=-log10(D_mech+D_env); exact density+runout survival; six location parameters; water enhancement estimated within training only',
         "used_for_selection": True},
        {"model": "WB-M", "role": "mechanical-proxy ablation", "location_parameters": 3,
         "equation": "a - b*log10(S_Walker) - c*M", "used_for_selection": False},
        {"model": "WB-Ox", "role": "oxidation-proxy ablation", "location_parameters": 3,
         "equation": "a - b*log10(S_Walker) - c*Dox", "used_for_selection": False},
        {"model": "WB-PIML-Anchor", "role": 'development-selected competing-damage residual branch', "location_parameters": 6,
         "equation": 'mu_phys+eta*soft250*residual(X_without_environment); ExtraTrees, mechanically augmented ExtraTrees or Matérn; shared residual preserves physical-trunk environment ordering', "used_for_selection": True},
        {"model": "WB-PIML", "role": 'reported always-physical censored-AFT hybrid', "location_parameters": 6,
         "equation": 'mu_phys+eta*soft250*residual(X); 54 development-only candidates; lowest runout NLL within inner RMSE+0.02 budget', "used_for_selection": True},
        {"model": "WB-PIML-M", "role": 'mechanical-anchor residual ablation', "location_parameters": 3,
         "equation": 'mu_WB-M+eta*soft250*residual(X); retrained with primary-selected residual settings', "used_for_selection": False},
        {"model": "WB-PIML-Ox", "role": 'oxidation-anchor residual ablation', "location_parameters": 3,
         "equation": 'mu_WB-Ox+eta*soft250*residual(X); retrained with primary-selected residual settings', "used_for_selection": False},
        {"model": "ExtraTrees", "role": "external traditional comparator",
          "location_parameters": np.nan, "equation": "same observable inputs; never replaces the physical trunk",
          "used_for_selection": True},
        {"model": "ET-Walker", "role": "Walker-transformed ExtraTrees comparator",
         "location_parameters": np.nan,
         "equation": "ExtraTrees[X, selected log10(S_Walker)]; same campaign folds, separate 12-configuration gamma/tree search",
         "used_for_selection": True},
        {"model": "RF / GBDT / SVR", "role": "fixed external traditional comparators",
         "location_parameters": np.nan, "equation": "same observable inputs; not used inside WB-PIML",
         "used_for_selection": False},
        {"model": "ML-Ens", "role": "external traditional ensemble comparator",
         "location_parameters": np.nan,
         "equation": "mean of fixed RF/GBDT/SVR and nested-selected ExtraTrees predictions",
         "used_for_selection": False},
        {"model": "Lognormal-AFT", "role": "fair observable-input right-censored comparator",
         "location_parameters": np.nan,
         "equation": "penalized AFT location; exact normal density + runout survival",
         "used_for_selection": True},
        {"model": "Weibull-AFT", "role": "fair observable-input right-censored comparator",
         "location_parameters": np.nan,
         "equation": "penalized Weibull AFT; exact density + runout survival",
         "used_for_selection": True},
        {'model':'ML-Strong','role':'development-selected predictive control','location_parameters':np.nan,'equation':'64 candidates: 18 ExtraTrees, 18 GBDT, 18 Matérn, one fixed ML ensemble and nine direct ExtraTrees models with physical features; physical coefficients are fitted on development records only','used_for_selection':True},
    ])

    model_specs = pd.concat([model_specs, pd.DataFrame([
        {"model": name, "role": "independently development-selected simple-trunk control",
         "location_parameters": 2, "used_for_selection": True,
         "equation": ("a-b*log10(S_Walker)+eta*soft250*residual(X); 18 candidates"
                      if name == "WB-Residual" else
                      "a-b*log10(S*)+eta*soft250*residual(X); gamma=0; 6 candidates")}
        for name in SIMPLE_CONTROL_MODELS
    ])], ignore_index=True)

    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    data_audit = pd.DataFrame([
        ("input_file", data_path.name), ("sheet", SHEET), ("sha256", data_hash),
        ("n_verified_rows", len(full_df)),
        ("n_exact_density_training_and_evaluation_rows", len(exact_df)),
        ("n_right_censored_likelihood_rows", len(runout_df)),
        ("n_workbook_exact_fractures", int(full_df["is_exact"].sum())),
        ("n_workbook_runouts", int(full_df["is_runout"].sum())),
        ("n_unique_record_equivalence_ids", int(full_df["record_equivalence_id"].nunique())),
        ("duplicate_equivalence_ids_in_verified", bool(full_df["record_equivalence_id"].duplicated().any())),
        ("exact_training_weight_min", f"{float(exact_df['sample_weight'].min()):.1f}"),
        ("exact_training_weight_max", f"{float(exact_df['sample_weight'].max()):.1f}"),
        ("runout_training_weight_min", f"{float(runout_df['sample_weight'].min()):.1f}"),
        ("runout_training_weight_max", f"{float(runout_df['sample_weight'].max()):.1f}"),
        ("n_code_side_label_overrides", int(full_df["label_override_applied"].sum())),
        ("n_sources", int(full_df["source_id"].nunique())),
        ("n_exact_sources", int(exact_df["source_id"].nunique())),
        ("n_campaigns_all_verified", int(full_df["campaign_id"].nunique())),
        ("n_campaigns_exact_analysis", int(exact_df["campaign_id"].nunique())),
        ("n_series_groups", int(df["series_group_id"].nunique())),
        ("n_physical_duplicate_clusters", len(duplicate_clusters)),
        ("n_cross_source_duplicate_clusters", int(duplicate_clusters["n_sources"].gt(1).sum())),
        ("n_duplicate_traceability_rows", len(duplicate_candidates)),
        ("n_cross_source_duplicate_exact", int(
            duplicate_candidates.loc[
                duplicate_candidates["cross_source_duplicate_candidate"], "is_exact"
            ].sum()
        )),
        ("n_supplied_label_conflict_clusters", int(duplicate_clusters["supplied_label_conflict"].sum())),
        ("n_reviewed_label_conflict_clusters", int(duplicate_clusters["reviewed_label_conflict"].sum())),
        ("n_supplied_label_conflict_rows", int(full_df["supplied_label_conflict_flag"].sum())),
        ("n_reviewed_label_conflict_rows", int(full_df["label_conflict_flag"].sum())),
        ("n_reviewed_secondary_sources", int(full_df.loc[full_df["reviewed_secondary_source"], "source_id"].nunique())),
        ("label_consistency_all_rows", bool(full_df["label_consistent"].all())),
        ("duplicate_clusters_cross_outer_folds", int(duplicate_clusters["n_outer_folds"].gt(1).sum())),
        ("n_Sstar_gt_1", int(df["qa_sstar_gt_1"].sum())),
        ("n_exact_round_limit_flags", int(df["qa_exact_round_limit"].sum())),
        ("missing_required_values", 0), ("duplicate_row_id", False),
        ("runouts_enter_preprocessor_fit", False),
        ("runouts_enter_location_training", True),
        ("runouts_enter_hyperparameter_selection", True),
        ("runouts_enter_calibration", True),
        ("runouts_enter_point_error_metrics", False), ("random_seed", SEED),
        ("row_or_source_dependent_training_weights", False),
    ], columns=["item", "value"])

    references = pd.DataFrame([
        ("C/SiC fatigue data and source mapping", "exact/runout labels and source-campaign mapping",
         f"{data_path.name} / Plot_Data_Verified, Plot_Data and Plot_Data_Original"),
        ("Walker mean-stress correction", "physics trunk", "https://doi.org/10.1115/1.4001673"),
        ("Physics-informed feature engineering for fatigue", "residual-hybrid context", "https://www.mdpi.com/2076-3417/16/13/6493"),
        ("Accelerated failure-time modelling", "right-censoring likelihood audit", "https://academic.oup.com/biomet/article-abstract/79/2/311/225970"),
        ("Ceramic-matrix-composite fatigue mechanisms", "mechanism-informed proxy design", "https://journals.sagepub.com/doi/10.1177/14644207211008574"),
        ("High-temperature oxidation of SiC/SiC composites", "discussion context; no thresholded oxidation feature in the primary model", "https://www.mdpi.com/1996-1944/9/3/207"),
        ("Oxidation-dependent CMC fatigue life", "environment-fatigue coupling", "https://www.sciencedirect.com/science/article/pii/S0921509315300575"),
        ("Physics-constrained ML for CMCs", "advanced-method context", "https://www.sciencedirect.com/science/article/pii/S1359836825007310"),
    ], columns=["reference_topic", "use_in_analysis", "url_or_local_source"])

    outer_fold_assignment = df[
        ["row_id", "source_id", "campaign_id", "record_equivalence_id",
         "series_group_id", "cross_source_duplicate_candidate",
         "supplied_label_conflict_flag", "label_conflict_flag",
         "source_campaign_role", "campaign_mapping_status",
         "supplied_is_exact_failure", "supplied_is_runout",
         "is_exact", "is_runout", "label_override_applied", "outer_fold"]
    ].copy()
    if args.export_csv:
        csv_tables = {
            "predictions.csv": predictions,
            "campaign_balanced_metrics.csv": metrics,
            "fold_metrics.csv": fold_metrics,
            "primary_paired_effects.csv": effects,
            "empirical_interval_summary.csv": uq,
            "empirical_interval_rows.csv": proposed_exact,
            "temperature_gate_rows.csv": temperature_gate_rows,
            "temperature_gate_summary.csv": temperature_gate_summary,
            "runout_scale_audit.csv": runout,
            "runout_outer_predictions.csv": runout_predictions,
            "physical_parameters_and_scale.csv": parameters,
            "validation_audit.csv": audit,
            "retrained_ablation_effects.csv": ablation,
            "fixed_sensitivity_audit.csv": sensitivity,
            "component_ablation.csv": component_ablation,
            "selection_rule_sensitivity.csv": selection_sensitivity,
            "source_level_gains.csv": source_gains,
            "claim_gate.csv": gate,
            "outer_fold_assignment.csv": outer_fold_assignment,
            "nested_selection.csv": nested_selection,
            "nested_candidates.csv": nested_candidates,
            "validation_tier_metrics.csv": repeated_metrics,
            "validation_tier_summary.csv": validation_summary,
            "external_holdout_summary.csv": external_summary,
            "external_holdout_groups.csv": external_groups,
            "external_selection.csv": external_selection,
            "external_paired_effects.csv": external_effects,
            "external_interval_summary.csv": external_uq,
            "external_interval_rows.csv": external_pi_rows,
            "external_temperature_gate_summary.csv": external_temperature_gate_summary,
            "sstar_sensitivity.csv": sstar_sensitivity,
            "duplicate_source_sensitivity.csv": duplicate_sensitivity,
            "duplicate_clusters.csv": duplicate_clusters,
            "duplicate_candidates.csv": duplicate_candidates,
            "campaign_map.csv": campaign_map,
            "label_review.csv": label_review,
            "qa_flagged_rows.csv": qa_flagged,
        }
        for filename, table in csv_tables.items():
            table.to_csv(out / filename, index=False)
    external_wb_selection = external_selection.loc[
        external_selection["model"].eq(PROPOSED_MODEL)
    ].copy()
    external_relaxed_flags = pd.to_numeric(
        external_wb_selection["censor_constraint_relaxed"], errors="coerce"
    )
    external_censor_status_known = bool(
        len(external_wb_selection) > 0
        and external_relaxed_flags.notna().all()
        and external_relaxed_flags.isin([0, 1]).all()
    )
    external_campaign_overlap_flags = external_wb_selection[
        "campaign_overlap"
    ].astype(bool)

    protocol = {
        "model": "WB-PIML", "data": data_path.name, "sheet": SHEET,
        "data_sha256": data_hash, "n_verified_rows": len(full_df),
        "n_exact_density_training_evaluation": len(exact_df), "n_right_censored_training": len(runout_df),
        "n_label_overrides_in_code": 0,
        "n_sources": int(full_df["source_id"].nunique()), "n_exact_sources": int(exact_df["source_id"].nunique()),
        "n_campaigns": int(df["campaign_id"].nunique()),
        "n_physical_duplicate_clusters": len(duplicate_clusters),
        "n_cross_source_duplicate_clusters": int(duplicate_clusters["n_sources"].gt(1).sum()),
        "n_duplicate_traceability_rows": len(duplicate_candidates),
        "n_supplied_label_conflict_clusters": int(duplicate_clusters["supplied_label_conflict"].sum()),
        "n_reviewed_label_conflict_clusters": int(duplicate_clusters["reviewed_label_conflict"].sum()),
        "outer_folds": len(outer_splits),
        "candidate_gamma": sorted({float(item["gamma"]) for item in HYBRID_CANDIDATES}),
        "candidate_min_samples_leaf": [3],
        "candidate_max_features": [0.85],
        "candidate_eta": sorted({float(item["eta"]) for item in HYBRID_CANDIDATES}),
        "candidate_residual_modes": list(HYBRID_FAMILIES),
        "candidate_temperature_policies": list(TEMPERATURE_POLICIES),
        "candidate_residual_basis_count": len(HYBRID_BASES),
        "candidate_nonzero_eta": list(RESIDUAL_GAIN_GRID),
        "hybrid_candidate_count": len(HYBRID_CANDIDATES),
        "censor_diagnostic_used_for_selection": False,
        "candidate_physics_mix": list(PHYSICS_MIX_GRID),
        "calibration_modes": list(CALIBRATION_MODES),
        "calibration_status": "locked to none; affine output calibration is absent from the executable path",
        **selection_protocol_fields(),
        "damage_cap_for_ablation_only": DAMAGE_CAP,
        "fold_balance": "campaign-disjoint nested outer/inner folds constructed from all exact and right-censored records",
        "point_metric_balance": "equal campaign contribution; SB_* denotes campaign-balanced metrics",
        "campaign_mapping": "campaign_id read directly from Plot_Data_Verified; no code-side remapping",
        "duplicate_handling": 'one retained row per record_equivalence_id; source-identity qualifications are retained in the workbook; unique IDs do not prove every cross-publication specimen is independent',
        "training_weighting": f"all {len(df)} retained likelihood records use unit weights; exact density and runout survival differ by censoring status",
        "label_review": "is_exact_failure and is_runout read directly from Plot_Data_Verified; no code-side override",
        "location_training": "joint lognormal AFT likelihood: exact density plus right-censored survival",
        "proposed_architecture": 'six-parameter censored-AFT competing-damage trunk with training-estimated water multiplier with selected environmental-decrement penalty, plus an environment-independent exact-fracture ExtraTrees, mechanically augmented ExtraTrees, or Matérn residual',
        "proposed_location_equation": 'mu_phys=-log10(D_mech+D_env); yhat=mu_phys+eta*exp[-(nearest_exact_training_temperature_distance/250)^2]*residual(X_without_environment); physical trunk weight is one; identical non-environment covariates share the same residual correction',
        "residual_extrapolation_guard": 'fixed soft temperature support exp[-(nearest exact-development temperature distance/250 C)^2]; no hard temperature mask in primary selection. Binary flags describe same-temperature availability rather than effective soft residual weight',
        "fitted_physics_parameters": 6, "nuisance_scale_parameters_per_fold": 1,
        "runout_use": "runout stopping cycles enter only as lower bounds through -log survival; they affect physical fitting, scale calibration and survival-NLL selection within the exact-RMSE budget, never point residual regression or point-life metrics",
        "outer_test_tuning": False,
        "primary_validation_scope": (
            "campaign-disjoint outer validation; no campaign crosses train/test"
        ),
        "domain_holdout_scope": (
            "LOSO/LOAO/LOTO/LOEO are factor-held-out stress tests, not an "
            "independent external dataset and not campaign-disjoint by design; "
            "actual overlap is reported per holdout"
        ),
        "hyperparameter_selection": f"{len(HYBRID_CANDIDATES)} hybrid candidates (3 families x 3 gamma x 3 environmental strengths x 2 gains), and {len(CONTROL_CANDIDATES)} selected-control candidates; separate development-only selections on common folds",
        "selection_endpoint": 'each hybrid/control pool: inner campaign-balanced exact RMSE <= that pool minimum+0.02; minimum calibrated inner runout survival NLL, then RMSE, then candidate index. No LOTO model selection. ET-Walker retains its separately documented 0.001 tolerance',
        "censor_guardrail": "diagnostic only: candidate runout survival NLL <= WB-CD runout survival NLL + one paired inner-fold standard error. This comparison does not filter candidates. censor_feasible records its result; the compatibility field censor_constraint_relaxed records a selected diagnostic failure",
        "ET_Walker_fairness": f"same observable inputs and campaign folds; ET-Walker retains {len(NESTED_CANDIDATES)} gamma/tree configurations, WB-PIML has {len(HYBRID_CANDIDATES)} candidates and ML-Strong has {len(CONTROL_CANDIDATES)}. Total search budgets differ",
        "fair_censored_comparators": list(CENSORED_MODELS),
        "primary_comparators": list(PRIMARY_COMPARATORS),
        "primary_inputs": TRAD_NUMERIC + TRAD_CATEGORICAL,
        "arbitrary_oxidation_proxy_in_primary_inputs": False,
        "validation_tiers": ["record_interpolation", "series_disjoint", "source_disjoint", "campaign_disjoint", "LOSO", "LOAO", "LOTO", "LOEO"],
        "external_holdout_selection": (
            'for each LOSO/LOAO/LOTO/LOEO split, WB-PIML, ML-Strong, ExtraTrees and ET-Walker are re-selected on remaining records with campaign-disjoint inner folds; held-out responses are not used for selection or calibration'
        ),
        "external_holdout_count": int(
            external_selection[["scenario", "holdout_group"]]
            .drop_duplicates().shape[0]
        ),
        "external_selection_rows": int(len(external_selection)),
        "external_censor_guard_relaxed_holdouts": int(
            external_relaxed_flags.eq(1).sum()
        ),
        "external_censor_guard_never_relaxed": bool(
            external_censor_status_known
            and external_relaxed_flags.eq(0).all()
        ),
        "external_campaign_overlap_holdouts": int(
            external_campaign_overlap_flags.sum()
        ),
        "external_campaign_disjoint_holdouts": int(
            (~external_campaign_overlap_flags).sum()
        ),
        "external_inner_group": PRIMARY_GROUP_COLUMN,
        "external_outer_test_tuning": False,
        "external_outer_test_interval_calibration": False,
        "secondary_validation_configuration": {'scope': 'repeated record/series/source/campaign validation, duplicate/label sensitivity and S-star sensitivity', 'selection': 'fixed secondary configuration; distinct from nested-selected primary and factor-held-out models', 'gamma': GAMMA, 'strength': 0.0, 'eta': 0.75, 'min_samples_leaf': 3, 'max_features': 0.85, 'family': 'ET_residual', 'temperature_policy': 'soft250', 'temperature_scale_C': 250.0},
        "repeated_validation_runs": args.repeats,
        "primary_model_trees": args.trees,
        "sensitivity_audit_trees": args.audit_trees,
        "sensitivity_audit_changes_primary_model": False,
        "intervals": (
            "campaign-equal weighted development-OOF empirical absolute-residual "
            "intervals; the same predeclared construction is recalibrated from "
            "each held-out training set's OOF residuals in LOSO/LOAO/LOTO/LOEO"
        ),
        "interval_label": "empirical prediction intervals; not formal finite-sample conformal guarantees",
        "external_effect_intervals": "paired campaign-cluster bootstrap CIs within each factor-held-out scenario",
        "output_contract": (
            "one Excel workbook, ten composite PNG figures, and the original 29-slide "
            "PPTX containing each composite and every non-empty subplot; CSV disabled by default"
        ),
        "csv_export_enabled": bool(args.export_csv),
        "bootstrap": args.bootstrap, "seed": SEED,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        'candidate_environment_strength': list(ENVIRONMENT_STRENGTH_GRID),
        'candidate_residual_families': sorted({str(item['family']) for item in HYBRID_CANDIDATES}),
        'strong_control_candidate_count': len(CONTROL_CANDIDATES),
        'strong_control_model': 'ML-Strong',
        'strong_control_pool': '18 ExtraTrees + 18 GBDT + 18 Matérn + one fixed ML-Ens + nine physics-feature ExtraTrees; all selected on development folds; the pool is not purely data-driven',
        'environment_regularizer': 'lambda*mean_training[log10(1+D_env/D_mech)^2], lambda in {0,0.1,1}; all exact/runout development rows contribute covariates to this nonnegative penalty; original joint AFT likelihood and coefficient ridge retained',
        'matern_residual': 'Matérn nu=1.5, length_scale=2, alpha=1; zero residual prior; numeric scaling and categorical encoding fitted only on development exact-fracture rows',
        'physics_residual_features': 'mechanical_prediction and walker_stress only; environmental decrement excluded from every residual family; direct physics-feature controls retain the original mechanical/environmental/Walker features',
        'residual_input_policy': RESIDUAL_INPUT_POLICY,
        'residual_inputs': list(RESIDUAL_NUMERIC + RESIDUAL_CATEGORICAL),
        'residual_environment_features': [],
        'environment_order_guarantee': 'same non-environment covariates give identical residual corrections; the complete model inherits the physical trunk environment ordering; primary beta>=0 gives water life<=air life; beta=0 allows equality; direction-reversing sensitivity variants remain outside the primary assumption', 
        'revision_model_id': 'WB-PIML-environment-shared-residual-20261008',
        'environment_factors': {'vacuum':0.0,'air_or_oxygen':1.0,'water_oxygen':'10**beta, beta>=0; training-only fit'},
        'environment_factor_meaning': 'air reference; water direction-constrained nuisance parameter; no universal physical calibration claim',
        'revision_reason': 'retain training-only water multiplier and use environment-independent shared residuals to preserve trunk ordering throughout nested selection, calibration, validation and frozen inference; post-review model revision; all outputs regenerated',
        'architecture_encoding': 'coarse architecture plus architecture_detail are predictive inputs; LOAO holds out coarse architecture groups',
    }
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    protocol_table = pd.DataFrame([{"item": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}
                                   for key, value in protocol.items()])
    readme_table = pd.DataFrame([
        ("Purpose", "Verified-data, censor-aware, temperature-support, campaign-disjoint WB-PIML analysis"),
        ("Data", f"{data_path.name} / {SHEET} only"),
        ("Label review", "verified worksheet labels retained exactly; zero code-side overrides"),
        ("Point endpoint", f"Point prediction metrics use the {len(exact_df)} exact fractures only"),
        ("Runouts", f"{len(runout_df)} runouts enter training as right-censored survival terms and are never treated as point failure lives"),
        ("Campaign audit", f"{df['campaign_id'].nunique()} human-reviewed campaigns; {len(duplicate_clusters)} physical duplicate clusters ({int(duplicate_clusters['n_sources'].gt(1).sum())} cross-source); {int(duplicate_clusters['reviewed_label_conflict'].sum())} unresolved label-conflict clusters"),
        ("Primary validation", "nested five-fold campaign-disjoint outer validation; no campaign crosses training and test partitions"),
        ("Factor-held-out checks", 'LOSO/LOAO/LOTO/LOEO use training-only nested selection for WB-PIML, ML-Strong, ExtraTrees and ET-Walker; the factor is excluded from development; actual campaign overlap is reported'),
        ("Primary model", 'Censored competing-damage trunk with selected environmental-decrement penalty and ExtraTrees, physics-feature ExtraTrees or Matérn residual; 54 candidates with fixed soft250 support; minimum runout NLL inside the inner RMSE+0.02 budget'),
        ("Secondary validation", 'Fixed ExtraTrees residual: gamma=0.85, environmental strength=0, eta=0.75, leaf=3, max_features=0.85, soft250 support; repeated and duplicate/label/S-star analyses are distinct from nested-selected primary results'),
        ("Censored controls", "Lognormal-AFT and Weibull-AFT use all exact/runout rows, the same observable inputs and the same outer campaign folds"),
        ("Paired control", 'ML-Strong selects from 64 predictive candidates including nine ExtraTrees models with physical features; ET-Walker retains its separate 12-configuration search; all comparisons use common campaign folds and search budgets are not equal'),
        ("Primary inputs", 'Physical trunk: stress, R, temperature, frequency and environment. Residual: stress, R, temperature, frequency, UTS and coarse/detailed architecture; mechanically augmented residual excludes environmental decrement. Direct controls retain environment inputs. No source, batch or record IDs'),
        ("Tabular output", "one Excel workbook including External_Selection; CSV files disabled unless --export-csv is supplied"),
        ("Figure output", f"eight composite PNGs plus {PPTX_OUTPUT_NAME}; every non-empty subplot has a separate slide"),
        ("Interpretation", "Use Claim_Gate before drafting superiority claims"),
    ], columns=["item", "value"])
    metric_definitions = pd.DataFrame([
        ("SB_*", "campaign-balanced metric prefix", "campaign-balanced in every primary and external summary; not publication-source-balanced"),
        ("SB_RMSE", "campaign-balanced RMSE", "square root of the equal-campaign weighted mean squared exact-fracture error"),
        ("SB_MAE", "campaign-balanced MAE", "equal-campaign weighted mean absolute exact-fracture error"),
        ("SB_R2", "campaign-balanced R2", "equal-campaign weighted coefficient of determination on exact fractures"),
        ("SB_F5", "campaign-balanced factor-five accuracy", "equal-campaign weighted fraction with absolute log10 error <= log10(5)"),
        ("SB_C_star", "campaign-balanced C*", "mean campaign-wise five-component composite error; lower is better"),
        ("SB_runout_NLL", "campaign-balanced survival NLL", "right-censored outer-test lower-bound likelihood; separate from point metrics"),
    ], columns=["column", "meaning", "definition"])
    tables = {
        "README": readme_table, "Protocol": protocol_table, "Data_Audit": data_audit,
        "Metric_Definitions": metric_definitions,
        "Model_Specs": model_specs, "Split_Audit": audit,
        "Validation_Definitions": validation_definitions,
        "Nested_Selection": nested_selection, "Nested_Candidates": nested_candidates,
        "Exact_Outer_Pred": predictions.loc[predictions["is_exact"]].copy(),
        "Runout_Outer_Pred": runout_predictions,
        "Runout_Evaluation": runout, "Primary_Campaign_Metrics": metrics,
        "Fold_Metrics": fold_metrics, "Paired_Bootstrap": effects,
        "True_Ablation": ablation, "Source_Gains": source_gains,
        "Component_Ablation": component_ablation,
        "Selection_Sensitivity": selection_sensitivity,
        "Empirical_PI_Rows": proposed_exact, "Empirical_PI_Summary": uq,
        "Temperature_Gate_Rows": temperature_gate_rows,
        "Temperature_Gate_Summary": temperature_gate_summary,
        "Physical_Parameters": parameters, "Sensitivity": sensitivity,
        "Duplicate_Sensitivity": duplicate_sensitivity,
        "Duplicate_Clusters": duplicate_clusters,
        "Duplicate_Candidates": duplicate_candidates,
        "Label_Review": label_review,
        "Campaign_Map": campaign_map,
        "Validation_Tier_Metrics": repeated_metrics,
        "Validation_Tier_Summary": validation_summary,
        "Validation_Fold_Assign": validation_assignments,
        "External_Summary": external_summary, "External_Groups": external_groups,
        "External_Selection": external_selection,
        "External_Paired_CI": external_effects,
        "External_PI_Summary": external_uq,
        "External_PI_Rows": external_pi_rows,
        "External_T_Gate": external_temperature_gate_summary,
        "Sstar_Sensitivity": sstar_sensitivity, "QA_Flagged_Rows": qa_flagged,
        "Claim_Gate": gate, "Outer_Fold_Assignment": outer_fold_assignment,
        "References": references,
    }
    probability_calibration_all = pd.concat([probability_calibration, external_probability_calibration],
                                            ignore_index=True, sort=False)
    probability_outputs = assemble_probability_tables(
        probability_predictions, external_probability_predictions, probability_calibration_all, args.bootstrap
    )
    tables.update(probability_outputs)
    tables.update(simple_control_tables(
        predictions, external_holdout_predictions, probability_outputs, args.bootstrap
    ))
    tables["Trunk_External_Candidates"] = simple_external_candidates
    if args.export_csv:
        for name, table in tables.items():
            if name.startswith("Trunk_"):
                table.to_csv(out / (name.lower() + ".csv"), index=False)
    workbook_path = out / "WB_PIML_results.xlsx"
    write_workbook(workbook_path, tables)
    deck = FigureDeckCollector(data_path.name)
    legacy = lambda table: table.loc[~table["model"].isin(SIMPLE_CONTROL_MODELS)]
    make_figures(out, legacy(predictions), legacy(metrics), effects, uq,
                 legacy(runout), source_gains, deck)
    make_validation_figure(out, validation_summary, external_summary, deck)
    pptx_path = out / PPTX_OUTPUT_NAME
    deck.write(pptx_path)
    write_readme(
        out / "README_results.md", data_path, metrics, effects, gate, df,
        parameters, len(runout_df),
    )
    probability_figure = make_probability_figure(out, probability_outputs)
    make_trunk_comparison_figure(out, tables)
    write_probability_notes(out, probability_outputs)
    print(f"Probability comparison added to {workbook_path}; figure: {probability_figure}")
    console_models = ["WB-PIML", "ExtraTrees", "ET-Walker", "ML-Ens"]
    console_rows = []
    for scope, label in [
        ("record_interpolation", "Fixed-config repeated record split"),
        ("series_disjoint", "Fixed-config repeated series-disjoint"),
        ("source_disjoint", "Fixed-config repeated source-disjoint"),
        ("campaign_disjoint", "Fixed-config repeated campaign-disjoint"),
    ]:
        row = {"validation_scope": label}
        for model in console_models:
            value = validation_summary.loc[
                validation_summary["strategy"].eq(scope) & validation_summary["model"].eq(model),
                "SB_RMSE_mean",
            ].iloc[0]
            row[f"{model}_RMSE"] = float(value)
        console_rows.append(row)
    primary_row = {"validation_scope": "Primary nested campaign-disjoint"}
    for model in console_models:
        primary_row[f"{model}_RMSE"] = float(metrics.loc[metrics["model"].eq(model), "SB_RMSE"].iloc[0])
    console_rows.append(primary_row)
    for scope in ["LOSO", "LOAO", "LOTO", "LOEO"]:
        row = {"validation_scope": scope}
        for model in console_models:
            row[f"{model}_RMSE"] = float(external_summary.loc[
                external_summary["scenario"].eq(scope) & external_summary["model"].eq(model), "SB_RMSE"
            ].iloc[0])
        console_rows.append(row)
    console_summary = pd.DataFrame(console_rows)
    for name in SIMPLE_CONTROL_MODELS:
        console_summary[name + "_RMSE"] = [
            np.nan if i < 4 else
            float(metrics.loc[metrics.model.eq(name), "SB_RMSE"].iloc[0]) if i == 4 else
            float(external_summary.loc[external_summary.scenario.eq(
                ("LOSO", "LOAO", "LOTO", "LOEO")[i - 5]) & external_summary.model.eq(name), "SB_RMSE"].iloc[0])
            for i in range(len(console_summary))
        ]
    print("\nValidation-scope RMSE summary (one table; lower is better)\n",
          console_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved one tabular results workbook: {workbook_path}")
    print(f"Saved complete figure presentation: {pptx_path}")
    print(f"PNG figures and audit metadata: {out}")
    print(f"CSV export: {'enabled' if args.export_csv else 'disabled'}")
    print(f"Elapsed: {time.perf_counter() - started:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument(
        "--audit-trees", type=int, default=160,
        help=(
            "Trees per residual model in the expanded sensitivity audit. "
            "This does not change the primary model (default: 160)."
        ),
    )
    parser.add_argument("--repeats", type=int, default=10,
                        help="Repeated validation runs for each of the four validation tiers (1-10).")
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Also export per-table CSV files (disabled by default).",
    )
    parser.add_argument("--jobs", type=int, default=min(8, max(1, (os.cpu_count() or 2) // 2)),
                        help="Independent validation processes (default: up to 8).")
    parser.add_argument("--tree-jobs", type=int, default=1,
                        help="Threads per forest; 1 avoids nested thread overhead.")
    parser.add_argument("--blas-threads", type=int, default=1,
                        help="Numerical-library threads per process (default: 1).")
    args = parser.parse_args()
    global WORKERS, TREE_JOBS, BLAS_THREADS
    if args.jobs < 1 or args.tree_jobs == 0 or args.tree_jobs < -1 or args.blas_threads < 1:
        parser.error("--jobs/--blas-threads must be positive; --tree-jobs must be -1 or positive")
    WORKERS, TREE_JOBS, BLAS_THREADS = args.jobs, args.tree_jobs, args.blas_threads
    if Font is None:
        raise ImportError("Excel output requires openpyxl")
    if any(value is None for value in (Image, Presentation, RGBColor, PP_ALIGN, Inches, Pt)):
        raise ImportError("PowerPoint output requires Pillow and python-pptx")
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 campaign bootstrap replicates")
    if args.trees < 20:
        raise ValueError("--trees must be at least 20")
    if args.audit_trees < 20:
        raise ValueError("--audit-trees must be at least 20")
    if not 1 <= args.repeats <= len(REPEATED_VALIDATION_SEEDS):
        raise ValueError(f"--repeats must be between 1 and {len(REPEATED_VALIDATION_SEEDS)}")
    with threadpool_limits(limits=BLAS_THREADS):
        _run_analysis(args)


import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtri

