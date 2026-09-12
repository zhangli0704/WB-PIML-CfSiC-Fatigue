


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
    _, _, log_k, stress_exponent, temperature_exponent = np.asarray(coefficients, float)
    exposure = frame["environment_exposure"].to_numpy(float)
    active = exposure > 0.0
    decrement = np.zeros(len(frame), dtype=float)
    log_sw = walker_log_stress(frame, gamma)
    log_ratio = (
        log_k + np.log10(exposure[active])
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
    """Predict both predeclared pools; gains share the same fitted residual."""
    require_exact_training(train, "predict_structural_configs")
    exact = train.loc[train.is_exact].reset_index(drop=True)
    y = exact.logN.to_numpy(float)
    prep = make_preprocessor()
    x = prep.fit_transform(exact)
    xv = prep.transform(valid)
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
                physics[key] = (base[:len(exact)], base[len(exact):], xa, xav)
            base_train, base_valid, xa, xav = physics[key]
        family = c["family"]
        fit_key = tuple(sorted((k, v) for k, v in c.items() if k not in ("candidate_id", "eta", "pool")))
        if fit_key not in fitted:
            if c["pool"] == "hybrid":
                target = y - base_train
                if family == "Matern_residual":
                    correction = kernel_predict(x, xv, target,
                        float(c.get("kernel_length_scale", 2.0)), float(c.get("ridge_alpha", 1.0)), False)
                else:
                    xx, xxv = (xa, xav) if family == "physics_feature_ET_residual" else (x, xv)
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
        {"model": "WB-CD", "role": 'right-censored competing-damage physical reference', "location_parameters": 5,
         "equation": 'mu=-log10(D_mech+D_env); exact density+runout survival; five location parameters',
         "used_for_selection": True},
        {"model": "WB-M", "role": "mechanical-proxy ablation", "location_parameters": 3,
         "equation": "a - b*log10(S_Walker) - c*M", "used_for_selection": False},
        {"model": "WB-Ox", "role": "oxidation-proxy ablation", "location_parameters": 3,
         "equation": "a - b*log10(S_Walker) - c*Dox", "used_for_selection": False},
        {"model": "WB-PIML-Anchor", "role": 'development-selected competing-damage residual branch', "location_parameters": 5,
         "equation": 'mu_phys+eta*soft250*residual(X); ExtraTrees, physics-feature ExtraTrees or Matérn; selected environmental-decrement penalty', "used_for_selection": True},
        {"model": "WB-PIML", "role": 'reported always-physical censored-AFT hybrid', "location_parameters": 5,
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
        "proposed_architecture": 'five-parameter censored-AFT competing-damage trunk with selected environmental-decrement penalty, plus an exact-fracture ExtraTrees, physics-feature ExtraTrees, or Matérn residual',
        "proposed_location_equation": 'mu_phys=-log10(D_mech+D_env); yhat=mu_phys+eta*exp[-(nearest_exact_training_temperature_distance/250)^2]*residual(X); physical trunk weight is one',
        "residual_extrapolation_guard": 'fixed soft temperature support exp[-(nearest exact-development temperature distance/250 C)^2]; no hard temperature mask in primary selection. Binary flags describe same-temperature availability rather than effective soft residual weight',
        "fitted_physics_parameters": 5, "nuisance_scale_parameters_per_fold": 1,
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
        'physics_residual_features': 'mechanical_prediction, environment_decrement, walker_stress; physical coefficients fitted only on the corresponding development subset; the same derived inputs are available to nine direct ExtraTrees controls',
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
        ("Primary inputs", 'Observable stress, R, temperature, frequency, UTS, coarse and detailed architecture, and environment; the physics-feature family adds physical quantities fitted on development records; no source, batch or record IDs'),
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



"""Censored predictive scores and intervals on Y = log10(N).

All evaluation weights give campaigns equal mass and, within each campaign,
give record-equivalence groups equal mass. These weights do not fit models or
calibrate their scales. Interval coverage uses observed exact failures only.
"""

import numpy as np
import pandas as pd
from scipy.special import log_ndtr, ndtri


def _prob_family_array(family):
    values = np.asarray(family, dtype=str)
    values = np.char.lower(np.char.strip(values))
    valid = np.isin(values, ["normal", "lognormal", "weibull"])
    if not np.all(valid):
        raise ValueError(f"Unknown probability families: {np.unique(values[~valid])}")
    return values


def _prob_bool_array(values, name):
    values = np.asarray(values)
    if values.dtype.kind == "b":
        return values
    if values.dtype.kind in "iuf" and np.all(np.isin(values, [0, 1])):
        return values.astype(bool)
    strings = np.char.lower(np.char.strip(values.astype(str)))
    if not np.all(np.isin(strings, ["true", "false", "0", "1"])):
        raise ValueError(f"{name} must contain boolean or 0/1 values")
    return np.isin(strings, ["true", "1"])


def _prob_distribution_inputs(family, location, scale):
    family, location, scale = np.broadcast_arrays(
        _prob_family_array(family), np.asarray(location, float), np.asarray(scale, float)
    )
    if not np.all(np.isfinite(location)):
        raise ValueError("Probability locations must be finite")
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("Probability scales must be finite and positive")
    return family, location, scale


def distribution_nll(family, y, location, scale, is_exact):
    """Return exact density or right-censored survival NLL, row by row.

    ``normal`` and ``lognormal`` mean Normal(location, scale) on Y. ``weibull``
    means the existing Weibull-AFT parameterization: Y has a minimum-Gumbel
    distribution, F(Y)=1-exp(-exp((Y-location)/scale)). Its location is not its
    median. Densities are with respect to Y, not N; no additional Jacobian is
    applied here. Unrepresentably large tail losses remain positive infinity.
    """
    family, location, scale = _prob_distribution_inputs(family, location, scale)
    family, y, location, scale, exact = np.broadcast_arrays(
        family, np.asarray(y, float), location, scale,
        _prob_bool_array(is_exact, "is_exact"),
    )
    if not np.all(np.isfinite(y)):
        raise ValueError("Observed log-lifetimes and censoring bounds must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        z = (y - location) / scale
        normal_exact = 0.5 * np.log(2.0 * np.pi) + np.log(scale) + 0.5 * z**2
        normal_runout = -log_ndtr(-z)
        exp_z = np.exp(z)
        weibull_exact = np.log(scale) - z + exp_z
        weibull_exact = np.where(np.isposinf(z), np.inf, weibull_exact)
        normal_loss = np.where(exact, normal_exact, normal_runout)
        weibull_loss = np.where(exact, weibull_exact, exp_z)
        result = np.where(family == "weibull", weibull_loss, normal_loss)
    if np.any(np.isnan(result)):
        raise FloatingPointError("Undefined predictive NLL")
    return result


def distribution_quantile(family, location, scale, p):
    """Return predictive quantiles on log10(N); p may be a scalar or array."""
    family, location, scale = _prob_distribution_inputs(family, location, scale)
    family, location, scale, p = np.broadcast_arrays(
        family, location, scale, np.asarray(p, float)
    )
    if not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise ValueError("Quantile probabilities must lie in [0, 1]")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        standard = np.where(family == "weibull", np.log(-np.log1p(-p)), ndtri(p))
        return location + scale * standard


def _prob_weights(frame):
    """Normalized campaign/equivalence weights, recomputed for this frame."""
    if frame.empty:
        return np.empty(0, float)
    identity = frame[["campaign_id", "record_equivalence_id"]].astype(str)
    multiplicity = identity.groupby(
        ["campaign_id", "record_equivalence_id"], sort=False
    )["record_equivalence_id"].transform("size").to_numpy(float)
    groups_per_campaign = identity.drop_duplicates().groupby(
        "campaign_id", sort=False
    )["record_equivalence_id"].nunique()
    counts = identity["campaign_id"].map(groups_per_campaign).to_numpy(float)
    weights = 1.0 / (multiplicity * counts)
    return weights / weights.sum()


def _prob_mean(values, weights):
    if len(weights) == 0:
        return float("nan")
    return float(np.dot(np.asarray(values, float), weights) / np.sum(weights))


def _prob_level_tag(level):
    return f"{100.0 * level:g}"


def _prob_validate_predictions(predictions):
    required = {
        "model", "row_id", "campaign_id", "record_equivalence_id", "logN",
        "is_exact", "is_runout", "pred_logN", "prob_location", "prob_scale",
        "prob_family", "scenario", "fold", "holdout_group",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing probability prediction columns: {missing}")
    frame = predictions.copy().reset_index(drop=True)
    for column in ["model", "row_id", "campaign_id", "record_equivalence_id", "scenario"]:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must have a nonempty identity for every row")
    frame["holdout_group"] = frame["holdout_group"].fillna("").astype(str)
    frame["is_exact"] = _prob_bool_array(frame["is_exact"], "is_exact")
    frame["is_runout"] = _prob_bool_array(frame["is_runout"], "is_runout")
    if np.any(frame["is_exact"].to_numpy() == frame["is_runout"].to_numpy()):
        raise ValueError("Every row must be exactly one of exact failure or right-censored runout")
    for column in ["logN", "pred_logN", "prob_location", "prob_scale"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} must be finite for every row")
    if (frame["prob_scale"] <= 0).any():
        raise ValueError("prob_scale must be positive for every row")
    frame["prob_family"] = _prob_family_array(frame["prob_family"])
    identity = ["scenario", "model", "holdout_group", "row_id"]
    if frame.duplicated(identity).any():
        raise ValueError("Multiple test predictions for one scenario/model/holdout/row_id")
    return frame


def _prob_summary(frame):
    weights = _prob_weights(frame)
    exact = frame["is_exact"].to_numpy(bool)
    runout = frame["is_runout"].to_numpy(bool)
    loss = frame["row_nll"].to_numpy(float)
    exact_mass = float(weights[exact].sum())
    runout_mass = float(weights[runout].sum())
    exact_component = float(np.dot(weights[exact], loss[exact]))
    runout_component = float(np.dot(weights[runout], loss[runout]))
    exact_frame = frame.loc[exact]
    runout_frame = frame.loc[runout]
    exact_weights = _prob_weights(exact_frame)
    runout_weights = _prob_weights(runout_frame)
    error = exact_frame["pred_logN"].to_numpy(float) - exact_frame["logN"].to_numpy(float)
    with np.errstate(over="ignore", invalid="ignore"):
        mse = _prob_mean(error**2, exact_weights)
    exact_y = exact_frame["logN"].to_numpy(float)
    y_mean = _prob_mean(exact_y, exact_weights)
    y_variance = _prob_mean((exact_y - y_mean)**2, exact_weights)
    return {
        "n_rows": len(frame),
        "n_exact": int(exact.sum()),
        "n_runout": int(runout.sum()),
        "n_campaigns": int(frame["campaign_id"].nunique()),
        "n_exact_campaigns": int(exact_frame["campaign_id"].nunique()),
        "n_runout_campaigns": int(runout_frame["campaign_id"].nunique()),
        "n_equivalence_groups": int(frame[["campaign_id", "record_equivalence_id"]].drop_duplicates().shape[0]),
        "n_nonfinite_NLL": int((~np.isfinite(loss)).sum()),
        "SB_joint_censored_NLL": _prob_mean(loss, weights),
        "SB_exact_component_NLL": exact_component,
        "SB_runout_component_NLL": runout_component,
        "exact_weight_mass": exact_mass,
        "runout_weight_mass": runout_mass,
        "conditional_exact_NLL": exact_component / exact_mass if exact_mass > 0 else np.nan,
        "conditional_runout_NLL": runout_component / runout_mass if runout_mass > 0 else np.nan,
        "SB_exact_NLL": _prob_mean(exact_frame["row_nll"], exact_weights),
        "SB_runout_NLL": _prob_mean(runout_frame["row_nll"], runout_weights),
        "SB_MSE": mse,
        "SB_RMSE": float(np.sqrt(mse)),
        "SB_MAE": _prob_mean(np.abs(error), exact_weights),
        "SB_F5": _prob_mean(np.abs(error) <= np.log10(5.0), exact_weights),
        "SB_R2": 1.0 - mse / y_variance if y_variance > 0 else np.nan,
        "density_coordinate": "Y=log10(N)",
        "evaluation_weighting": "equal campaigns; equal equivalence groups within campaign",
        "component_weighting": "same full-test weights; components sum to joint NLL",
        "conditional_weighting": "full-test components divided by exact/runout weight mass",
        "subset_score_weighting": "SB_exact_NLL/SB_runout_NLL recompute weights within each subset",
        "point_prediction": "stored pred_logN; Weibull location is retained, not replaced by its median",
    }


def _prob_interval_summary(frame, level):
    tag = _prob_level_tag(level)
    exact_frame = frame.loc[frame["is_exact"]]
    weights = _prob_weights(frame)
    exact_weights = _prob_weights(exact_frame)
    return {
        "nominal_level": float(level),
        "n_rows": len(frame),
        "n_exact": len(exact_frame),
        "n_runout": int(frame["is_runout"].sum()),
        "n_campaigns": int(frame["campaign_id"].nunique()),
        "n_exact_campaigns": int(exact_frame["campaign_id"].nunique()),
        "SB_exact_coverage": _prob_mean(exact_frame[f"prob_covered{tag}"], exact_weights),
        "SB_all_width_log10N": _prob_mean(frame[f"prob_width{tag}"], weights),
        "SB_exact_width_log10N": _prob_mean(exact_frame[f"prob_width{tag}"], exact_weights),
        "interval_definition": "central parametric predictive interval on log10(N)",
        "coverage_population": "observed exact failures only; censoring may bias coverage",
        "coverage_adjustment": "none; no IPCW",
        "all_width_population": "all test conditions, including right-censored records",
    }


def probability_tables(predictions, levels=(0.8, 0.9), include_factor_groups=True):
    """Return metrics, intervals, enriched rows, and campaign summaries.

    Overall scores pool test folds within each scenario/model. Nonempty factor
    levels are additionally reported when ``include_factor_groups`` is true.
    Exact/runout NLL components share the full-population weights. Their
    conditional scores divide those components by their respective weight
    masses. Separately named SB_exact_NLL and SB_runout_NLL instead give equal
    mass to campaigns represented in their corresponding subsets.

    Point metrics always use the supplied pred_logN on exact failures. Coverage
    uses exact failures only and is not a censoring-adjusted population measure.
    No model fitting, hyperparameter choice, or calibration occurs here.
    """
    levels = tuple(float(level) for level in levels)
    if not levels or any(not (0 < level < 1) for level in levels) or len(set(levels)) != len(levels):
        raise ValueError("Interval levels must be distinct values strictly between zero and one")
    frame = _prob_validate_predictions(predictions)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), frame, pd.DataFrame()
    family = frame["prob_family"].to_numpy(str)
    location = frame["prob_location"].to_numpy(float)
    scale = frame["prob_scale"].to_numpy(float)
    exact = frame["is_exact"].to_numpy(bool)
    y = frame["logN"].to_numpy(float)
    frame["row_nll"] = distribution_nll(family, y, location, scale, exact)
    frame["row_exact_NLL"] = np.where(exact, frame["row_nll"], np.nan)
    frame["row_runout_NLL"] = np.where(~exact, frame["row_nll"], np.nan)
    frame["prob_median_logN"] = distribution_quantile(family, location, scale, 0.5)
    for level in levels:
        tag = _prob_level_tag(level)
        tail = (1.0 - level) / 2.0
        low = distribution_quantile(family, location, scale, tail)
        high = distribution_quantile(family, location, scale, 1.0 - tail)
        frame[f"prob_lower{tag}"] = low
        frame[f"prob_upper{tag}"] = high
        frame[f"prob_width{tag}"] = high - low
        frame[f"prob_covered{tag}"] = np.where(exact, ((y >= low) & (y <= high)).astype(float), np.nan)
    frame["row_weight_overall"] = np.nan
    rows, intervals, campaigns = [], [], []

    def append_scope(group, scenario, model, aggregation_level, holdout_group):
        keys = {
            "scenario": scenario, "aggregation_level": aggregation_level,
            "holdout_group": holdout_group, "model": model,
        }
        rows.append({**keys, **_prob_summary(group)})
        for level in levels:
            intervals.append({**keys, **_prob_interval_summary(group, level)})
        for campaign, campaign_frame in group.groupby("campaign_id", sort=False):
            summary = _prob_summary(campaign_frame)
            summary["joint_censored_NLL"] = summary.pop("SB_joint_censored_NLL")
            summary["exact_component_NLL"] = summary.pop("SB_exact_component_NLL")
            summary["runout_component_NLL"] = summary.pop("SB_runout_component_NLL")
            summary["point_MSE"] = summary.pop("SB_MSE")
            for level in levels:
                interval = _prob_interval_summary(campaign_frame, level)
                tag = _prob_level_tag(level)
                summary[f"exact_coverage{tag}"] = interval["SB_exact_coverage"]
                summary[f"all_width{tag}_log10N"] = interval["SB_all_width_log10N"]
                summary[f"exact_width{tag}_log10N"] = interval["SB_exact_width_log10N"]
            campaigns.append({**keys, "campaign_id": campaign, **summary})

    for (scenario, model), group in frame.groupby(["scenario", "model"], sort=False):
        frame.loc[group.index, "row_weight_overall"] = _prob_weights(group)
        append_scope(group, scenario, model, "overall", "")
        if include_factor_groups and str(scenario).lower() != "primary":
            factors = group.loc[group["holdout_group"].str.strip().ne("")]
            for holdout_group, factor in factors.groupby("holdout_group", sort=False):
                append_scope(factor, scenario, model, "factor_level", holdout_group)
    return pd.DataFrame(rows), pd.DataFrame(intervals), frame, pd.DataFrame(campaigns)


"""Paired distribution comparisons and a compact figure for the results workbook."""

PROBABILITY_SHEETS = (
    "Prob_Metrics", "Prob_Intervals", "Prob_Predictions", "Prob_Campaigns",
    "Prob_Paired_CI", "Prob_Calibration", "Prob_Method",
)


def probability_paired_effects(rows, campaigns, n_bootstrap=5000, seed=20260824):
    results = []
    scenarios = [s for s in ("Primary", "LOSO", "LOAO", "LOTO", "LOEO")
                 if s in set(rows.scenario)]
    for scope_index, scenario in enumerate(scenarios):
        scope = rows.loc[rows.scenario.eq(scenario)]
        proposed = scope.loc[scope.model.eq("WB-PIML")]
        keys = ["row_id", "campaign_id", "record_equivalence_id", "logN",
                "is_exact", "is_runout", "fold", "holdout_group"]
        identity = proposed[keys].sort_values("row_id").reset_index(drop=True)
        if identity.row_id.duplicated().any():
            raise AssertionError("Probability comparison requires one test prediction per record")
        scope_campaigns = campaigns.loc[
            campaigns.scenario.eq(scenario) & campaigns.aggregation_level.eq("overall")]
        base = scope_campaigns.loc[scope_campaigns.model.eq("WB-PIML")].set_index("campaign_id")
        for index, model in enumerate(legacy_first_models(set(scope.model) - {"WB-PIML"})):
            comparator = scope.loc[scope.model.eq(model)]
            other_identity = comparator[keys].sort_values("row_id").reset_index(drop=True)
            pd.testing.assert_frame_equal(identity, other_identity, check_dtype=False,
                                          check_exact=False, rtol=0, atol=1e-12)
            other = scope_campaigns.loc[scope_campaigns.model.eq(model)].set_index("campaign_id")
            paired = base[["joint_censored_NLL"]].join(
                other[["joint_censored_NLL"]], how="outer", lsuffix="_proposed", rsuffix="_comparator")
            if paired.isna().any().any():
                raise AssertionError("Both probability models must cover the same campaigns")
            delta = (paired.joint_censored_NLL_comparator - paired.joint_censored_NLL_proposed).to_numpy(float)
            low = high = np.nan
            bootstrap_seed = int(seed + 980000 + 1000 * scope_index + index)
            if len(delta) > 1 and np.isfinite(delta).all():
                rng = np.random.default_rng(bootstrap_seed)
                draws = rng.integers(0, len(delta), size=(n_bootstrap, len(delta)))
                low, high = np.quantile(delta[draws].mean(axis=1), [.025, .975])
            results.append(dict(
                scenario=scenario, comparator=model, proposed="WB-PIML",
                effect="joint_censored_NLL_gain", estimate=float(np.mean(delta)),
                CI95_lo=float(low), CI95_hi=float(high), n_campaigns=len(delta),
                n_test_records=len(identity), bootstrap_replicates=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
                gain_definition="comparator minus WB-PIML; positive favors WB-PIML",
                uncertainty_scope="conditional campaign bootstrap; does not account for repeated model development",
            ))
    return pd.DataFrame(results)


def probability_method_table():
    return pd.DataFrame([
        ("Target", "Conditional predictive distribution of Y=log10(Nf) at each held-out test condition."),
        ("Point predictions", "All existing point predictions and selection rules are retained. Probability calibration fits scale only, without a location shift."),
        ("Primary models", "WB, WB-CD, WB-M, WB-Ox, WB-PIML, RF, ExtraTrees, ET-Walker, GBDT, SVR, ML-Ens, ML-Strong, Lognormal-AFT, Weibull-AFT, WB-Residual and Basquin-Residual."),
        ("Factor models", "WB, WB-PIML, ExtraTrees, ET-Walker, ML-Ens, ML-Strong, WB-Residual and Basquin-Residual in LOSO/LOAO/LOTO/LOEO. The two native AFT controls are evaluated in primary validation only."),
        ("Normal distributions", "Y follows Normal(location, scale). Existing primary WB/physical/ML-Strong scales are retained. Ordinary regressors and factor-holdout WB receive development OOF scale estimates; factor WB-PIML/ML-Strong use their selected development OOF predictions."),
        ("Ordinary regression calibration", "The selected ExtraTrees/ET-Walker configurations are fixed in the development campaign folds. RF, GBDT and SVR retain their existing settings. ML-Ens averages the four corresponding fold predictions. Each model fits its own single scale."),
        ("Calibration weighting", "Unit record weights in exact-density plus right-censored-survival likelihood. Hyperparameter selection has already used the development data; these calibration predictions are not an independent calibration sample."),
        ("Scale bounds", "The existing log10-life scale range [0.08, 4.0] is retained. It is a numerical regularization range, not a material constant."),
        ("Weibull distribution", "F_Y(y)=1-exp(-exp((y-location)/scale)). Equivalently N has Weibull scale 10**location and shape 1/(scale*ln(10)). Existing point location is retained and is not the distribution median."),
        ("Common density coordinate", "Exact densities are evaluated with respect to log10(Nf) for every model; survival probabilities are dimensionless. No additional change-of-variable term is added to a density already defined on this coordinate."),
        ("Full censored NLL", "For an exact fracture use -log f_Y(y|x); for a runout at c use -log S_Y(c|x). Both terms are evaluated out of the outer training set."),
        ("Censoring assumption", "The likelihood treats termination as noninformative for the latent lifetime conditional on the recorded test conditions."),
        ("Native AFT calibration records", "Native AFT scale is fitted jointly with location on development records after its existing inner selection of regularization. Zero calibration folds in native-fit metadata means no separate OOF scale-calibration stage, not absence of hyperparameter selection."),
        ("Joint weighting", "Campaigns are equally weighted; equivalence identities and records within identity share each campaign's mass. Exact and runout components use the same full-sample weights and sum to the joint score."),
        ("Conditional score fields", "conditional_exact_NLL and conditional_runout_NLL condition the full-sample mass on status. SB_exact_NLL and SB_runout_NLL instead rebalance the corresponding status subset by campaign and must not be combined by raw record fractions."),
        ("Predictive intervals", "Central 80% and 90% quantiles of each model's predictive distribution. They are separate from the retained empirical absolute-residual intervals."),
        ("Observed coverage", "Coverage is observable only among exact failures here. Exact-failure-only coverage is affected by censoring selection and does not establish unconditional coverage of all latent lifetimes. Runout stopping times are never scored as exact failures."),
        ("Interval width", "Width is measured in log10-life units. Both all-test-condition and exact-failure-only campaign-balanced means are reported."),
        ("Paired uncertainty", "5,000 campaign bootstrap draws by default on the same test records; positive NLL gain means the comparator has higher loss. Single-campaign confidence intervals are not reported."),
        ("Factor validation scope", "Each factor is excluded from its corresponding training set. Campaign overlap in factor stress tests remains as reported in External_Selection; these are not automatically independent experiments."),
        ("Interpretation of scatter", "Calibrated predictive dispersion combines specimen variability, unobserved conditions, data uncertainty and model error. It does not identify pure intrinsic material scatter or separate aleatoric and epistemic components."),
        ("Conditional scale limitation", "This extension uses a constant scale within each trained model. A covariate-dependent scale is not assumed or established by these comparisons."),
        ("Proper comparison", "A larger interval is not itself an improvement. Read coverage together with width and the full censored likelihood, while retaining the point metrics."),
        ("Additional simple trunks", "WB-Residual: classic WB plus residual, 18 candidates. Basquin-Residual: gamma=0 plus residual, 6 candidates. Both independently selected on the original inner campaign folds and included in primary and all factor holdouts, with their own OOF scales. Environment penalty is absent."),
        ("Revision-stage comparison", "The simple-trunk comparisons were motivated by inspection of existing results. Reusing these campaigns does not establish a new independent confirmation; uncertainty remains conditional on the studied data."),
    ], columns=["item", "definition"])


def assemble_probability_tables(primary, external, calibration, bootstrap):
    combined = pd.concat([primary, external], ignore_index=True, sort=False)
    metrics, intervals, rows, campaigns = probability_tables(combined)
    effects = probability_paired_effects(rows, campaigns, bootstrap, SEED)
    return {
        "Prob_Metrics": metrics, "Prob_Intervals": intervals,
        "Prob_Predictions": rows, "Prob_Campaigns": campaigns,
        "Prob_Paired_CI": effects, "Prob_Calibration": calibration,
        "Prob_Method": probability_method_table(),
    }


def make_probability_figure(out, tables):
    models = ["WB", "WB-CD", "ExtraTrees", "ET-Walker", "ML-Ens",
              "ML-Strong", "Lognormal-AFT", "Weibull-AFT", "WB-PIML"]
    metrics = tables["Prob_Metrics"]
    metrics = metrics.loc[metrics.scenario.eq("Primary") & metrics.aggregation_level.eq("overall")].set_index("model")
    intervals = tables["Prob_Intervals"]
    intervals = intervals.loc[intervals.scenario.eq("Primary") & intervals.aggregation_level.eq("overall")]
    effects = tables["Prob_Paired_CI"]
    effects = effects.loc[effects.scenario.eq("Primary")].set_index("comparator")
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5))
    color = ["#b44a43" if m == "WB-PIML" else "#477c97" for m in models]
    positions = np.arange(len(models))
    loss = metrics.loc[models, "SB_joint_censored_NLL"].to_numpy(float)
    axes[0, 0].barh(positions, loss, color=color, height=.65)
    axes[0, 0].set(yticks=positions, yticklabels=models,
                   xlabel="Campaign-balanced full censored NLL (lower is better)",
                   title="(a) Full predictive-distribution score")
    axes[0, 0].invert_yaxis()
    comparisons = models[:-1]
    estimates = effects.loc[comparisons, "estimate"].to_numpy(float)
    lows = effects.loc[comparisons, "CI95_lo"].to_numpy(float)
    highs = effects.loc[comparisons, "CI95_hi"].to_numpy(float)
    for index, (estimate, low, high) in enumerate(zip(estimates, lows, highs)):
        axes[0, 1].plot([low, high], [index, index], color="#477c97", lw=1.8)
        axes[0, 1].plot(estimate, index, "o", color="#b44a43", ms=5)
    axes[0, 1].axvline(0, color="#777777", linestyle="--", lw=1)
    axes[0, 1].set(yticks=np.arange(len(comparisons)), yticklabels=comparisons,
                   xlabel="Comparator NLL minus WB-PIML NLL",
                   title="(b) Paired gain and conditional 95% CI")
    axes[0, 1].invert_yaxis()
    for level, offset, shade in ((.8, -.16, "#cf8b45"), (.9, .16, "#60976c")):
        level_rows = intervals.loc[np.isclose(intervals.nominal_level, level)].set_index("model").loc[models]
        axes[1, 0].plot(positions + offset, level_rows.SB_exact_coverage,
                        "o", color=shade, label=f"{int(level*100)}% interval", ms=5)
        axes[1, 0].axhline(level, color=shade, linestyle="--", lw=.9)
        axes[1, 1].bar(positions + offset, level_rows.SB_all_width_log10N,
                       width=.3, color=shade, label=f"{int(level*100)}% interval")
    axes[1, 0].set(ylim=(0, 1.03), ylabel="Exact-failure-only coverage",
                   title="(c) Coverage among observed exact failures")
    axes[1, 1].set(ylabel="Mean interval width in log10(Nf)",
                   title="(d) Width over all test conditions")
    for ax in axes[1]:
        ax.set_xticks(positions, models, rotation=35, ha="right")
        ax.legend(fontsize=9, frameon=False)
    for ax in axes.flat:
        ax.tick_params(labelsize=9)
        ax.grid(axis="x" if ax in axes[0] else "y", alpha=.2)
    fig.suptitle("Point predictions retained; probability models compared on the same held-out records", fontsize=14)
    fig.text(.5, .018,
             "Coverage excludes runouts and is not unconditional lifetime coverage. The likelihood uses both exact failures and runouts.\n"
             "Intervals are predictive ranges; paired-score confidence intervals are conditional on these previously studied campaigns.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .96))
    path = out / "09_probability_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_probability_notes(out, tables):
    metadata = {
        "density_coordinate": "log10(Nf)",
        "primary_probability_model_count": 14 + len(SIMPLE_CONTROL_MODELS),
        "factor_probability_model_count": 6 + len(SIMPLE_CONTROL_MODELS),
        "normal_scale": "one training-calibrated constant per model and outer training set; location unchanged",
        "native_AFT_distribution": "original fitted lognormal or Weibull distribution retained",
        "main_probability_metric": "campaign-balanced full exact-density/right-censored-survival NLL",
        "interval_levels": [.8, .9],
        "coverage_scope": "exact failures only; not unconditional lifetime coverage",
        "scatter_interpretation": "combined predictive uncertainty, not identified intrinsic scatter",
        "sheets": list(PROBABILITY_SHEETS),
        "figure": "09_probability_comparison.png",
        "point_predictions_changed_by_probability_calibration": False,
        "hyperparameters_reselected_using_probability_test_scores": False,
    }
    protocol_path = out / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["probability_comparison"] = metadata
    protocol["simple_trunk_controls"] = {
        "models": list(SIMPLE_CONTROL_MODELS), "candidate_counts": [18, 6],
        "gamma_grids": [list(STRUCTURAL_GAMMAS), [0.0]],
        "residual_families": list(HYBRID_FAMILIES), "eta_grid": list(RESIDUAL_GAIN_GRID),
        "environment_penalty": "not applicable; not searched",
        "selection": "independent per model; same inner splits and RMSE budget/runout NLL rule",
        "primary_and_factor_holdouts": True, "fixed_repeated_validation_changed": False,
        "origin": "revision-stage comparisons motivated by previously inspected results",
        "uncertainty": "conditional campaign bootstrap; no new independent test data",
    }
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    readme_path = out / "README_results.md"
    text = readme_path.read_text(encoding="utf-8")
    text += """\n\n## Conditional lifetime distributions\n\nPoint predictions and their original comparisons are retained. `Prob_Metrics` adds complete censored negative log-likelihood alongside RMSE, MAE and F5. `Prob_Intervals` reports central 80% and 90% distribution intervals, with exact-failure-only coverage and width over all test conditions. `Prob_Predictions` contains the sample-level distributions and quantiles; `Prob_Calibration` records training-only calibration predictions. `Prob_Paired_CI`, `Prob_Campaigns` and `Prob_Method` document the comparison and its definitions.\n\nOrdinary regressors receive a Normal distribution on log10 life with a model-specific scale estimated from development predictions under their selected configuration. Existing primary WB-PIML, physical and selected-control scales and the native AFT distributions are retained. Factor-holdout WB receives a new development OOF scale; factor WB-PIML/ML-Strong use their selected development OOF predictions. Weibull location is not its median; the original point prediction remains unchanged. The six existing factor-holdout models are compared probabilistically, while the two native AFT models are compared in primary validation.\n\nCoverage is observed only for exact failures. Censoring selection prevents treating this as unconditional coverage of latent lifetimes; runout stopping times are evaluated through survival probabilities. The calibrated scale combines specimen scatter and prediction error and does not separate aleatoric from epistemic uncertainty. Probability scores supplement the point results.\n\n`09_probability_comparison.png` summarizes the primary probability comparison. The existing eight figures and their presentation retain their original roles.\n"""
    text += """

## Simple-trunk comparisons

WB-Residual uses the classic Walker-Basquin trunk (18 configurations); Basquin-Residual fixes gamma=0 (6 configurations). Each independently selects from the same three residual families and two gains on the original development campaign folds. The environmental penalty is absent. Basquin-Residual retains R and the other original residual inputs; it only removes Walker correction from the trunk. Both use development OOF censored-scale calibration and empirical interval calibration.

The two controls enter primary validation and all four factor holdouts, with point, probability, and interval results in the existing tables. Trunk_Metrics, Trunk_Paired_CI, Trunk_Empirical_PI and Trunk_Method collect the comparisons; 10_trunk_comparison.png summarizes them. Trunk_External_Candidates retains all added factor-holdout candidate scores. Original fixed-configuration repeated validation is unchanged, so its new-control console entries are blank.

Positive gains in Trunk_Paired_CI favor the model named proposed, including the WB-Residual versus Basquin-Residual comparison. These are revision-stage comparisons on previously studied data. They do not select a new primary model automatically.
"""
    readme_path.write_text(text, encoding="utf-8")


# Formal comparisons of the physical trunk; the primary candidate pool is unchanged.
SIMPLE_HYBRID_MODELS = ("WB-Residual", "Basquin-Residual")


def simple_hybrid_candidate_grid():
    candidates = []
    next_id = max(c["candidate_id"] for c in STRUCTURAL_CANDIDATES) + 1
    for model, gammas in zip(SIMPLE_HYBRID_MODELS, (STRUCTURAL_GAMMAS, (0.0,))):
        for family, gamma, eta in itertools.product(HYBRID_FAMILIES, gammas, RESIDUAL_GAIN_GRID):
            candidates.append({
                "candidate_id": next_id + len(candidates), "model": model,
                "pool": "simple_hybrid", "family": family,
                "gamma": float(gamma), "eta": float(eta),
                # Required by the shared fit interface; never a search dimension.
                "strength": 0.0,
            })
    return tuple(candidates)


SIMPLE_HYBRID_CANDIDATES = simple_hybrid_candidate_grid()


def predict_simple_hybrid_configs(train, valid, seed, configs, trees=400):
    """Fit each WB/Basquin trunk and residual basis once, then apply both gains."""
    require_exact_training(train, "predict_simple_hybrid_configs")
    if not len(valid):
        raise ValueError("Simple-hybrid validation rows must be nonempty")
    exact = train.loc[train.is_exact].reset_index(drop=True)
    full = pd.concat([exact, valid], ignore_index=True)
    y = exact.logN.to_numpy(float)
    prep = make_preprocessor()
    x = prep.fit_transform(exact)
    xv = prep.transform(valid)
    support = residual_policy_weight(train, valid, "ET_residual", "soft250")
    physics, residuals, predictions = {}, {}, {}
    for c in configs:
        model, family = c["model"], c["family"]
        gamma, eta = float(c["gamma"]), float(c["eta"])
        allowed_gammas = STRUCTURAL_GAMMAS if model == "WB-Residual" else (0.0,)
        if (model not in SIMPLE_HYBRID_MODELS or family not in HYBRID_FAMILIES
                or gamma not in allowed_gammas or eta not in RESIDUAL_GAIN_GRID
                or float(c.get("strength", 0.0)) != 0.0):
            raise ValueError(f"Invalid simple-hybrid candidate: {c}")
        if gamma not in physics:
            coef = fit_physical(train, CLASSIC_WB_SPEC, gamma)
            base = predict_physical(full, CLASSIC_WB_SPEC, coef, gamma)
            if not np.isfinite(coef).all() or not np.isfinite(base).all():
                raise FloatingPointError("Nonfinite simple physical fit")
            physics[gamma] = (base, coef)
        base, coef = physics[gamma]
        key = (gamma, family)
        if key not in residuals:
            target = y - base[:len(exact)]
            if family == "Matern_residual":
                correction = kernel_predict(x, xv, target, 2.0, 1.0, False)
            else:
                xx, xxv = x, xv
                if family == "physics_feature_ET_residual":
                    aug = full.copy()
                    mechanical = coef[0] - coef[1] * walker_log_stress(full, gamma)
                    aug["mechanical_prediction"] = mechanical
                    aug["environment_decrement"] = mechanical - base
                    aug["walker_stress"] = walker_log_stress(full, gamma)
                    extra = ("mechanical_prediction", "environment_decrement", "walker_stress")
                    aug_prep = make_preprocessor(extra)
                    xx = aug_prep.fit_transform(aug.iloc[:len(exact)])
                    xxv = aug_prep.transform(aug.iloc[len(exact):])
                estimator = ExtraTreesRegressor(
                    n_estimators=trees, min_samples_leaf=3, max_features=.85,
                    random_state=seed, n_jobs=TREE_JOBS)
                correction = estimator.fit(xx, target).predict(xxv)
            if not np.isfinite(correction).all():
                raise FloatingPointError(f"Nonfinite simple residual basis: {key}")
            residuals[key] = correction
        prediction = base[len(exact):] + eta * support * residuals[key]
        if np.shape(prediction) != (len(valid),) or not np.isfinite(prediction).all():
            raise FloatingPointError(f"Invalid predictions: candidate {c['candidate_id']}")
        predictions[int(c["candidate_id"])] = prediction
    return predictions


def nested_select_simple_controls(train, seed, trees):
    """Independently select the two simple trunks using development campaigns only.

    Returns model -> (choice, selected OOF rows, candidate table). Candidate-scale
    calibration and the RMSE-budget/runout-NLL rule match the primary selector.
    """
    require_exact_training(train, "nested_select_simple_controls")
    splits, _ = grouped_folds(train, min(4, train[PRIMARY_GROUP_COLUMN].nunique()), seed)
    columns = ["row_id", "source_id", "campaign_id", "record_equivalence_id",
               "logN", "is_exact", "is_runout", "temp_bin"]
    stored = {c["candidate_id"]: [] for c in SIMPLE_HYBRID_CANDIDATES}
    for inner_fold, (fit_idx, valid_idx) in enumerate(splits, 1):
        fit = train.iloc[fit_idx].reset_index(drop=True)
        valid = train.iloc[valid_idx].reset_index(drop=True)
        if set(fit.campaign_id) & set(valid.campaign_id):
            raise AssertionError("Inner campaign leakage in simple-trunk comparison")
        fit_seed = int(seed) + 100 * inner_fold + 1
        predictions = predict_simple_hybrid_configs(
            fit, valid, fit_seed, SIMPLE_HYBRID_CANDIDATES, trees)
        for cid, prediction in predictions.items():
            part = valid[columns].copy()
            part["inner_fold"], part["mu"], part["fit_seed"] = inner_fold, prediction, fit_seed
            stored[cid].append(part)
    rows = []
    for c in SIMPLE_HYBRID_CANDIDATES:
        cid = int(c["candidate_id"])
        oof = pd.concat(stored[cid], ignore_index=True)
        if (len(oof) != len(train) or not oof.row_id.is_unique
                or set(oof.row_id) != set(train.row_id)):
            raise AssertionError("Every development row needs exactly one simple-hybrid OOF prediction")
        if not np.isfinite(oof.mu.to_numpy(float)).all():
            raise FloatingPointError(f"Nonfinite simple-hybrid OOF: candidate {cid}")
        stored[cid] = oof
        exact = oof.loc[oof.is_exact].copy()
        exact["pred_logN"] = exact.mu
        score = sb_metrics([g.reset_index(drop=True) for _, g in exact.groupby(PRIMARY_GROUP_COLUMN)])
        scale = calibrate_aft_scale(oof)
        required_scores = [score["SB_RMSE"], scale["sigma"], scale["joint_censored_NLL"]]
        if train.is_runout.any():
            required_scores.append(scale["runout_survival_NLL"])
        if not np.isfinite(required_scores).all() or scale["sigma"] <= 0:
            raise FloatingPointError(f"Nonfinite simple-hybrid candidate scores: {cid}")
        n_candidates = sum(cc["model"] == c["model"] for cc in SIMPLE_HYBRID_CANDIDATES)
        rows.append({
            **c, "candidate_index": cid, "basis_id": (cid-SIMPLE_HYBRID_CANDIDATES[0]["candidate_id"])//2,
            "min_samples_leaf": 3, "max_features": .85, "residual_mode": c["family"],
            "ridge_alpha": 1.0, "kernel_length_scale": 2.0,
            "kernel_parameters_applicable": c["family"] == "Matern_residual",
            "temperature_scale": 250.0, "temperature_policy": "soft250",
            "physical_spec": CLASSIC_WB_SPEC.key, "n_physics_parameters": 2,
            "walker_correction_enabled": c["model"] == "WB-Residual",
            "gamma_selected": c["model"] == "WB-Residual",
            "environment_regularization_applicable": False,
            "environment_penalty_searched": False,
            "candidate_count": n_candidates, "selection_seed": int(seed),
            "inner_basis_seed_offset": 0,
            "is_reference": False, "censor_feasible": False,
            "censor_feasibility_assessed": False,
            "inner_SB_RMSE": score["SB_RMSE"],
            "inner_runout_hinge_RMSE": censored_hinge_rmse(oof),
            "inner_joint_censored_NLL": float(scale["joint_censored_NLL"]),
            "inner_runout_survival_NLL": float(scale["runout_survival_NLL"]),
            "inner_sigma": float(scale["sigma"]),
        })
    table = pd.DataFrame(rows)
    outputs = {}
    for model in SIMPLE_HYBRID_MODELS:
        choice, candidates = select_budgeted_candidate(table.loc[table.model.eq(model)])
        # The primary selector never uses feasibility for selection; a new,
        # unused noninferiority reference fit would add no information here.
        choice["censor_feasible"] = np.nan
        choice["censor_constraint_relaxed"] = False
        candidates["censor_feasible"] = np.nan
        candidates["censor_constraint_relaxed"] = False
        outputs[model] = (choice, stored[int(choice["candidate_index"])], candidates)
    return outputs


def fit_selected_simple_control(train, valid, choice, seed, trees):
    """Refit a selected simple trunk with the existing hybrid implementation."""
    config = next((c for c in SIMPLE_HYBRID_CANDIDATES
                   if c["candidate_id"] == int(choice["candidate_id"])), None)
    if config is None or any(choice[k] != config[k]
                             for k in ("model", "family", "gamma", "eta", "strength")):
        raise ValueError("Selected simple-hybrid configuration is not in its declared pool")
    prediction, coefficients, metadata = fit_selected_hybrid(
        train, valid, choice, seed, trees, spec=CLASSIC_WB_SPEC)
    if (np.shape(prediction) != (len(valid),) or not np.isfinite(prediction).all()
            or not np.isfinite(coefficients).all()):
        raise FloatingPointError(f"Nonfinite selected simple-hybrid fit: {choice['model']}")
    metadata.update({
        "physical_spec": CLASSIC_WB_SPEC.key, "n_physics_parameters": 2,
        "walker_correction_enabled": choice["model"] == "WB-Residual",
        "gamma_selected": choice["model"] == "WB-Residual",
        "environment_penalty_searched": False,
    })
    return prediction, coefficients, metadata

def fit_simple_control_models(train, test, selection_seed, fit_seed, trees):
    """Select on development data, refit, and freeze predictions for one test set."""
    if set(train.row_id) & set(test.row_id):
        raise AssertionError("Simple-trunk train/test row overlap")
    results = {}
    for name, (choice, oof, candidates) in nested_select_simple_controls(
            train, selection_seed, trees).items():
        mu, coef, metadata = fit_selected_simple_control(train, test, choice, fit_seed, trees)
        scale = calibrate_aft_scale(oof)
        half80, half90 = empirical_interval_halfwidths(oof)
        results[name] = dict(choice=choice, oof=oof, candidates=candidates,
                             mu=mu, coef=coef, metadata=metadata, scale=scale,
                             half80=half80, half90=half90,
                             supported=exact_temperature_residual_gate(train, test).astype(bool))
    return results


def simple_control_prediction_part(test, columns, name, result):
    part = test[[column for column in columns if column in test.columns]].copy()
    part["model"] = name
    part["pred_logN"] = result["mu"]
    part["sigma"] = float(result["scale"]["sigma"])
    part["temperature_policy"] = "soft250"
    part["residual_temperature_supported"] = result["supported"]
    for level in (80, 90):
        half = result[f"half{level}"]
        part[f"lower{level}"] = result["mu"] - half
        part[f"upper{level}"] = result["mu"] + half
    return part


def extract_simple_external_candidates(selection):
    """Expand the worker payload into a readable candidate table for the workbook."""
    records = []
    for _, row in selection.loc[selection.model.isin(SIMPLE_CONTROL_MODELS)].iterrows():
        candidates = json.loads(row["candidate_scores_json"])
        if len(candidates) != int(row["n_candidates_evaluated"]):
            raise AssertionError("Incomplete simple-trunk external candidate records")
        if sum(bool(c["selected"]) for c in candidates) != 1:
            raise AssertionError("Each external control needs exactly one selected candidate")
        common = {key: row[key] for key in ("scenario", "holdout_group", "holdout_column", "fold", "repeat")}
        records.extend({**common, **candidate} for candidate in candidates)
    return selection.drop(columns=["candidate_scores_json"]), pd.DataFrame(records)


def simple_control_tables(primary, external, probability, bootstrap):
    """Collect trunk comparisons without choosing a new primary model."""
    names = (PROPOSED_MODEL,) + SIMPLE_CONTROL_MODELS
    scopes = [("Primary", primary)] + [
        (scenario, external.loc[external.strategy.eq(scenario)])
        for scenario in ("LOSO", "LOAO", "LOTO", "LOEO")
        if scenario in set(external.strategy)
    ]
    summary, empirical, effects = [], [], []
    comparisons = ((PROPOSED_MODEL, "WB-Residual"),
                   (PROPOSED_MODEL, "Basquin-Residual"),
                   ("WB-Residual", "Basquin-Residual"))
    for scope_index, (scenario, frame) in enumerate(scopes):
        exact = frame.loc[frame.is_exact]
        for model_index, name in enumerate(names):
            subset = exact.loc[exact.model.eq(name)]
            if subset.empty or not subset.row_id.is_unique:
                raise AssertionError(f"Incomplete trunk comparison: {scenario}/{name}")
            score = sb_metrics([g.reset_index(drop=True) for _, g in subset.groupby(PRIMARY_GROUP_COLUMN)])
            summary.append(dict(scenario=scenario, model=name, n_exact=len(subset),
                                n_campaigns=subset.campaign_id.nunique(), **score))
            # Reuse the empirical-interval estimator with an isolated model frame.
            isolated = frame.loc[frame.model.eq(name)].copy()
            isolated["model"] = PROPOSED_MODEL
            pi_seed = (SEED + 3 if scenario == "Primary" else
                       SEED + 940000 + 10000 * sorted(set(external.strategy)).index(scenario))
            table = uq_summary(isolated, bootstrap, pi_seed + model_index * 101)
            table.insert(0, "model", name)
            table.insert(0, "scenario", scenario)
            empirical.append(table)
        for pair_index, (proposed, comparator) in enumerate(comparisons):
            isolated = frame.loc[frame.model.isin((proposed, comparator))].copy()
            identity = ["row_id", "campaign_id", "source_id", "record_equivalence_id", "logN", "is_exact", "is_runout"]
            a = isolated.loc[isolated.model.eq(proposed), identity].sort_values("row_id").reset_index(drop=True)
            b = isolated.loc[isolated.model.eq(comparator), identity].sort_values("row_id").reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False)
            isolated.loc[isolated.model.eq(proposed), "model"] = PROPOSED_MODEL
            # Match the corresponding existing summary when the pair is already reported.
            if proposed == PROPOSED_MODEL:
                control_index = SIMPLE_CONTROL_MODELS.index(comparator)
                pair_seed = (SEED + 20 + len(EXTERNAL_MODELS) + control_index
                    if scenario == "Primary" else SEED + 930000
                    + 10000 * sorted(set(external.strategy)).index(scenario)
                    + 101 * (len(PRIMARY_COMPARATORS) + control_index))
            else:
                pair_seed = SEED + 1100000 + 10000 * scope_index + 101 * pair_index
            table = paired_effects(isolated, bootstrap, pair_seed, comparator)
            table.insert(0, "proposed", proposed)
            table.insert(0, "scenario", scenario)
            table["gain_definition"] = "positive favors proposed; errors: comparator-proposed; R2/F5: proposed-comparator"
            table["uncertainty_scope"] = "conditional campaign bootstrap on previously studied data"
            effects.append(table)

    # Full-model probability pairs already exist; additionally compare the two simple trunks.
    prob_pairs = probability["Prob_Paired_CI"]
    effects.append(prob_pairs.loc[prob_pairs.comparator.isin(SIMPLE_CONTROL_MODELS)].copy())
    rows = probability["Prob_Predictions"]
    campaigns = probability["Prob_Campaigns"]
    simple_rows = rows.loc[rows.model.isin(SIMPLE_CONTROL_MODELS)].copy()
    simple_campaigns = campaigns.loc[campaigns.model.isin(SIMPLE_CONTROL_MODELS)].copy()
    for frame in (simple_rows, simple_campaigns):
        frame.loc[frame.model.eq("WB-Residual"), "model"] = PROPOSED_MODEL
    pairs = probability_paired_effects(simple_rows, simple_campaigns, bootstrap, SEED + 1200000)
    pairs["proposed"] = "WB-Residual"
    pairs["gain_definition"] = "Basquin-Residual minus WB-Residual; positive favors WB-Residual"
    effects.append(pairs)
    prob_summary = probability["Prob_Metrics"]
    prob_summary = prob_summary.loc[prob_summary.model.isin(names) & prob_summary.aggregation_level.eq("overall")]
    columns = ["scenario", "model", "SB_joint_censored_NLL", "SB_exact_NLL", "SB_runout_NLL"]
    merged = pd.DataFrame(summary).merge(prob_summary[columns], on=["scenario", "model"], validate="one_to_one")
    methods = pd.DataFrame([
        ("WB-Residual", "Classic Walker-Basquin trunk; 18 configurations: 3 residual families x 3 gamma values x 2 gains."),
        ("Basquin-Residual", "Same two-parameter trunk with gamma=0; 6 configurations: 3 residual families x 2 gains. R, Sa and Smean remain residual inputs."),
        ("Selection", "Independent within each model; original campaign splits, fit seeds and RMSE+0.020 budget, followed by minimum calibrated runout NLL. No environmental penalty search."),
        ("Training", "Physical fit uses exact fractures and right-censored runouts; residual fit uses exact fractures only. Temperature support is unchanged."),
        ("Calibration", "Each selected model estimates its own constant log-life scale and empirical interval widths from development OOF predictions. Test outcomes are not used."),
        ("Interpretation", "Comparisons evaluate the complete independently selected pipelines; selected residual families may differ between trunks."),
        ("Probability intervals", "Central 80%/90% probability intervals are in Prob_Intervals and Prob_Predictions; they are separate from Trunk_Empirical_PI."),
        ("Point and interval rows", "Exact_Outer_Pred and Runout_Outer_Pred include primary new-control predictions. Prob_Predictions also retains factor point predictions and both empirical and probability interval bounds."),
        ("Candidate records", "Primary: Nested_Candidates. Factor holdouts: Trunk_External_Candidates. Each selected model retains its OOF rows in Prob_Calibration."),
        ("Fixed repeated validation", "Original repeated fixed-configuration analyses are unchanged; new controls use primary nested validation and factor holdouts only."),
        ("Comparison origin", "Revision-stage comparisons motivated by previously inspected results; no new independent test dataset. Bootstrap intervals condition on the studied campaigns."),
        ("Primary model", "WB-PIML remains the primary model. No primary-model change is made automatically from these comparisons."),
    ], columns=["item", "definition"])
    return {"Trunk_Metrics": merged, "Trunk_Paired_CI": pd.concat(effects, ignore_index=True, sort=False),
            "Trunk_Empirical_PI": pd.concat(empirical, ignore_index=True), "Trunk_Method": methods}


def make_trunk_comparison_figure(out, tables):
    names = (PROPOSED_MODEL,) + SIMPLE_CONTROL_MODELS
    table = tables["Trunk_Metrics"].set_index(["scenario", "model"])
    colors = ("#df8525", "#477c97", "#6b9662")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric, title in zip(axes[0], ("SB_RMSE", "SB_joint_censored_NLL"),
                               ("(a) Primary exact-fracture RMSE", "(b) Primary full censored NLL")):
        ax.bar(np.arange(3), [table.loc[("Primary", name), metric] for name in names], color=colors)
        ax.set_xticks(np.arange(3), names, rotation=15)
        ax.set_title(title)
        ax.set_ylabel("Lower is better")
    pair = tables["Trunk_Paired_CI"]
    pair = pair.loc[pair.scenario.eq("Primary") & pair.effect.eq("joint_censored_NLL_gain")]
    for position, row in enumerate(pair.itertuples()):
        axes[1, 0].plot([row.CI95_lo, row.CI95_hi], [position, position], color=colors[position])
        axes[1, 0].plot(row.estimate, position, "o", color=colors[position])
    axes[1, 0].set_yticks(np.arange(len(pair)), [f"{r.proposed}\nvs {r.comparator}" for r in pair.itertuples()], fontsize=8)
    axes[1, 0].axvline(0, color="black", lw=.8)
    axes[1, 0].set_title("(c) Paired NLL gains and 95% intervals")
    axes[1, 0].set_xlabel("Positive favors the first-named model")
    scenarios = [s for s in ("LOSO", "LOAO", "LOTO", "LOEO") if s in table.index.get_level_values(0)]
    for index, name in enumerate(names):
        axes[1, 1].bar(np.arange(len(scenarios)) + (index-1)*.25,
                      [table.loc[(s, name), "SB_RMSE"] for s in scenarios], .25, color=colors[index], label=name)
    axes[1, 1].set_xticks(np.arange(len(scenarios)), scenarios)
    axes[1, 1].set_title("(d) Factor-held-out exact-fracture RMSE")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(axis="x" if ax is axes[1, 0] else "y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Physical-trunk comparisons with independently selected residual configurations")
    fig.tight_layout()
    fig.savefig(out / "10_trunk_comparison.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
