def inner_oof(train: pd.DataFrame, spec: PhysicalSpec, seed: int) -> pd.DataFrame:
    require_exact_training(train, f"inner_oof[{spec.key}]")
    splits, _ = grouped_folds(train, min(4, train[PRIMARY_GROUP_COLUMN].nunique()), seed)
    pieces = []
    for fold, (fit_idx, valid_idx) in enumerate(splits, 1):
        fit = train.iloc[fit_idx].reset_index(drop=True)
        valid = train.iloc[valid_idx].reset_index(drop=True)
        coef = fit_physical(fit, spec)
        part = valid[["row_id", "source_id", "campaign_id", "record_equivalence_id",
                      "logN", "is_exact", "is_runout"]].copy()
        part["inner_fold"] = fold
        part["mu"] = predict_physical(valid, spec, coef)
        pieces.append(part)
    oof = pd.concat(pieces, ignore_index=True)
    if oof["row_id"].duplicated().any() or set(oof["row_id"]) != set(train["row_id"]):
        raise AssertionError("Inner OOF must predict each development row once")
    return oof


def calibrate_aft_scale(oof: pd.DataFrame) -> dict[str, float]:
    require_exact_training(oof, "calibrate_aft_scale")
    exact = oof.loc[oof["is_exact"]].reset_index(drop=True)
    ew = training_weights(exact)

    def objective(value: np.ndarray) -> float:
        sigma = float(np.exp(value[0]))
        return joint_aft_nll(oof, oof["mu"].to_numpy(float), sigma)["joint_censored_NLL"]

    start = math.log(np.clip(np.sqrt(np.average((exact["logN"] - exact["mu"]) ** 2, weights=ew)), 0.10, 3.0))
    result = minimize(objective, np.array([start]), method="L-BFGS-B", bounds=[(math.log(0.08), math.log(4.0))])
    if not result.success:
        raise RuntimeError("AFT scale calibration failed")
    sigma = float(np.exp(result.x[0]))
    losses = joint_aft_nll(oof, oof["mu"].to_numpy(float), sigma)
    return {"sigma": sigma, "calibration_objective": float(result.fun), **losses}


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = np.asarray(values, float)[order]
    weights = np.asarray(weights, float)[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(quantile, cumulative, values))


def make_preprocessor(extra_numeric: tuple[str, ...] = ()) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [("numeric", StandardScaler(), TRAD_NUMERIC + list(extra_numeric)),
         ("categorical", encoder, TRAD_CATEGORICAL)],
        sparse_threshold=0.0,
    )


def assert_residual_preprocessor(preprocessor):

    allowed = set(RESIDUAL_NUMERIC + RESIDUAL_CATEGORICAL + RESIDUAL_PHYSICS_NUMERIC)
    for name, transformer, columns in preprocessor.transformers:
        if name == "remainder":
            continue
        if not isinstance(columns, (tuple, list)) or any(c not in allowed for c in columns):
            raise ValueError(f"Residual inputs contain non-whitelisted features: {columns}")
    if preprocessor.remainder != "drop":
        raise ValueError("Residual preprocessing must drop all unused columns")


def make_residual_preprocessor(extra_numeric=()):

    extra_numeric = tuple(extra_numeric)
    if len(set(extra_numeric)) != len(extra_numeric) or any(
            c not in RESIDUAL_PHYSICS_NUMERIC for c in extra_numeric):
        raise ValueError(f"Invalid residual physics inputs: {extra_numeric}")
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    prep = ColumnTransformer(
        [("numeric", StandardScaler(), list(RESIDUAL_NUMERIC + extra_numeric)),
         ("categorical", encoder, list(RESIDUAL_CATEGORICAL))],
        remainder="drop", sparse_threshold=0.0)
    assert_residual_preprocessor(prep)
    return prep


def residual_design_frame(frame, coefficients, gamma, family):


    augmented = frame.copy()
    extra = ()
    if family == "physics_feature_ET_residual":
        augmented["mechanical_prediction"] = (
            coefficients[0] - coefficients[1] * walker_log_stress(frame, gamma))
        augmented["walker_stress"] = walker_log_stress(frame, gamma)
        extra = RESIDUAL_PHYSICS_NUMERIC
    return augmented, extra


def censored_hinge_rmse(frame: pd.DataFrame, prediction_column: str = "mu") -> float:

    runout = frame.loc[frame["is_runout"]].reset_index(drop=True)
    if not len(runout):
        return 0.0
    violation = np.maximum(
        runout["logN"].to_numpy(float) - runout[prediction_column].to_numpy(float),
        0.0,
    )
    weight = analysis_weights(runout)
    return float(np.sqrt(np.sum(weight * violation**2) / np.sum(weight)))


def choose_rmse_first(candidates: pd.DataFrame) -> pd.Series:

    best_rmse = float(candidates["inner_SB_RMSE"].min())
    eligible = candidates.loc[
        candidates["inner_SB_RMSE"] <= best_rmse + RMSE_TIE_TOLERANCE
    ].copy()
    sort_columns = [
        column for column in (
            "inner_joint_censored_NLL", "calibration_complexity",
            "physics_mix", "min_samples_leaf", "candidate_index",
        ) if column in eligible.columns
    ]
    if not sort_columns:
        sort_columns = ["inner_SB_RMSE"]
    return eligible.sort_values(sort_columns, na_position="last").iloc[0]


def apply_safe_fusion(et_prediction: np.ndarray, physical_prediction: np.ndarray,
                      physics_mix: float, calibration_intercept: float,
                      calibration_slope: float) -> np.ndarray:

    if not (
        np.isclose(physics_mix, 1.0)
        and np.isclose(calibration_intercept, 0.0)
        and np.isclose(calibration_slope, 1.0)
    ):
        raise AssertionError("WB-PIML cannot switch off physics or calibrate outer predictions")
    if len(et_prediction) != len(physical_prediction):
        raise AssertionError("Safe-fusion branches must cover identical rows")
    return np.asarray(physical_prediction, float)


def select_safe_fusion(physical_oof: pd.DataFrame, et_oof: pd.DataFrame
                       ) -> tuple[dict[str, float | str], pd.DataFrame, pd.DataFrame]:


    if PHYSICS_MIX_GRID != (1.0,) or CALIBRATION_MODES != ("none",):
        raise AssertionError("WB-PIML requires an always-active, uncalibrated physical trunk")
    metadata = [
        "row_id", "source_id", "campaign_id", "record_equivalence_id",
        "logN", "is_exact", "is_runout", "inner_fold",
    ]
    physical = physical_oof[metadata + ["mu"]].rename(columns={"mu": "physical_mu"})
    direct = et_oof[["row_id", "mu"]].rename(columns={"mu": "et_mu"})
    merged = physical.merge(direct, on="row_id", how="inner", validate="one_to_one")
    if len(merged) != len(physical_oof):
        raise AssertionError("Safe-fusion OOF branches do not cover identical development rows")

    candidate = merged.copy()
    candidate["mu"] = candidate["physical_mu"].to_numpy(float)
    exact = candidate.loc[candidate["is_exact"]].copy()
    exact["pred_logN"] = exact["mu"]
    exact_score = sb_metrics([
        group.reset_index(drop=True)
        for _, group in exact.groupby(PRIMARY_GROUP_COLUMN)
    ])["SB_RMSE"]
    censor_score = censored_hinge_rmse(candidate)
    scale = calibrate_aft_scale(candidate)
    candidates = pd.DataFrame([{
        "physics_mix": 1.0,
        "calibration_mode": "none",
        "calibration_intercept": 0.0,
        "calibration_slope": 1.0,
        "calibration_complexity": 0,
        "inner_SB_RMSE": float(exact_score),
        "inner_runout_hinge_RMSE": float(censor_score),
        "inner_joint_censored_NLL": float(scale["joint_censored_NLL"]),
        "within_RMSE_tolerance": True,
    }])
    best = candidates.iloc[0]
    selected_oof = candidate[metadata + ["mu"]].copy()
    choice: dict[str, float | str] = {
        "physics_mix": 1.0,
        "calibration_mode": "none",
        "calibration_intercept": float(best["calibration_intercept"]),
        "calibration_slope": float(best["calibration_slope"]),
        "inner_SB_RMSE": float(best["inner_SB_RMSE"]),
        "inner_runout_hinge_RMSE": float(best["inner_runout_hinge_RMSE"]),
        "inner_joint_censored_NLL": float(best["inner_joint_censored_NLL"]),
        "RMSE_tolerance": RMSE_TIE_TOLERANCE,
    }
    candidates = candidates.sort_values(
        ["within_RMSE_tolerance", "inner_SB_RMSE", "inner_joint_censored_NLL"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return choice, selected_oof, candidates


def fit_extra_trees(train: pd.DataFrame, valid: pd.DataFrame, seed: int, trees: int,
                    min_samples_leaf: int = 3,
                    max_features: float = 0.85) -> np.ndarray:
    require_exact_training(train, "fit_extra_trees")
    exact = train.loc[train["is_exact"]].reset_index(drop=True)
    estimator = ExtraTreesRegressor(
        n_estimators=trees, min_samples_leaf=min_samples_leaf,
        max_features=max_features, random_state=seed, n_jobs=TREE_JOBS,
    )
    pipe = Pipeline([("preprocess", clone(make_preprocessor())), ("model", estimator)])
    pipe.fit(exact, exact["logN"], model__sample_weight=training_weights(exact))
    return np.asarray(pipe.predict(valid), float)


def nested_select_extra_trees(train: pd.DataFrame, seed: int, trees: int
                              ) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    require_exact_training(train, "nested_select_extra_trees")
    splits, _ = grouped_folds(train, min(4, train[PRIMARY_GROUP_COLUMN].nunique()), seed)
    unique_configs = sorted({(leaf, features) for _, leaf, features, _ in NESTED_CANDIDATES})
    rows = []
    stored: dict[int, pd.DataFrame] = {}
    for config_index, (min_leaf, max_features) in enumerate(unique_configs):
        pieces = []
        for inner_fold, (fit_idx, valid_idx) in enumerate(splits, 1):
            fit = train.iloc[fit_idx].reset_index(drop=True)
            valid = train.iloc[valid_idx].reset_index(drop=True)
            mu = fit_extra_trees(
                fit, valid, seed + 10000 * config_index + 100 * inner_fold + 7,
                min(trees, 300), min_leaf, max_features,
            )
            part = valid[["row_id", "source_id", "campaign_id", "record_equivalence_id",
                          "logN", "is_exact", "is_runout"]].copy()
            part["inner_fold"] = inner_fold
            part["mu"] = mu
            pieces.append(part)
        oof = pd.concat(pieces, ignore_index=True)
        exact = oof.loc[oof["is_exact"]].copy()
        exact["pred_logN"] = exact["mu"]
        score = sb_metrics([g.reset_index(drop=True) for _, g in exact.groupby(PRIMARY_GROUP_COLUMN)])["SB_RMSE"]
        rows.append({"config_index": config_index, "min_samples_leaf": min_leaf,
                     "max_features": max_features, "inner_SB_RMSE": score})
        stored[config_index] = oof
    candidates = pd.DataFrame(rows).sort_values(
        ["inner_SB_RMSE", "min_samples_leaf", "config_index"]
    ).reset_index(drop=True)
    best = candidates.iloc[0]
    chosen = {"config_index": int(best["config_index"]),
              "min_samples_leaf": int(best["min_samples_leaf"]),
              "max_features": float(best["max_features"]),
              "inner_SB_RMSE": float(best["inner_SB_RMSE"])}
    return chosen, stored[chosen["config_index"]], candidates


def fit_et_walker(train: pd.DataFrame, valid: pd.DataFrame, seed: int, trees: int,
                  gamma: float, min_samples_leaf: int,
                  max_features: float) -> np.ndarray:

    require_exact_training(train, "fit_et_walker")
    exact = train.loc[train["is_exact"]].reset_index(drop=True).copy()
    exact["walker_log_stress"] = walker_log_stress(exact, gamma)
    valid_aug = valid.copy()
    valid_aug["walker_log_stress"] = walker_log_stress(valid_aug, gamma)

    def new_pipe() -> Pipeline:
        return Pipeline([
            ("preprocess", make_preprocessor(("walker_log_stress",))),
            ("model", ExtraTreesRegressor(
                n_estimators=trees, min_samples_leaf=min_samples_leaf,
                max_features=max_features, random_state=seed, n_jobs=TREE_JOBS,
            )),
        ])

    pipe = new_pipe()
    pipe.fit(exact, exact["logN"], model__sample_weight=training_weights(exact))
    return np.asarray(pipe.predict(valid_aug), float)


def nested_select_et_walker(train: pd.DataFrame, seed: int, trees: int
                            ) -> tuple[dict[str, float], pd.DataFrame]:

    require_exact_training(train, "nested_select_et_walker")
    splits, _ = grouped_folds(train, min(4, train[PRIMARY_GROUP_COLUMN].nunique()), seed)
    rows = []
    for candidate_index, (gamma, min_leaf, max_features, eta_budget) in enumerate(NESTED_CANDIDATES):
        pieces = []
        for inner_fold, (fit_idx, valid_idx) in enumerate(splits, 1):
            fit = train.iloc[fit_idx].reset_index(drop=True)
            valid = train.iloc[valid_idx].reset_index(drop=True)
            mu = fit_et_walker(
                fit, valid,
                seed + 10000 * candidate_index + 100 * inner_fold + 1,
                min(trees, 300), gamma, min_leaf, max_features,
            )
            part = valid[["row_id", "source_id", "campaign_id", "record_equivalence_id",
                          "logN", "is_exact", "is_runout"]].copy()
            part["mu"] = mu
            part["inner_fold"] = inner_fold
            pieces.append(part)
        oof = pd.concat(pieces, ignore_index=True)
        exact = oof.loc[oof["is_exact"]].copy()
        exact["pred_logN"] = exact["mu"]
        exact_score = sb_metrics([g.reset_index(drop=True) for _, g in exact.groupby(PRIMARY_GROUP_COLUMN)])["SB_RMSE"]
        censor_score = censored_hinge_rmse(oof)
        scale = calibrate_aft_scale(oof)
        rows.append({
            "candidate_index": candidate_index, "gamma": gamma,
            "min_samples_leaf": min_leaf, "max_features": max_features,
            "eta_budget_slot": eta_budget, "inner_SB_RMSE": exact_score,
            "inner_runout_hinge_RMSE": censor_score,
            "inner_joint_censored_NLL": scale["joint_censored_NLL"],
            "within_RMSE_tolerance": False,
        })
    candidates = pd.DataFrame(rows)
    best_rmse = float(candidates["inner_SB_RMSE"].min())
    candidates["within_RMSE_tolerance"] = (
        candidates["inner_SB_RMSE"] <= best_rmse + RMSE_TIE_TOLERANCE
    )
    best = choose_rmse_first(candidates)
    candidates = candidates.sort_values(
        ["within_RMSE_tolerance", "inner_SB_RMSE", "inner_joint_censored_NLL"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    chosen = {
        "candidate_index": int(best["candidate_index"]),
        "gamma": float(best["gamma"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
        "max_features": float(best["max_features"]),
        "eta_budget_slot": float(best["eta_budget_slot"]),
        "inner_SB_RMSE": float(best["inner_SB_RMSE"]),
        "inner_runout_hinge_RMSE": float(best["inner_runout_hinge_RMSE"]),
        "inner_joint_censored_NLL": float(best["inner_joint_censored_NLL"]),
        "RMSE_tolerance": RMSE_TIE_TOLERANCE,
    }
    return chosen, candidates


def external_predictions(train: pd.DataFrame, valid: pd.DataFrame, seed: int, trees: int,
                         et_min_samples_leaf: int = 3,
                         et_max_features: float = 0.85,
                         walker_gamma: float = GAMMA,
                         walker_min_samples_leaf: int = 3,
                         walker_max_features: float = 0.85) -> dict[str, np.ndarray]:
    require_exact_training(train, "external_predictions")
    exact = train.loc[train["is_exact"]].reset_index(drop=True)
    models = {
        "RF": RandomForestRegressor(n_estimators=trees, min_samples_leaf=3, max_features=0.75, random_state=seed, n_jobs=TREE_JOBS),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=et_min_samples_leaf,
                                           max_features=et_max_features, random_state=seed + 1, n_jobs=TREE_JOBS),
        "GBDT": GradientBoostingRegressor(n_estimators=180, learning_rate=0.025, max_depth=2, min_samples_leaf=4, loss="huber", random_state=seed + 2),
        "SVR": SVR(C=2.0, epsilon=0.15, gamma="scale"),
    }
    output = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocess", clone(make_preprocessor())), ("model", clone(estimator))])
        pipe.fit(exact, exact["logN"], model__sample_weight=training_weights(exact))
        output[name] = pipe.predict(valid)
    output["ET-Walker"] = fit_et_walker(
        train, valid, seed + 1, trees, walker_gamma,
        walker_min_samples_leaf, walker_max_features,
    )
    output["ML-Ens"] = np.mean(
        np.column_stack([output[name] for name in ["RF", "ExtraTrees", "GBDT", "SVR"]]),
        axis=1,
    )
    return output


def composite_score(error: np.ndarray, y: np.ndarray) -> float:
    q05, q95 = np.percentile(y, [5, 95]) if len(y) > 1 else (0.0, 1.0)
    scale = max(float(q95 - q05), LOG5)
    absolute = np.abs(error)
    return float(np.mean([
        np.clip(np.sqrt(np.mean(error**2)) / scale, 0, 1),
        np.clip(np.mean(absolute) / scale, 0, 1),
        np.mean(absolute > math.log10(3)), np.mean(absolute > LOG5),
        np.clip(abs(np.mean(error)) / scale, 0, 1),
    ]))


def _prepare_sb_groups(groups):
    prepared = []
    for group in groups:
        y = group["logN"].to_numpy(float)
        pred = group["pred_logN"].to_numpy(float)
        if "record_equivalence_id" in group.columns:
            equivalence_size = group.groupby("record_equivalence_id")[
                "record_equivalence_id"
            ].transform("size").to_numpy(float)
            weight = 1.0 / (
                max(group["record_equivalence_id"].nunique(), 1) * equivalence_size
            )
        else:
            weight = np.full(len(group), 1.0 / len(group))
        prepared.append((y, pred, weight, composite_score(pred - y, y)))
    return prepared


def _sb_metrics_prepared(groups):
    y = np.concatenate([g[0] for g in groups])
    pred = np.concatenate([g[1] for g in groups])
    weight = np.concatenate([g[2] for g in groups])
    error = pred - y
    mean_y = float(np.sum(weight * y) / np.sum(weight))
    denominator = float(np.sum(weight * (y - mean_y) ** 2))
    mse = float(np.sum(weight * error**2) / np.sum(weight))
    return {
        "SB_RMSE": math.sqrt(mse),
        "SB_MAE": float(np.sum(weight * np.abs(error)) / np.sum(weight)),
        "SB_R2": float(1.0 - np.sum(weight * error**2) / denominator) if denominator > 0 else np.nan,
        "SB_F5": float(np.sum(weight * (np.abs(error) <= LOG5)) / np.sum(weight)),
        "SB_C_star": float(np.mean([g[3] for g in groups])),
    }


def _sb_rmse_prepared(groups):
    y = np.concatenate([g[0] for g in groups])
    pred = np.concatenate([g[1] for g in groups])
    weight = np.concatenate([g[2] for g in groups])
    error = pred - y
    mse = float(np.sum(weight * error**2) / np.sum(weight))
    return math.sqrt(mse)


def sb_metrics(groups: list[pd.DataFrame]) -> dict[str, float]:

    return _sb_metrics_prepared(_prepare_sb_groups(groups))


def metric_summary(predictions: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    exact = predictions.loc[predictions["is_exact"]].copy()
    for model in legacy_first_models(exact["model"]):
        model_df = exact.loc[exact["model"].eq(model)]
        groups = _prepare_sb_groups([
            g.reset_index(drop=True) for _, g in model_df.groupby(PRIMARY_GROUP_COLUMN)
        ])
        point = _sb_metrics_prepared(groups)
        boots = {key: [] for key in point}
        for _ in range(bootstrap):
            sampled = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
            values = _sb_metrics_prepared(sampled)
            for key, value in values.items():
                boots[key].append(value)
        row = {"model": model, "n_exact": len(model_df),
               "n_campaigns": len(groups), "n_sources": model_df["source_id"].nunique(), **point}
        for key, values in boots.items():
            row[f"{key}_lo"] = float(np.quantile(values, 0.025))
            row[f"{key}_hi"] = float(np.quantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("SB_RMSE").reset_index(drop=True)


def paired_effects(predictions: pd.DataFrame, bootstrap: int, seed: int,
                   comparator: str = "ML-Ens") -> pd.DataFrame:
    exact = predictions.loc[predictions["is_exact"] & predictions["model"].isin(["WB-PIML", comparator])].copy()
    wide = exact.pivot(
        index=["campaign_id", "source_id", "record_equivalence_id", "row_id", "logN"],
        columns="model", values="pred_logN"
    ).reset_index()
    groups = [g.reset_index(drop=True) for _, g in wide.groupby(PRIMARY_GROUP_COLUMN)]
    prepared = {
        model: _prepare_sb_groups([
            g[["record_equivalence_id", "logN", model]].rename(columns={model: "pred_logN"})
            for g in groups
        ])
        for model in ["WB-PIML", comparator]
    }

    def effect(indices):
        metrics = {
            model: _sb_metrics_prepared([prepared[model][i] for i in indices])
            for model in ["WB-PIML", comparator]
        }
        return {
            "RMSE_gain": metrics[comparator]["SB_RMSE"] - metrics["WB-PIML"]["SB_RMSE"],
            "MAE_gain": metrics[comparator]["SB_MAE"] - metrics["WB-PIML"]["SB_MAE"],
            "R2_gain": metrics["WB-PIML"]["SB_R2"] - metrics[comparator]["SB_R2"],
            "F5_gain": metrics["WB-PIML"]["SB_F5"] - metrics[comparator]["SB_F5"],
            "C_star_gain": metrics[comparator]["SB_C_star"] - metrics["WB-PIML"]["SB_C_star"],
        }

    point = effect(range(len(groups)))
    rng = np.random.default_rng(seed)
    boot = {key: [] for key in point}
    for _ in range(bootstrap):
        indices = rng.integers(0, len(groups), len(groups))
        values = effect(indices)
        for key, value in values.items():
            boot[key].append(value)
    return pd.DataFrame([{"comparator": comparator, "effect": key, "estimate": value,
                          "CI95_lo": float(np.quantile(boot[key], 0.025)),
                          "CI95_hi": float(np.quantile(boot[key], 0.975)),
                          "positive_is_better": True} for key, value in point.items()])


def uq_summary(predictions: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    exact = predictions.loc[predictions["is_exact"] & predictions["model"].eq("WB-PIML")].copy()
    rng = np.random.default_rng(seed)
    rows = []
    for level in [80, 90]:
        low, high = f"lower{level}", f"upper{level}"
        exact["covered"] = (exact["logN"] >= exact[low]) & (exact["logN"] <= exact[high])
        exact["width"] = exact[high] - exact[low]
        alpha = 1.0 - level / 100.0
        exact["interval_score"] = exact["width"] + 2 / alpha * (exact[low] - exact["logN"]).clip(lower=0) + 2 / alpha * (exact["logN"] - exact[high]).clip(lower=0)
        groups = [g.reset_index(drop=True) for _, g in exact.groupby(PRIMARY_GROUP_COLUMN)]
        prepared = [
            (g["covered"].mean(), g["width"].mean(), g["interval_score"].mean())
            for g in groups
        ]

        def values(sampled):
            return {
                "coverage": float(np.mean([g[0] for g in sampled])),
                "mean_width": float(np.mean([g[1] for g in sampled])),
                "interval_score": float(np.mean([g[2] for g in sampled])),
            }

        point = values(prepared)
        boot = {key: [] for key in point}
        for _ in range(bootstrap):
            sampled = [prepared[i] for i in rng.integers(0, len(groups), len(groups))]
            for key, value in values(sampled).items():
                boot[key].append(value)
        row = {"nominal": level / 100.0, "n_exact": len(exact),
               "n_campaigns": exact["campaign_id"].nunique(),
               "n_sources": exact["source_id"].nunique(),
               "interval_completeness": float(np.mean(np.isfinite(exact[low]) & np.isfinite(exact[high]))),
               "calibration": "development grouped-OOF absolute residuals", **point}
        for key, values_ in boot.items():
            row[f"{key}_lo"] = float(np.quantile(values_, 0.025))
            row[f"{key}_hi"] = float(np.quantile(values_, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def runout_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    audited_models = ([s.key for s in PHYSICAL_SPECS]
                      + [PROPOSED_MODEL, "ET-Walker", "ExtraTrees"]
                      + list(CENSORED_MODELS) + ["ML-Strong"] + list(SIMPLE_CONTROL_MODELS))
    runout = predictions.loc[
        predictions["is_runout"] & predictions["model"].isin(audited_models)
    ].copy()
    for model, group in runout.groupby("model"):
        weight = analysis_weights(group)
        finite_sigma = np.isfinite(group["sigma"].to_numpy(float))
        nll_value = np.nan
        if finite_sigma.all():
            if model == "Weibull-AFT":
                z = ((group["logN"].to_numpy(float) - group["pred_logN"].to_numpy(float))
                     / group["sigma"].to_numpy(float))
                loss = np.exp(np.clip(z, -30.0, 30.0))
            else:
                z = ((group["pred_logN"].to_numpy(float) - group["logN"].to_numpy(float))
                     / group["sigma"].to_numpy(float))
                loss = -scipy_special.log_ndtr(z)
            nll_value = float(np.sum(weight * loss) / np.sum(weight))
        below = (group["pred_logN"] < group["logN"]).astype(float)
        group_rates = below.groupby(group[PRIMARY_GROUP_COLUMN]).mean()
        rows.append({"model": model, "n_runout": len(group),
                     "n_campaigns": group["campaign_id"].nunique(),
                     "n_sources": group["source_id"].nunique(),
                     "SB_runout_NLL": nll_value,
                     "SB_point_below_runout_limit_rate": float(group_rates.mean()),
                     "runouts_in_location_training": bool(group["used_for_training"].all()),
                     "runouts_in_candidate_selection": bool(
                         group["used_for_candidate_selection"].all()
                     ),
                     "used_as_censor_feasibility_reference": bool(
                         group["used_as_censor_feasibility_reference"].all()
                     ),
                     "runouts_in_model_selection_or_reference": bool(
                         group["used_for_selection"].all()
                     ),
                     "interpretation": "outer-test right-censored audit; stopping cycle is a lower bound, not a point failure life"})
    return pd.DataFrame(rows).sort_values("SB_runout_NLL", na_position="last")


def source_gain_table(predictions: pd.DataFrame) -> pd.DataFrame:
    exact = predictions.loc[
        predictions["is_exact"] & predictions["model"].isin(["ML-Ens", "WB-PIML"])
    ]
    wide = exact.pivot(
        index=["campaign_id", "source_id", "record_equivalence_id", "row_id", "logN"],
        columns="model", values="pred_logN"
    ).reset_index()
    rows = []
    for source, group in wide.groupby("source_id"):
        equivalence_size = group.groupby("record_equivalence_id")[
            "record_equivalence_id"
        ].transform("size").to_numpy(float)
        weight = 1.0 / (
            max(group["record_equivalence_id"].nunique(), 1) * equivalence_size
        )
        ml_error = (group["ML-Ens"] - group["logN"]).to_numpy(float)
        piml_error = (group["WB-PIML"] - group["logN"]).to_numpy(float)
        rmse_ml = float(np.sqrt(np.sum(weight * ml_error**2) / np.sum(weight)))
        rmse_piml = float(np.sqrt(np.sum(weight * piml_error**2) / np.sum(weight)))
        rows.append({"source_id": source, "n_exact": len(group), "RMSE_ML_Ens": rmse_ml,
                     "RMSE_WB_PIML": rmse_piml, "RMSE_gain": rmse_ml - rmse_piml})
    table = pd.DataFrame(rows).sort_values("RMSE_gain", ascending=False).reset_index(drop=True)
    label_map = {
        source: f"S{index:02d}"
        for index, source in enumerate(sorted(table["source_id"].astype(str).unique()), 1)
    }
    table.insert(1, "plot_label", table["source_id"].astype(str).map(label_map))
    return table


def ablation_effects(predictions: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    models = ["WB", "WB-CD", "WB-M", "WB-Ox", "ExtraTrees", "ET-Walker",
              "WB-PIML", "WB-PIML-M", "WB-PIML-Ox", "ML-Ens"]
    exact = predictions.loc[predictions["is_exact"] & predictions["model"].isin(models)]
    wide = exact.pivot(
        index=["campaign_id", "source_id", "record_equivalence_id", "row_id", "logN"],
        columns="model", values="pred_logN"
    ).reset_index()
    groups = [g.reset_index(drop=True) for _, g in wide.groupby(PRIMARY_GROUP_COLUMN)]
    prepared = {
        model: _prepare_sb_groups([
            g[["record_equivalence_id", "logN", model]].rename(columns={model: "pred_logN"})
            for g in groups
        ])
        for model in models
    }

    def rmse(indices, model):
        return _sb_rmse_prepared([prepared[model][i] for i in indices])

    rng = np.random.default_rng(seed)
    rows = []
    for model in models:
        if model == PROPOSED_MODEL:
            continue
        point = rmse(range(len(groups)), model) - rmse(range(len(groups)), PROPOSED_MODEL)
        draws = []
        for _ in range(bootstrap):
            indices = rng.integers(0, len(groups), len(groups))
            draws.append(rmse(indices, model) - rmse(indices, PROPOSED_MODEL))
        rows.append({"comparison": f"{PROPOSED_MODEL} gain over {model}", "RMSE_gain": point,
                     "CI95_lo": float(np.quantile(draws, 0.025)),
                     "CI95_hi": float(np.quantile(draws, 0.975)),
                     "positive_favors_WB_PIML": True})
    return pd.DataFrame(rows)

