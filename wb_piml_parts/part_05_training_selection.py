def configure_audit_context(path: Path) -> None:

    global AUDIT_CONTEXT, ACTIVE_DATA_PATH
    ACTIVE_DATA_PATH = Path(path).resolve()
    workbook = pd.ExcelFile(path)
    raw = pd.read_excel(path, sheet_name="Plot_Data") if "Plot_Data" in workbook.sheet_names else pd.DataFrame()
    original = (
        pd.read_excel(path, sheet_name="Plot_Data_Original")
        if "Plot_Data_Original" in workbook.sheet_names else pd.DataFrame()
    )
    cluster_stats: dict[str, dict[str, object]] = {}
    if not raw.empty and "record_equivalence_id" in raw:
        raw = raw.copy()
        raw["record_equivalence_id"] = raw["record_equivalence_id"].fillna("").astype(str).str.strip()
        raw_runout = parse_bool(raw["is_runout"], "Plot_Data.is_runout")
        raw["_audit_runout"] = raw_runout
        for equivalence_id, group in raw.groupby("record_equivalence_id", sort=False):
            if not equivalence_id:
                continue
            cluster_stats[equivalence_id] = {
                "duplicate_cluster_size": int(len(group)),
                "duplicate_source_count": int(group["source_id"].astype(str).nunique()),
                "cross_source_duplicate_candidate": bool(group["source_id"].astype(str).nunique() > 1),
                "supplied_label_conflict_flag": bool(group["_audit_runout"].nunique() > 1),
            }
    AUDIT_CONTEXT = {
        "path": str(ACTIVE_DATA_PATH),
        "raw": raw,
        "original": original,
        "cluster_stats": cluster_stats,
    }


def add_campaign_metadata(df: pd.DataFrame) -> pd.DataFrame:

    result = df.copy()
    required = {
        "record_equivalence_id", "campaign_id", "record_role", "audit_status",
        "evidence_grade", "duplicate_group_status",
    }
    missing = sorted(required - set(result.columns))
    if missing:
        raise KeyError(f"Plot_Data_Verified is missing audit fields: {missing}")
    result["record_equivalence_id"] = (
        result["record_equivalence_id"].fillna("").astype(str).str.strip()
    )
    if result["record_equivalence_id"].eq("").any():
        bad = result.loc[result["record_equivalence_id"].eq(""), "row_id"].astype(int).tolist()
        raise ValueError(f"Blank record_equivalence_id at row_id: {bad[:20]}")
    duplicated = result["record_equivalence_id"].duplicated(keep=False)
    if duplicated.any():
        bad = result.loc[duplicated, ["row_id", "record_equivalence_id"]]
        raise ValueError(
            "Plot_Data_Verified is not physically deduplicated: "
            f"{bad.to_dict('records')[:20]}"
        )

    stats = AUDIT_CONTEXT.get("cluster_stats", {})
    default_stats = {
        "duplicate_cluster_size": 1,
        "duplicate_source_count": 1,
        "cross_source_duplicate_candidate": False,
        "supplied_label_conflict_flag": False,
    }
    for key in default_stats:
        result[key] = [
            stats.get(equivalence_id, default_stats).get(key, default_stats[key])
            for equivalence_id in result["record_equivalence_id"]
        ]
    result["duplicate_cluster_size"] = result["duplicate_cluster_size"].astype(int)
    result["duplicate_source_count"] = result["duplicate_source_count"].astype(int)
    result["cross_source_duplicate_candidate"] = result[
        "cross_source_duplicate_candidate"
    ].astype(bool)
    result["supplied_label_conflict_flag"] = result[
        "supplied_label_conflict_flag"
    ].astype(bool)
    result["label_conflict_flag"] = False
    result["label_conflict_resolved"] = result["supplied_label_conflict_flag"]

    result["campaign_id"] = result["campaign_id"].fillna("").astype(str).str.strip()
    if result["campaign_id"].eq("").any():
        bad = result.loc[result["campaign_id"].eq(""), "row_id"].astype(int).tolist()
        raise ValueError(f"Verified campaign_id contains blanks at row_id: {bad[:20]}")
    result["record_role"] = result["record_role"].fillna("").astype(str).str.strip()
    result["audit_status"] = result["audit_status"].fillna("").astype(str).str.strip()
    result["evidence_grade"] = result["evidence_grade"].fillna("").astype(str).str.strip()
    result["duplicate_group_status"] = (
        result["duplicate_group_status"].fillna("").astype(str).str.strip()
    )
    result["campaign_representative_source"] = (
        result.groupby("campaign_id")["source_id"].transform("first").astype(str)
    )
    result["campaign_mapping_basis"] = (
        "campaign_id and audit_status read from Plot_Data_Verified"
    )
    result["campaign_mapping_status"] = np.where(
        result["audit_status"].ne(""),
        "workbook_" + result["audit_status"],
        "not_audited",
    )
    result["source_campaign_role"] = result["record_role"]
    result["campaign_source_count"] = (
        result.groupby("campaign_id")["source_id"].transform("nunique").astype(int)
    )
    role = result["record_role"].str.lower()
    result["reviewed_secondary_source"] = role.str.contains(
        "secondary_replot", regex=False, na=False
    )
    result["suspected_secondary_source"] = pd.Series(False, index=result.index, dtype=bool)
    result["audit_record_complete"] = (
        result[["record_role", "audit_status", "evidence_grade", "duplicate_group_status"]]
        .ne("").all(axis=1)
    )
    return result


def empirical_interval_halfwidths(oof: pd.DataFrame) -> tuple[float, float]:

    exact = oof.loc[oof["is_exact"]].reset_index(drop=True)
    score = np.abs(exact["logN"].to_numpy(float) - exact["mu"].to_numpy(float))
    campaign_size = exact.groupby(PRIMARY_GROUP_COLUMN)[PRIMARY_GROUP_COLUMN].transform("size").to_numpy(float)
    weight = 1.0 / campaign_size
    weight /= np.mean(weight)
    return weighted_quantile(score, 0.80, weight), weighted_quantile(score, 0.90, weight)


def exact_temperature_residual_gate(train: pd.DataFrame,
                                    valid: pd.DataFrame) -> np.ndarray:

    exact_temperature = np.unique(
        train.loc[train["is_exact"], "T_C"].to_numpy(float)
    )
    if not len(exact_temperature):
        return np.zeros(len(valid), dtype=float)
    valid_temperature = valid["T_C"].to_numpy(float)
    return np.any(
        np.isclose(
            valid_temperature[:, None], exact_temperature[None, :],
            rtol=0.0, atol=RESIDUAL_TEMPERATURE_MATCH_ATOL_C,
        ),
        axis=1,
    ).astype(float)


def summarize_temperature_gate(
    valid: pd.DataFrame,
    gate: np.ndarray,
) -> dict[str, float | int]:

    gate_array = np.asarray(gate, dtype=bool)
    if len(gate_array) != len(valid):
        raise ValueError("Temperature gate length does not match validation rows")
    if "is_exact" not in valid.columns:
        raise KeyError("Temperature-gate summary requires an is_exact column")

    exact_mask = valid["is_exact"].to_numpy(bool)
    exact_gate = gate_array[exact_mask]
    n_exact_unsupported = int((~exact_gate).sum())
    n_all_unsupported = int((~gate_array).sum())
    return {
        "exact_temperature_support_rate": (
            float(exact_gate.mean()) if len(exact_gate) else np.nan
        ),
        "n_exact_temperature_unsupported": n_exact_unsupported,
        "n_residual_temperature_unsupported": n_exact_unsupported,
        "all_test_temperature_support_rate": (
            float(gate_array.mean()) if len(gate_array) else np.nan
        ),
        "n_all_test_temperature_unsupported": n_all_unsupported,
    }


def _runout_fold_losses(oof: pd.DataFrame, sigma: float) -> pd.Series:
    runout = oof.loc[oof["is_runout"]].copy()
    if runout.empty:
        return pd.Series(dtype=float)
    z = (runout["mu"].to_numpy(float) - runout["logN"].to_numpy(float)) / float(sigma)
    runout["loss"] = -scipy_special.log_ndtr(z)
    return runout.groupby("inner_fold")["loss"].mean()


def residual_policy_weight(train, valid, residual_mode, temperature_policy):
    if temperature_policy == "hard":
        return exact_temperature_residual_gate(train, valid)
    if temperature_policy not in ("soft", "soft250"):
        raise ValueError(f"Unknown temperature policy: {temperature_policy}")
    temperatures = train.loc[train["is_exact"], "T_C"].to_numpy(float)
    if not len(temperatures):
        return np.zeros(len(valid), dtype=float)
    distance = np.min(np.abs(valid["T_C"].to_numpy(float)[:, None] - temperatures[None, :]), axis=1)
    return np.exp(-np.square(distance / 250.0))


def fit_hybrid_policy(
    train, valid, spec, seed, trees, gamma=GAMMA, min_samples_leaf=3,
    max_features=.85, eta=.75, residual_mode="ET_residual", ridge_alpha=1.0,
    temperature_scale=250.0, temperature_policy="soft250", strength=0.0,
    family=None, competing_config=DEFAULT_COMPETING_CONFIG,
    exact_temperature_gate_enabled=True, kernel_length_scale=2.0, **kwargs,
):

    require_exact_training(train, f"fit_hybrid_policy[{spec.key}]")
    family = family or residual_mode
    family = {"extra_trees": "ET_residual"}.get(family, family)
    if family not in HYBRID_FAMILIES:
        raise ValueError(f"Unknown residual family: {family}")
    if not np.isfinite(strength) or strength < 0 or not 0 <= eta <= 1:
        raise ValueError("Invalid residual gain or environmental regularization")
    if not np.isfinite(ridge_alpha) or ridge_alpha <= 0 or not np.isfinite(kernel_length_scale) or kernel_length_scale <= 0:
        raise ValueError("Matérn length scale and ridge alpha must be positive")
    applied_strength = float(strength) if spec.mode == "competing_damage" else 0.0
    config = replace(competing_config, environment_strength=applied_strength)
    coef = fit_physical(train, spec, gamma, config)
    exact = train.loc[train.is_exact].reset_index(drop=True)
    full = pd.concat([exact, valid], ignore_index=True)
    base = predict_physical(full, spec, coef, gamma)
    target = exact.logN.to_numpy(float) - base[:len(exact)]
    augmented, extra = residual_design_frame(full, coef, gamma, family)
    prep = make_residual_preprocessor(extra)
    x = prep.fit_transform(augmented.iloc[:len(exact)])
    xv = prep.transform(augmented.iloc[len(exact):])
    importance_sum = np.nan
    if eta == 0.0:
        correction = np.zeros(len(valid), float)
    elif family == "Matern_residual":
        correction = kernel_predict(x, xv, target, float(kernel_length_scale), float(ridge_alpha), False)
    else:
        estimator = ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=min_samples_leaf,
            max_features=max_features, random_state=seed, n_jobs=TREE_JOBS)
        correction = estimator.fit(x, target).predict(xv)
        importance_sum = float(estimator.feature_importances_.sum())
    support = (residual_policy_weight(train, valid, family, temperature_policy)
               if exact_temperature_gate_enabled else np.ones(len(valid), float))
    correction = float(eta) * support * correction
    metadata = {
        **(water_fit_metadata(train,coef) if spec.mode=="competing_damage" else {}),
        "eta": float(eta), "family": family, "residual_mode": family,
        "residual_input_policy": RESIDUAL_INPUT_POLICY,
        "residual_environment_shared": True,
        "ridge_alpha": float(ridge_alpha), "kernel_length_scale": float(kernel_length_scale),
        "kernel_parameters_applicable": family == "Matern_residual",
        "strength": applied_strength, "requested_strength": float(strength),
        "environment_regularization_applicable": spec.mode == "competing_damage",
        "temperature_policy": temperature_policy,
        "exact_temperature_gate_enabled": bool(exact_temperature_gate_enabled and temperature_policy == "hard"),
        "temperature_support_enabled": bool(exact_temperature_gate_enabled),
        "active_runout_count": float(train.is_runout.sum()),
        "active_runout_rate": float(train.is_runout.mean()),
        "residual_correction_rms": float(np.sqrt(np.mean(correction ** 2))),
        "residual_feature_importance_sum": importance_sum,
        "mean_residual_policy_weight": float(np.mean(support)),
        **summarize_temperature_gate(valid, exact_temperature_residual_gate(train, valid)),
    }
    if kwargs.get("return_frozen_state", False):
        state = dict(water_fit=water_fit_metadata(train,coef),coefficients=np.asarray(coef,float), gamma=float(gamma), family=family,
            eta=float(eta), preprocessor=prep, temperature_policy=temperature_policy,
            residual_input_policy=RESIDUAL_INPUT_POLICY,
            training_temperatures=np.unique(exact.T_C.to_numpy(float)),
            temperature_support_enabled=bool(exact_temperature_gate_enabled),
            kernel_length_scale=float(kernel_length_scale), ridge_alpha=float(ridge_alpha))
        if eta != 0.0 and family == "Matern_residual":
            kernel = Matern(length_scale=float(kernel_length_scale), nu=1.5)
            matrix = kernel(x); matrix.flat[::len(x)+1] += float(ridge_alpha)
            state['kernel_x'] = x
            state['kernel_weights'] = cho_solve(cho_factor(matrix,lower=True,check_finite=False),target,check_finite=False)
        elif eta != 0.0:
            state['estimator'] = estimator
        metadata['_frozen_state'] = state
    return base[len(exact):] + correction, coef, metadata


def fit_selected_hybrid(train, valid, candidate, seed, trees, spec=COMPETING_WB_SPEC):
    return fit_hybrid_policy(train, valid, spec, seed, trees,
        gamma=float(candidate["gamma"]), min_samples_leaf=int(candidate.get("min_samples_leaf", 3)),
        max_features=float(candidate.get("max_features", .85)), eta=float(candidate["eta"]),
        family=str(candidate["family"]), strength=float(candidate["strength"]),
        ridge_alpha=float(candidate.get("ridge_alpha", 1.0)),
        kernel_length_scale=float(candidate.get("kernel_length_scale", 2.0)),
        temperature_policy="soft250")


def select_budgeted_candidate(candidate_table, rmse_tolerance=NEAR_TIE_RMSE):
    candidates = candidate_table.copy()
    required = {"candidate_index", "inner_SB_RMSE", "inner_runout_survival_NLL",
                "censor_feasible", "is_reference", "basis_id", "min_samples_leaf"}
    if not required.issubset(candidates.columns):
        raise ValueError(f"Missing candidate columns: {sorted(required - set(candidates.columns))}")
    if candidates.empty or candidates["candidate_index"].duplicated().any():
        raise ValueError("Candidate indices must be unique and nonempty")
    finite_rmse = np.isfinite(candidates["inner_SB_RMSE"].to_numpy(float))
    if not finite_rmse.any():
        raise ValueError("No finite inner RMSE candidate is available")
    best_rmse = float(candidates.loc[finite_rmse, "inner_SB_RMSE"].min())
    candidates["within_RMSE_tolerance"] = (
        finite_rmse & candidates["inner_SB_RMSE"].le(best_rmse + float(rmse_tolerance))
    )
    pool = candidates.loc[candidates["within_RMSE_tolerance"]].sort_values([
        "inner_runout_survival_NLL", "inner_SB_RMSE", "candidate_index"
    ], na_position="last")
    best = pool.iloc[0]
    candidates["inner_LOTO_RMSE"] = np.nan
    candidates["inner_LOTO_seed"] = np.nan
    chosen = best.to_dict()
    chosen["inner_LOTO_RMSE"] = np.nan
    chosen["inner_LOTO_seed"] = np.nan
    for key in ("candidate_index", "min_samples_leaf", "basis_id"):
        chosen[key] = int(chosen[key])
    for key in ("censor_feasible", "is_reference", "within_RMSE_tolerance"):
        chosen[key] = bool(chosen[key])
    chosen["censor_constraint_relaxed"] = not chosen["censor_feasible"]
    chosen["selected"] = True
    chosen["RMSE_tolerance"] = float(rmse_tolerance)
    chosen["global_best_inner_RMSE"] = best_rmse
    chosen["censor_guard_used_for_selection"] = False
    chosen["selection_rule"] = "global_RMSE_budget_then_min_runout_NLL"
    for key in ("censor_constraint_relaxed", "selection_rule", "censor_guard_used_for_selection",
                "global_best_inner_RMSE", "RMSE_tolerance"):
        candidates[key] = chosen[key]
    candidates["selected"] = candidates["candidate_index"].eq(chosen["candidate_index"])
    candidates = candidates.sort_values([
        "within_RMSE_tolerance", "inner_runout_survival_NLL", "inner_SB_RMSE", "candidate_index"
    ], ascending=[False, True, True, True], na_position="last").reset_index(drop=True)
    return chosen, candidates


def nested_select_hybrid(train, seed, trees):
    return nested_select_structural(train, seed, trees)[:3]


@dataclass
class GenericAFTModel:
    family: str
    preprocessor: ColumnTransformer
    coefficients: np.ndarray
    sigma: float
    alpha: float


def _generic_aft_row_losses(family: str, y: np.ndarray, mu: np.ndarray,
                            sigma: float, exact_mask: np.ndarray) -> np.ndarray:
    z = (np.asarray(y, float) - np.asarray(mu, float)) / float(sigma)
    losses = np.empty(len(z), float)
    if family == "lognormal":
        losses[exact_mask] = (
            0.5 * math.log(2.0 * math.pi) + math.log(sigma)
            + 0.5 * z[exact_mask] ** 2
        )
        losses[~exact_mask] = -scipy_special.log_ndtr(-z[~exact_mask])
    elif family == "weibull":
        z_clip = np.clip(z, -30.0, 30.0)
        exp_z = np.exp(z_clip)
        losses[exact_mask] = math.log(sigma) - z_clip[exact_mask] + exp_z[exact_mask]
        losses[~exact_mask] = exp_z[~exact_mask]
    else:
        raise ValueError(f"Unknown AFT family: {family}")
    return losses


def fit_generic_aft(train: pd.DataFrame, family: str, alpha: float) -> GenericAFTModel:

    require_exact_training(train, f"fit_generic_aft[{family}]")
    preprocessor = clone(make_preprocessor())
    transformed = np.asarray(preprocessor.fit_transform(train), float)
    design = np.column_stack([np.ones(len(train)), transformed])
    exact_mask = train["is_exact"].to_numpy(bool)
    y = train["logN"].to_numpy(float)
    exact_x, exact_y = design[exact_mask], y[exact_mask]
    ridge = np.eye(design.shape[1]) * float(alpha)
    ridge[0, 0] = 0.0
    start_beta = np.linalg.solve(
        exact_x.T @ exact_x + ridge + 1e-8 * np.eye(design.shape[1]),
        exact_x.T @ exact_y,
    )
    residual = exact_y - exact_x @ start_beta
    start_sigma = float(np.clip(np.sqrt(np.mean(residual ** 2)), 0.15, 2.5))
    start = np.r_[start_beta, math.log(start_sigma)]

    def objective(parameters: np.ndarray) -> float:
        beta = parameters[:-1]
        sigma = float(np.exp(parameters[-1]))
        mu = design @ beta
        losses = _generic_aft_row_losses(family, y, mu, sigma, exact_mask)
        penalty = 0.5 * float(alpha) * float(np.mean(beta[1:] ** 2))
        return float(np.mean(losses) + penalty)

    bounds = [(-20.0, 20.0)] * design.shape[1] + [(math.log(0.08), math.log(4.0))]
    result = minimize(
        objective, start, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 250, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"{family} AFT fit failed: {result.message}")
    return GenericAFTModel(
        family=family, preprocessor=preprocessor,
        coefficients=np.asarray(result.x[:-1], float),
        sigma=float(np.exp(result.x[-1])), alpha=float(alpha),
    )


def predict_generic_aft(model: GenericAFTModel, frame: pd.DataFrame) -> np.ndarray:
    transformed = np.asarray(model.preprocessor.transform(frame), float)
    design = np.column_stack([np.ones(len(frame)), transformed])
    return np.asarray(design @ model.coefficients, float)


def generic_aft_summary(frame: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray | float,
                        family: str) -> dict[str, float]:
    exact_mask = frame["is_exact"].to_numpy(bool)
    sigma_array = np.broadcast_to(np.asarray(sigma, float), len(frame))
    z_losses = np.empty(len(frame), float)
    for value in np.unique(sigma_array):
        selected = np.isclose(sigma_array, value)
        z_losses[selected] = _generic_aft_row_losses(
            family, frame.loc[selected, "logN"].to_numpy(float),
            np.asarray(mu, float)[selected], float(value), exact_mask[selected],
        )
    return {
        "joint_censored_NLL": float(np.mean(z_losses)),
        "exact_density_NLL": float(np.mean(z_losses[exact_mask])),
        "runout_survival_NLL": float(np.mean(z_losses[~exact_mask])) if (~exact_mask).any() else np.nan,
    }


def nested_select_generic_aft(train: pd.DataFrame, family: str, seed: int
                              ) -> tuple[dict[str, object], pd.DataFrame]:

    splits, _ = grouped_folds(train, min(4, train[PRIMARY_GROUP_COLUMN].nunique()), seed)
    rows = []
    for candidate_index, alpha in enumerate((0.01, 0.10, 1.0, 10.0)):
        pieces = []
        for inner_fold, (fit_idx, valid_idx) in enumerate(splits, 1):
            fit = train.iloc[fit_idx].reset_index(drop=True)
            valid = train.iloc[valid_idx].reset_index(drop=True)
            model = fit_generic_aft(fit, family, alpha)
            part = valid[["row_id", "source_id", "campaign_id", "record_equivalence_id",
                          "logN", "is_exact", "is_runout"]].copy()
            part["inner_fold"] = inner_fold
            part["mu"] = predict_generic_aft(model, valid)
            part["sigma"] = model.sigma
            pieces.append(part)
        oof = pd.concat(pieces, ignore_index=True)
        exact = oof.loc[oof["is_exact"]].copy()
        exact["pred_logN"] = exact["mu"]
        point = sb_metrics([
            group.reset_index(drop=True)
            for _, group in exact.groupby(PRIMARY_GROUP_COLUMN)
        ])["SB_RMSE"]
        summary = generic_aft_summary(
            oof, oof["mu"].to_numpy(float), oof["sigma"].to_numpy(float), family
        )
        rows.append({
            "candidate_index": candidate_index, "family": family, "alpha": alpha,
            "inner_SB_RMSE": point, **summary,
        })
    candidates = pd.DataFrame(rows).sort_values(
        ["joint_censored_NLL", "inner_SB_RMSE", "alpha"]
    ).reset_index(drop=True)
    best = candidates.iloc[0]
    return {
        "candidate_index": int(best["candidate_index"]),
        "family": family, "alpha": float(best["alpha"]),
        "inner_SB_RMSE": float(best["inner_SB_RMSE"]),
        "inner_joint_censored_NLL": float(best["joint_censored_NLL"]),
        "inner_runout_survival_NLL": float(best["runout_survival_NLL"]),
    }, candidates


def fit_censored_comparators_outer(train: pd.DataFrame, test: pd.DataFrame,
                                   seed: int) -> tuple[dict[str, dict[str, object]],
                                                       list[dict[str, object]],
                                                       list[pd.DataFrame]]:
    predictions: dict[str, dict[str, object]] = {}
    selections: list[dict[str, object]] = []
    candidate_tables: list[pd.DataFrame] = []
    for offset, (name, family) in enumerate(zip(CENSORED_MODELS, ("lognormal", "weibull"))):
        choice, candidates = nested_select_generic_aft(train, family, seed + 1000 * offset)
        fitted = fit_generic_aft(train, family, float(choice["alpha"]))
        predictions[name] = {
            "mu": predict_generic_aft(fitted, test),
            "sigma": fitted.sigma,
            "family": family,
            "alpha": fitted.alpha,
        }
        selections.append({"model": name, **choice})
        table = candidates.copy()
        table.insert(0, "model", name)
        candidate_tables.append(table)
    return predictions, selections, candidate_tables

