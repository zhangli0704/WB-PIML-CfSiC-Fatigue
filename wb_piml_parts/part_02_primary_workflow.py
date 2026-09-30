def configure_audit_context(path: Path) -> None:
    """Load workbook provenance without changing any modelling value."""
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
    """Preserve reviewed workbook provenance; never manufacture pass flags."""
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
    """Campaign-balanced development-OOF interval calibration."""
    exact = oof.loc[oof["is_exact"]].reset_index(drop=True)
    score = np.abs(exact["logN"].to_numpy(float) - exact["mu"].to_numpy(float))
    campaign_size = exact.groupby(PRIMARY_GROUP_COLUMN)[PRIMARY_GROUP_COLUMN].transform("size").to_numpy(float)
    weight = 1.0 / campaign_size
    weight /= np.mean(weight)
    return weighted_quantile(score, 0.80, weight), weighted_quantile(score, 0.90, weight)






def exact_temperature_residual_gate(train: pd.DataFrame,
                                    valid: pd.DataFrame) -> np.ndarray:
    """Response-free residual gate based only on development exact-T support."""
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
    """Report temperature support separately for exact and all test rows."""
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
    """Fit censored physics and an exact-failure residual using training data only."""
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
    augmented = full.copy()
    extra = ()
    if family == "physics_feature_ET_residual":
        mechanical = coef[0] - coef[1] * walker_log_stress(full, gamma)
        augmented["mechanical_prediction"] = mechanical
        augmented["environment_decrement"] = mechanical - base
        augmented["walker_stress"] = walker_log_stress(full, gamma)
        extra = ("mechanical_prediction", "environment_decrement", "walker_stress")
    prep = make_preprocessor(extra)
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
    """Penalized observable-covariate AFT using exact density + runout survival."""
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
    """Use the same campaign-disjoint development folds as WB-PIML."""
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


def claim_gate(metrics: pd.DataFrame, effects: pd.DataFrame, parameters: pd.DataFrame,
               df: pd.DataFrame, nested_selection: pd.DataFrame,
               nested_candidates: pd.DataFrame, runout: pd.DataFrame,
               external_summary: pd.DataFrame,
               external_groups: pd.DataFrame,
               external_effects: pd.DataFrame,
               external_uq: pd.DataFrame,
               external_selection: pd.DataFrame,
               temperature_gate_rows: pd.DataFrame) -> pd.DataFrame:
    """Evidence-derived gates; unknown audit fields remain unknown, never True."""
    metric = metrics.set_index("model")

    def effect_row(comparator: str) -> pd.Series:
        selected = effects.loc[
            effects["comparator"].eq(comparator) & effects["effect"].eq("RMSE_gain")
        ]
        if selected.empty:
            return pd.Series({"estimate": np.nan, "CI95_lo": np.nan, "CI95_hi": np.nan})
        return selected.iloc[0]

    paired = {name: effect_row(name) for name in PRIMARY_COMPARATORS}
    proposed = metric.loc[PROPOSED_MODEL]
    anchor = metric.loc["WB-PIML-Anchor"]
    external_models = [
        "WB", "RF", "ExtraTrees", "ET-Walker", "GBDT", "SVR", "ML-Ens", "ML-Strong",
        *CENSORED_MODELS, PROPOSED_MODEL,
    ]
    best_primary = str(metrics.loc[metrics["model"].isin(external_models)]
                       .sort_values("SB_RMSE").iloc[0]["model"])
    proposed_parameters = parameters.loc[parameters["model"].eq(PROPOSED_MODEL)]
    physics_mix = pd.to_numeric(proposed_parameters["physics_mix"], errors="coerce")
    n_rows = len(df)
    n_exact = int(df["is_exact"].sum())
    n_runout = int(df["is_runout"].sum())
    n_overrides = int(df["label_override_applied"].sum())
    audit_complete = bool(df.get("audit_record_complete", pd.Series(False, index=df.index)).all())
    supplied_conflicts = int(df.loc[
        df["supplied_label_conflict_flag"].fillna(False), "record_equivalence_id"
    ].nunique())
    reviewed_conflicts = int(df.loc[
        df["label_conflict_flag"].fillna(False), "record_equivalence_id"
    ].nunique())
    resolved_conflicts = int(df.loc[
        df["label_conflict_resolved"].fillna(False), "record_equivalence_id"
    ].nunique())
    secondary_rows = int(df["reviewed_secondary_source"].fillna(False).sum())
    duplicate_clusters = int(df.loc[
        df["cross_source_duplicate_candidate"].fillna(False), "record_equivalence_id"
    ].nunique())
    campaign_disjoint = bool(df.groupby("campaign_id")["outer_fold"].nunique().le(1).all())
    reviewed_map = bool(
        audit_complete
        and df["campaign_mapping_status"].astype(str).str.startswith("workbook_").all()
        and df["campaign_id"].astype(str).str.strip().ne("").all()
    )
    temperature_gate_verified = bool(
        len(temperature_gate_rows) == len(df)
        and not temperature_gate_rows["row_id"].duplicated().any()
    )
    if temperature_gate_verified:
        for fold, held_out in temperature_gate_rows.groupby("outer_fold"):
            development = df.loc[df["outer_fold"].ne(int(fold))]
            expected_gate = exact_temperature_residual_gate(
                development, held_out
            ).astype(bool)
            actual_gate = held_out["residual_temperature_supported"].astype(bool).to_numpy()
            if not np.array_equal(expected_gate, actual_gate):
                temperature_gate_verified = False
                break

    # Reconcile duplicate metadata with source-record multiplicities.
    raw = AUDIT_CONTEXT.get("raw")
    duplicate_audit_complete = False
    duplicate_audit_detail = "raw Plot_Data audit table unavailable"
    if (
        isinstance(raw, pd.DataFrame) and not raw.empty
        and {"record_equivalence_id", "source_id"}.issubset(raw.columns)
    ):
        raw_ids = raw["record_equivalence_id"].fillna("").astype(str).str.strip()
        raw_audit = pd.DataFrame({
            "record_equivalence_id": raw_ids,
            "source_id": raw["source_id"].fillna("").astype(str).str.strip(),
        })
        raw_audit = raw_audit.loc[raw_audit["record_equivalence_id"].ne("")]
        raw_counts = raw_audit.groupby("record_equivalence_id").size()
        raw_source_counts = raw_audit.groupby("record_equivalence_id")["source_id"].nunique()
        verified_ids = df["record_equivalence_id"].astype(str)
        expected_cluster_size = verified_ids.map(raw_counts)
        expected_source_count = verified_ids.map(raw_source_counts)
        duplicate_audit_complete = bool(
            df["record_equivalence_id"].is_unique
            and expected_cluster_size.notna().all()
            and expected_source_count.notna().all()
            and np.array_equal(
                expected_cluster_size.to_numpy(int),
                df["duplicate_cluster_size"].to_numpy(int),
            )
            and np.array_equal(
                expected_source_count.to_numpy(int),
                df["duplicate_source_count"].to_numpy(int),
            )
            and np.array_equal(
                expected_source_count.gt(1).to_numpy(bool),
                df["cross_source_duplicate_candidate"].to_numpy(bool),
            )
        )
        duplicate_audit_detail = (
            f"verified IDs covered by Plot_Data={int(expected_cluster_size.notna().sum())}/{len(df)}; "
            f"cross-source duplicate clusters={duplicate_clusters}"
        )

    # Check selected candidates against the complete inner-fold search tables.
    selection_trace_complete = False
    selection_trace_detail = "nested selection trace is incomplete"
    required_selection = {"outer_fold", "model", "candidate_index"}
    if (
        required_selection.issubset(nested_selection.columns)
        and required_selection.issubset(nested_candidates.columns)
    ):
        selected = nested_selection.loc[
            nested_selection["model"].eq("WB-PIML-Anchor"),
            ["outer_fold", "candidate_index"],
        ].dropna()
        candidates = nested_candidates.loc[
            nested_candidates["model"].eq("WB-PIML-Anchor"),
            ["outer_fold", "candidate_index"],
        ].dropna()
        selected_pairs = set(map(tuple, selected.astype(int).to_numpy()))
        candidate_pairs = set(map(tuple, candidates.astype(int).to_numpy()))
        candidate_counts = candidates.groupby("outer_fold")["candidate_index"].nunique()
        expected_folds = set(
            pd.to_numeric(proposed_parameters["outer_fold"], errors="coerce")
            .dropna().astype(int)
        )
        forbidden_outer_columns = {
            "pred_logN", "outer_test_logN", "outer_test_RMSE", "test_response"
        }
        selection_trace_complete = bool(
            expected_folds
            and set(selected["outer_fold"].astype(int)) == expected_folds
            and set(candidate_counts.index.astype(int)) == expected_folds
            and candidate_counts.eq(len(HYBRID_CANDIDATES)).all()
            and selected_pairs.issubset(candidate_pairs)
            and forbidden_outer_columns.isdisjoint(nested_candidates.columns)
        )
        selection_trace_detail = (
            f"selected folds={len(selected_pairs)}; candidate counts by fold="
            f"{candidate_counts.astype(int).to_dict()}"
        )

    strong_rows = nested_candidates.loc[nested_candidates["model"].eq("ML-Strong")]
    strong_selected = nested_selection.loc[nested_selection["model"].eq("ML-Strong")]
    strong_trace_complete = bool(
        not strong_rows.empty and len(strong_selected)==len(proposed_parameters)
        and strong_rows.groupby("outer_fold")["candidate_index"].nunique().eq(len(CONTROL_CANDIDATES)).all()
        and all(int(row.candidate_index) in set(strong_rows.loc[strong_rows.outer_fold.eq(row.outer_fold), "candidate_index"])
                for row in strong_selected.itertuples()))
    selection_trace_complete = selection_trace_complete and strong_trace_complete
    selection_trace_detail += f"; strong-control trace complete={strong_trace_complete}"
    censored_parameters = parameters.loc[parameters["model"].isin(CENSORED_MODELS)]
    censored_trace_complete = bool(
        all(name in metric.index for name in CENSORED_MODELS)
        and not censored_parameters.empty
        and censored_parameters.groupby("model")["outer_fold"].nunique()
        .reindex(CENSORED_MODELS).eq(len(proposed_parameters)).all()
        and pd.to_numeric(
            censored_parameters["active_runout_count"], errors="coerce"
        ).gt(0).all()
    )

    physical_columns = [
        "walker_slope", "environment_stress_exponent",
        "arrhenius_temperature_sensitivity", "water_log10_multiplier", "n_physics_parameters",
    ]
    physical_values = proposed_parameters[physical_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    mechanistic_constraints_verified = bool(
        not physical_values.empty
        and np.isfinite(physical_values.to_numpy(float)).all()
        and physical_values["walker_slope"].between(0.05, 20.0).all()
        and physical_values["environment_stress_exponent"].between(0.0, 10.0).all()
        and physical_values["arrhenius_temperature_sensitivity"].between(0.0, 10.0).all()
        and physical_values["n_physics_parameters"].eq(6).all()
        and physical_values["water_log10_multiplier"].ge(0).all()
        and physics_mix.notna().all() and physics_mix.gt(0).all()
    )
    rmse_ci_pass = {
        name: bool(np.isfinite(row["CI95_lo"]) and float(row["CI95_lo"]) > 0)
        for name, row in paired.items()
    }
    primary_rmse_superiority = all(rmse_ci_pass.values())

    required_effects = {"RMSE_gain", "MAE_gain", "R2_gain", "F5_gain", "C_star_gain"}
    primary_effect_rows = effects.loc[
        effects["comparator"].isin(PRIMARY_COMPARATORS)
        & effects["effect"].isin(required_effects)
    ].copy()
    primary_multimetric_superiority = bool(
        len(primary_effect_rows) == len(PRIMARY_COMPARATORS) * len(required_effects)
        and not primary_effect_rows.duplicated(["comparator", "effect"]).any()
        and pd.to_numeric(primary_effect_rows["CI95_lo"], errors="coerce").gt(0).all()
    )

    external_point_passes: dict[str, bool] = {}
    for scenario, scenario_table in external_summary.groupby("scenario"):
        proposed_rows = scenario_table.loc[scenario_table["model"].eq(PROPOSED_MODEL)]
        comparator_rows = scenario_table.loc[
            scenario_table["model"].isin(PRIMARY_COMPARATORS)
        ]
        external_point_passes[str(scenario)] = bool(
            len(proposed_rows) == 1 and len(comparator_rows) == len(PRIMARY_COMPARATORS)
            and float(proposed_rows.iloc[0]["SB_RMSE"])
            < float(comparator_rows["SB_RMSE"].min())
        )
    expected_external_scenarios = {"LOSO", "LOAO", "LOTO", "LOEO"}
    external_point_superiority = bool(
        set(external_point_passes) == expected_external_scenarios
        and all(external_point_passes.values())
    )
    external_rmse_rows = external_effects.loc[
        external_effects["comparator"].isin(PRIMARY_COMPARATORS)
        & external_effects["effect"].eq("RMSE_gain")
    ].copy()
    expected_external_rows = len(expected_external_scenarios) * len(PRIMARY_COMPARATORS)
    external_ci_superiority = bool(
        expected_external_rows > 0
        and len(external_rmse_rows) == expected_external_rows
        and set(external_rmse_rows["scenario"].astype(str)) == expected_external_scenarios
        and not external_rmse_rows.duplicated(["scenario", "comparator", "effect"]).any()
        and pd.to_numeric(external_rmse_rows["CI95_lo"], errors="coerce").gt(0).all()
    )
    expected_external_uq_rows = len(expected_external_scenarios) * 2
    external_interval_complete = bool(
        expected_external_uq_rows > 0
        and len(external_uq) == expected_external_uq_rows
        and set(external_uq["scenario"].astype(str)) == expected_external_scenarios
        and not external_uq.duplicated(["scenario", "nominal"]).any()
        and pd.to_numeric(
            external_uq["interval_completeness"], errors="coerce"
        ).eq(1.0).all()
    )
    external_interval_nominal_compatible = bool(
        external_interval_complete
        and (
            pd.to_numeric(external_uq["coverage_lo"], errors="coerce")
            <= pd.to_numeric(external_uq["nominal"], errors="coerce")
        ).all()
        and (
            pd.to_numeric(external_uq["coverage_hi"], errors="coerce")
            >= pd.to_numeric(external_uq["nominal"], errors="coerce")
        ).all()
    )

    external_nested_selection_complete = False
    external_selection_detail = "external nested-selection trace is incomplete"
    proposed_external_selection = pd.DataFrame()
    external_censor_guard_never_relaxed = False
    external_censor_status_known = False
    external_relaxed_count = 0
    external_wb_holdout_count = 0
    required_external_selection_columns = {
        "scenario", "holdout_group", "fold", "model", "inner_group",
        "n_inner_folds", "n_train_campaigns", "n_train_exact",
        "n_candidates_evaluated",
        "inner_SB_RMSE", "selected_in_holdout_training_only",
        "outer_test_used_for_selection", "validation_scope",
        "campaign_disjoint_by_design", "campaign_overlap",
        "holdout_campaign_disjoint", "censor_constraint_relaxed",
        "censor_feasible",
        "n_overlapping_campaigns", "empirical_halfwidth_80",
        "empirical_halfwidth_90", "n_interval_calibration_exact",
        "interval_calibration", "outer_test_used_for_interval_calibration",
    }
    required_external_group_columns = {"scenario", "holdout_group"}
    if (
        required_external_selection_columns.issubset(external_selection.columns)
        and required_external_group_columns.issubset(external_groups.columns)
    ):
        expected_holdouts = {
            (str(scenario), str(holdout))
            for scenario, holdout in external_groups[
                ["scenario", "holdout_group"]
            ].drop_duplicates().itertuples(index=False, name=None)
        }
        selected_models = {PROPOSED_MODEL, "ExtraTrees", "ET-Walker", "ML-Strong", *SIMPLE_CONTROL_MODELS}
        expected_rows = {
            (scenario, holdout, model)
            for scenario, holdout in expected_holdouts
            for model in selected_models
        }
        selection = external_selection.copy()
        forbidden_external_response_columns = {
            "logN", "pred_logN", "outer_test_logN", "outer_test_RMSE",
            "holdout_RMSE", "test_response",
        }
        actual_rows = {
            (str(scenario), str(holdout), str(model))
            for scenario, holdout, model in selection[
                ["scenario", "holdout_group", "model"]
            ].itertuples(index=False, name=None)
        }
        expected_candidate_counts = {
            PROPOSED_MODEL: len(HYBRID_CANDIDATES),
            "ML-Strong": len(CONTROL_CANDIDATES),
            "ExtraTrees": len({
                (leaf, features)
                for _, leaf, features, _ in NESTED_CANDIDATES
            }),
            "ET-Walker": len(NESTED_CANDIDATES),
            "WB-Residual": 18, "Basquin-Residual": 6,
        }
        observed_candidate_counts = pd.to_numeric(
            selection["n_candidates_evaluated"], errors="coerce"
        )
        required_candidate_counts = selection["model"].map(
            expected_candidate_counts
        )
        inner_folds = pd.to_numeric(selection["n_inner_folds"], errors="coerce")
        train_campaigns = pd.to_numeric(
            selection["n_train_campaigns"], errors="coerce"
        )
        overlap_count = pd.to_numeric(
            selection["n_overlapping_campaigns"], errors="coerce"
        )
        proposed_external_selection = selection.loc[
            selection["model"].eq(PROPOSED_MODEL)
        ].copy()
        interval_halfwidths = proposed_external_selection[
            ["empirical_halfwidth_80", "empirical_halfwidth_90"]
        ].apply(pd.to_numeric, errors="coerce")
        interval_calibration_exact = pd.to_numeric(
            proposed_external_selection["n_interval_calibration_exact"],
            errors="coerce",
        )
        proposed_train_exact = pd.to_numeric(
            proposed_external_selection["n_train_exact"], errors="coerce"
        )
        external_nested_selection_complete = bool(
            expected_holdouts
            and {scenario for scenario, _ in expected_holdouts}
            == expected_external_scenarios
            and actual_rows == expected_rows
            and len(selection) == len(expected_rows)
            and not selection.duplicated(
                ["scenario", "holdout_group", "model"]
            ).any()
            and selection["model"].isin(selected_models).all()
            and selection["inner_group"].eq(PRIMARY_GROUP_COLUMN).all()
            and selection["selected_in_holdout_training_only"].eq(True).all()
            and selection["outer_test_used_for_selection"].eq(False).all()
            and selection["validation_scope"].eq(
                "domain-held-out stress test"
            ).all()
            and selection["campaign_disjoint_by_design"].eq(False).all()
            and selection["holdout_campaign_disjoint"].astype(bool).eq(
                ~selection["campaign_overlap"].astype(bool)
            ).all()
            and forbidden_external_response_columns.isdisjoint(selection.columns)
            and observed_candidate_counts.eq(required_candidate_counts).all()
            and inner_folds.between(2, 4).all()
            and train_campaigns.ge(inner_folds).all()
            and np.isfinite(pd.to_numeric(
                selection["inner_SB_RMSE"], errors="coerce"
            )).all()
            and selection["campaign_overlap"].notna().all()
            and overlap_count.ge(0).all()
            and np.isfinite(interval_halfwidths.to_numpy(float)).all()
            and interval_halfwidths.ge(0).all().all()
            and interval_calibration_exact.eq(proposed_train_exact).all()
            and proposed_external_selection["interval_calibration"].eq(
                "development campaign-disjoint OOF absolute residuals"
            ).all()
            and proposed_external_selection[
                "outer_test_used_for_interval_calibration"
            ].eq(False).all()
        )
        external_selection_detail = (
            f"holdouts={len(expected_holdouts)}; selected rows={len(selection)}; "
            f"candidate counts={expected_candidate_counts}; "
            f"holdouts with campaign overlap="
            f"{selection.loc[selection['campaign_overlap'].eq(True), ['scenario', 'holdout_group']].drop_duplicates().shape[0]}"
        )
        external_relaxed_flags = pd.to_numeric(
            proposed_external_selection["censor_constraint_relaxed"],
            errors="coerce",
        )
        external_feasible_flags = pd.to_numeric(
            proposed_external_selection["censor_feasible"], errors="coerce"
        )
        external_wb_holdout_count = len(proposed_external_selection)
        external_relaxed_count = int(external_relaxed_flags.eq(1).sum())
        external_censor_status_known = bool(
            external_wb_holdout_count == len(expected_holdouts)
            and external_relaxed_flags.notna().all()
            and external_feasible_flags.notna().all()
            and external_relaxed_flags.isin([0, 1]).all()
            and external_feasible_flags.isin([0, 1]).all()
        )
        external_censor_guard_never_relaxed = bool(
            external_nested_selection_complete
            and external_censor_status_known
            and external_relaxed_flags.eq(0).all()
            and external_feasible_flags.eq(1).all()
        )
        external_selection_detail += (
            f"; WB-PIML holdouts failing the runout diagnostic="
            f"{external_relaxed_count}/{external_wb_holdout_count}"
        )

    runout_index = (
        runout.set_index("model")
        if "model" in runout and not runout["model"].duplicated().any()
        else pd.DataFrame()
    )
    if (
        isinstance(runout_index, pd.DataFrame)
        and {PROPOSED_MODEL, "WB-CD"}.issubset(runout_index.index)
    ):
        proposed_runout_nll = float(runout_index.loc[PROPOSED_MODEL, "SB_runout_NLL"])
        reference_runout_nll = float(runout_index.loc["WB-CD", "SB_runout_NLL"])
        runout_noninferior = bool(proposed_runout_nll <= reference_runout_nll)
    else:
        proposed_runout_nll = reference_runout_nll = np.nan
        runout_noninferior = False

    anchor_selection = nested_selection.loc[
        nested_selection["model"].eq("WB-PIML-Anchor")
    ]
    relaxed_series = (
        anchor_selection["censor_constraint_relaxed"]
        if "censor_constraint_relaxed" in anchor_selection
        else pd.Series(dtype=bool)
    )
    relaxation_known = bool(
        len(relaxed_series) == len(anchor_selection)
        and not relaxed_series.isna().any()
    )
    relaxed_count = int(relaxed_series.eq(True).sum())
    censor_guard_never_relaxed = bool(
        len(anchor_selection) == len(proposed_parameters)
        and relaxation_known
        and relaxed_count == 0
    )

    environment_ratio = pd.to_numeric(
        proposed_parameters["environment_log10_damage_ratio"], errors="coerce"
    )
    environment_stress = pd.to_numeric(
        proposed_parameters["environment_stress_exponent"], errors="coerce"
    )
    arrhenius = pd.to_numeric(
        proposed_parameters["arrhenius_temperature_sensitivity"], errors="coerce"
    )
    boundary_hits = (
        np.isclose(environment_ratio, -8.0, atol=1e-5)
        | np.isclose(environment_ratio, 4.0, atol=1e-5)
        | np.isclose(environment_stress, 0.0, atol=1e-5)
        | np.isclose(environment_stress, 10.0, atol=1e-5)
        | np.isclose(arrhenius, 0.0, atol=1e-5)
        | np.isclose(arrhenius, 10.0, atol=1e-5)
        | np.isclose(pd.to_numeric(proposed_parameters["water_log10_multiplier"], errors="coerce"), 0.0, atol=1e-5)
    )
    mechanistic_parameters_interior = bool(
        len(boundary_hits) == len(proposed_parameters) and not np.any(boundary_hits)
    )
    universal_superiority = bool(
        primary_rmse_superiority
        and primary_multimetric_superiority
        and external_point_superiority
        and external_ci_superiority
        and external_nested_selection_complete
        and runout_noninferior
        and censor_guard_never_relaxed
        and external_censor_guard_never_relaxed
        and mechanistic_parameters_interior
        and external_interval_nominal_compatible
        and temperature_gate_verified
    )
    checks: list[tuple[str, object, str]] = [
        ("correct_data", n_rows == EXPECTED_VERIFIED_ROWS and n_exact == EXPECTED_EXACT_ROWS
         and n_runout == EXPECTED_RUNOUT_ROWS and n_overrides == 0,
         f"{n_rows} reviewed analysis records: {n_exact} exact + {n_runout} right-censored; overrides={n_overrides}"),
        ("workbook_audit_complete", audit_complete,
         "record_role, audit_status, evidence_grade and duplicate_group_status are read row-by-row from the workbook"),
        ("human_reviewed_campaign_map", reviewed_map,
         f"mapping statuses are workbook-derived; reviewed secondary-replot rows={secondary_rows}"),
        ("label_consistency", bool(df["label_consistent"].all()),
         "is_exact_failure is the complement of is_runout for every verified row"),
        ("label_conflicts_resolved", reviewed_conflicts == 0,
         f"raw supplied conflicts={supplied_conflicts}; resolved={resolved_conflicts}; unresolved={reviewed_conflicts}"),
        ("physical_duplicate_audit", duplicate_audit_complete,
         duplicate_audit_detail),
        ("campaign_disjoint_validation", campaign_disjoint,
         f"{df['campaign_id'].nunique()} reviewed campaigns never cross an outer fold"),
        ("equal_training_weights", bool(np.allclose(df["sample_weight"], 1.0)),
         f"all {n_rows} analysis likelihood records have unit weight"),
        ("runouts_enter_right_censored_likelihood", n_runout == EXPECTED_RUNOUT_ROWS,
         "runouts enter density/survival model selection but never exact point metrics"),
        ("fair_censored_comparators_present", censored_trace_complete,
         "both censored AFT comparators have metrics, one parameter record per outer fold, and positive runout training counts"),
        ("physics_active_in_all_outer_folds", bool(physics_mix.notna().all() and physics_mix.gt(0).all()),
         f"active physical weight folds={int(physics_mix.gt(0).sum())}/{len(physics_mix)}"),
        ("development_only_model_selection", selection_trace_complete,
         selection_trace_detail),
        ("response_free_temperature_extrapolation_guard", temperature_gate_verified,
         "each outer-test exact-temperature availability flag is reproduced from development exact-fracture temperatures only; this binary flag does not imply zero residual under soft support"),
        ("best_primary_campaign_balanced_model", best_primary == PROPOSED_MODEL,
         f"lowest primary campaign-disjoint, campaign-balanced RMSE among pre-specified models: {best_primary}"),
        ("reported_model_matches_selected_residual_branch",
         abs(float(proposed["SB_RMSE"]) - float(anchor["SB_RMSE"])) < 1e-10,
         f"anchor RMSE={anchor['SB_RMSE']:.4f}; reported RMSE={proposed['SB_RMSE']:.4f}"),
    ]
    for comparator, row in paired.items():
        checks.append((
            f"beats_{comparator}_point", bool(np.isfinite(row["estimate"]) and float(row["estimate"]) > 0),
            f"paired RMSE gain={float(row['estimate']):.4f}" if np.isfinite(row["estimate"]) else "not evaluable",
        ))
        checks.append((
            f"beats_{comparator}_CI", rmse_ci_pass[comparator],
            (f"95% campaign-bootstrap CI [{float(row['CI95_lo']):.4f}, {float(row['CI95_hi']):.4f}]"
             if np.isfinite(row["CI95_lo"]) else "not evaluable"),
        ))
    checks.extend([
        ("primary_RMSE_superiority_gate", primary_rmse_superiority,
         "RMSE-gain 95% campaign-bootstrap CI is above zero versus ML-Ens, ExtraTrees, ET-Walker and ML-Strong"),
        ("primary_multimetric_superiority_gate", primary_multimetric_superiority,
         "requires positive 95% CIs for RMSE, MAE, R2, F5 and C* gains versus all four primary comparators"),
        ("external_domain_point_superiority", external_point_superiority,
         f"WB-PIML has the lowest RMSE in every LOSO/LOAO/LOTO/LOEO factor-held-out scenario: {external_point_passes}"),
        ("external_domain_CI_superiority", external_ci_superiority,
         "requires positive paired RMSE-gain CIs versus all four comparators in every factor-held-out scenario"),
        ("external_nested_selection_complete", external_nested_selection_complete,
         external_selection_detail),
        ("external_censor_guard_never_relaxed",
         external_censor_guard_never_relaxed,
         f"WB-PIML domain holdouts failing the runout diagnostic="
         f"{external_relaxed_count}/{external_wb_holdout_count}"),
        ("external_interval_audit_complete", external_interval_complete,
         "all LOSO/LOAO/LOTO/LOEO WB-PIML exact predictions have development-only 80% and 90% intervals"),
        ("external_interval_nominal_compatible", external_interval_nominal_compatible,
         "each scenario-level empirical coverage CI must contain its nominal 80% or 90% target"),
        ("runout_survival_noninferiority", runout_noninferior,
         f"outer runout survival NLL: WB-PIML={proposed_runout_nll:.4f}; WB-CD={reference_runout_nll:.4f}"),
        ("censor_guard_never_relaxed", censor_guard_never_relaxed,
         f"selected folds failing the runout diagnostic={relaxed_count}/{len(anchor_selection)}; diagnostic is not a selection filter"),
        ("mechanistic_damage_constraints", mechanistic_constraints_verified,
         "all reported WB-PIML folds have six finite physical parameters including a nonnegative log10 water multiplier, positive Walker slope, bounded non-negative environmental exponents, and active physics"),
        ("mechanistic_parameters_interior", mechanistic_parameters_interior,
         f"competing-damage parameter boundary hits={int(np.sum(boundary_hits))}/{len(boundary_hits)} folds"),
        ("overall_claim_gate", universal_superiority,
         "broad claims require primary multi-metric superiority, complete training-only nested selection and superiority in domain-held-out tests, outer runout non-inferiority, passing primary and domain-holdout runout diagnostics, interior physical parameters and compatible external interval coverage"),
    ])
    table = pd.DataFrame(checks, columns=["gate", "passed", "evidence"])
    table["evidence_origin"] = np.where(
        table["gate"].isin(["workbook_audit_complete", "human_reviewed_campaign_map",
                            "label_conflicts_resolved", "physical_duplicate_audit"]),
        "workbook", "computed",
    )
    return table


def _raw_duplicate_variant(formal: pd.DataFrame, use_source_labels: bool = False
                           ) -> pd.DataFrame:
    raw = AUDIT_CONTEXT.get("raw")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise RuntimeError("Plot_Data is unavailable for duplicate sensitivity")
    original = AUDIT_CONTEXT.get("original")
    original_by_row = (
        original.set_index("row_id") if use_source_labels
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
            source_record = original_by_row.loc[int(raw_row["row_id"])]
            if isinstance(source_record, pd.DataFrame):
                source_record = source_record.iloc[0]
            nf = float(pd.to_numeric(source_record["Nf_cycles"], errors="coerce"))
            if np.isfinite(nf) and nf > 0:
                clone_row["Nf_cycles"] = nf
                clone_row["logN"] = math.log10(nf)
            runout_value = parse_bool(pd.Series([source_record["is_runout"]]), "source is_runout").iloc[0]
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
    raw_equal = _raw_duplicate_variant(formal, use_source_labels=False)
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
            _raw_duplicate_variant(formal, use_source_labels=True),
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


def campaign_audit_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit all 23 physical duplicate groups from Plot_Data, not only retained rows."""
    raw = AUDIT_CONTEXT.get("raw")
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise RuntimeError("Plot_Data is required for duplicate provenance audit")
    raw = raw.copy()
    raw["record_equivalence_id"] = raw["record_equivalence_id"].fillna("").astype(str).str.strip()
    counts = raw.groupby("record_equivalence_id").size()
    duplicate_ids = set(counts.loc[counts.gt(1)].index)
    reviewed = df.set_index("record_equivalence_id", drop=False)
    fold_by_campaign = df.groupby("campaign_id")["outer_fold"].first().to_dict()
    cluster_rows = []
    for equivalence_id in sorted(duplicate_ids):
        group = raw.loc[raw["record_equivalence_id"].eq(equivalence_id)].copy()
        supplied_runout = parse_bool(group["is_runout"], "Plot_Data.is_runout")
        representative = reviewed.loc[equivalence_id]
        if isinstance(representative, pd.DataFrame):
            representative = representative.iloc[0]
        cluster_rows.append({
            "record_equivalence_id": equivalence_id,
            "campaign_id": representative["campaign_id"],
            "n_rows": len(group), "n_sources": group["source_id"].astype(str).nunique(),
            "n_exact": int((~supplied_runout).sum()), "n_runout": int(supplied_runout.sum()),
            "supplied_label_conflict": bool(supplied_runout.nunique() > 1),
            "reviewed_label_conflict": bool(representative["label_conflict_flag"]),
            "label_conflict_resolved": bool(representative["label_conflict_resolved"]),
            "n_label_overrides": 0, "n_outer_folds": 1,
            "source_ids": " | ".join(sorted(group["source_id"].astype(str).unique())),
            "retained_row_id": int(representative["row_id"]),
        })
    clusters = pd.DataFrame(cluster_rows)
    cross_source_ids = (
        set(clusters.loc[
            clusters["n_sources"].gt(1), "record_equivalence_id"
        ].astype(str))
        if not clusters.empty else set()
    )

    candidates = raw.loc[raw["record_equivalence_id"].isin(duplicate_ids)].copy()
    candidates["cross_source_duplicate_candidate"] = (
        candidates["record_equivalence_id"].astype(str).isin(cross_source_ids)
    )
    candidates["supplied_is_runout"] = parse_bool(candidates["is_runout"], "Plot_Data.is_runout")
    candidates["supplied_is_exact_failure"] = ~candidates["supplied_is_runout"]
    candidates["reviewed_is_runout"] = candidates["record_equivalence_id"].map(
        df.set_index("record_equivalence_id")["is_runout"]
    ).astype(bool)
    candidates["reviewed_is_exact_failure"] = ~candidates["reviewed_is_runout"]
    candidates["is_runout"] = candidates["reviewed_is_runout"]
    candidates["is_exact"] = candidates["reviewed_is_exact_failure"]
    candidates["label_override_applied"] = False
    candidates["campaign_id"] = candidates["record_equivalence_id"].map(
        df.set_index("record_equivalence_id")["campaign_id"]
    )
    candidates["outer_fold"] = candidates["campaign_id"].map(fold_by_campaign).astype(int)
    candidates["supplied_label_conflict_flag"] = candidates["record_equivalence_id"].map(
        clusters.set_index("record_equivalence_id")["supplied_label_conflict"]
    ).astype(bool)
    candidates["label_conflict_flag"] = False
    candidates = candidates.sort_values(
        ["record_equivalence_id", "source_id", "row_id"]
    ).reset_index(drop=True)

    campaign_map = df.groupby(["campaign_id", "source_id"], sort=True).agg(
        n_rows=("row_id", "size"), n_exact=("is_exact", "sum"),
        n_runout=("is_runout", "sum"), outer_fold=("outer_fold", "first"),
        representative_source=("campaign_representative_source", "first"),
        source_campaign_role=("source_campaign_role", "first"),
        reviewed_secondary_source=("reviewed_secondary_source", "first"),
        campaign_mapping_status=("campaign_mapping_status", "first"),
        mapping_basis=("campaign_mapping_basis", "first"),
        audit_status=("audit_status", "first"), evidence_grade=("evidence_grade", "first"),
    ).reset_index()
    return clusters, candidates, campaign_map


def probability_row_id_hash(values):
    text = ",".join(str(int(value)) for value in sorted(values))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ordinary_probability_oof(train, selection_seed, trees, et_choice, walker_choice):
    """Calibrate fixed selected point models using development-only OOF predictions."""
    splits, _ = grouped_folds(train, min(4, train[PRIMARY_GROUP_COLUMN].nunique()), selection_seed)
    names = ("RF", "ExtraTrees", "GBDT", "SVR", "ET-Walker", "ML-Ens")
    pieces = {name: [] for name in names}
    for inner_fold, (fit_idx, valid_idx) in enumerate(splits, 1):
        fit = train.iloc[fit_idx].reset_index(drop=True)
        valid = train.iloc[valid_idx].reset_index(drop=True)
        if set(fit.campaign_id) & set(valid.campaign_id):
            raise AssertionError("Probability calibration campaign leakage")
        fit_seed = selection_seed + 100 * inner_fold + 1
        output = external_predictions(
            fit, valid, fit_seed, trees,
            int(et_choice["min_samples_leaf"]), float(et_choice["max_features"]),
            float(walker_choice["gamma"]), int(walker_choice["min_samples_leaf"]),
            float(walker_choice["max_features"]),
        )
        for name in names:
            part = valid[["row_id", "source_id", "campaign_id", "record_equivalence_id",
                          "logN", "is_exact", "is_runout"]].copy()
            part["inner_fold"] = inner_fold
            part["mu"] = np.asarray(output[name], float)
            part["fit_seed"] = fit_seed
            pieces[name].append(part)
    output = {}
    for name, parts in pieces.items():
        oof = pd.concat(parts, ignore_index=True)
        if len(oof) != len(train) or not oof.row_id.is_unique or set(oof.row_id) != set(train.row_id):
            raise AssertionError("Probability OOF must cover each development record once")
        output[name] = (oof, calibrate_aft_scale(oof))
    return output


def probability_calibration_trace(train, oof, model, sigma, scenario, fold,
                                  holdout_group, split_seed, configuration,
                                  method, fit_seed_rule="deterministic_physical_fit", trees=None,
                                  family="lognormal"):
    """Record calibration provenance separately from the point-analysis tables."""
    common = {"model": model, "scenario": scenario, "fold": int(fold),
              "holdout_group": str(holdout_group), "prob_family": family,
              "prob_scale": float(sigma), "sigma": float(sigma),
              "calibration_method": method, "split_seed": split_seed,
              "fit_seed_rule": fit_seed_rule, "trees": trees,
              "configuration_json": json.dumps(configuration, sort_keys=True, default=str),
              "n_development": len(train), "n_development_exact": int(train.is_exact.sum()),
              "n_development_runout": int(train.is_runout.sum()),
              "development_row_ids_sha256": probability_row_id_hash(train.row_id),
              "calibration_only": False,
              "outer_test_used_for_calibration": False,
              "scale_lower_bound": 0.08, "scale_upper_bound": 4.0,
              "scale_at_bound": bool(np.isclose(sigma, .08) or np.isclose(sigma, 4.0))}
    if oof is None:
        part = pd.DataFrame([{**common, "record_type": "native_fit_metadata",
                              "fit_row_ids_sha256": common["development_row_ids_sha256"],
                              "n_fit": len(train), "n_inner_folds": 0}])
        for key in ("is_exact", "is_runout"):
            part[key] = pd.Series([pd.NA], dtype="boolean")
        return part
    if not oof.row_id.is_unique or set(oof.row_id) != set(train.row_id):
        raise AssertionError("Incomplete calibration provenance")
    part = oof.copy()
    aligned = train.set_index("row_id").loc[part.row_id]
    for key in ("logN", "is_exact", "is_runout"):
        if not np.array_equal(part[key].to_numpy(), aligned[key].to_numpy()):
            raise AssertionError("Calibration outcomes do not match development records")
    for key, value in common.items():
        part[key] = value
    for key in ("is_exact", "is_runout"):
        part[key] = part[key].astype("boolean")
    part["record_type"] = "calibration_oof"
    part["n_inner_folds"] = int(part.inner_fold.nunique())
    part["prob_location"] = part["mu"].to_numpy(float)
    part["pred_logN"] = part["mu"].to_numpy(float)
    nll = joint_aft_nll(part, part.mu.to_numpy(float), float(sigma))
    for key, value in nll.items():
        part["calibration_" + key] = value
    for inner_fold, group in part.groupby("inner_fold"):
        fit = train.loc[~train.row_id.isin(group.row_id)]
        if set(fit.campaign_id) & set(group.campaign_id):
            raise AssertionError("Calibration trace contains overlapping campaigns")
        index = group.index
        part.loc[index, "fit_row_ids_sha256"] = probability_row_id_hash(fit.row_id)
        part.loc[index, "validation_row_ids_sha256"] = probability_row_id_hash(group.row_id)
        part.loc[index, "n_fit"] = len(fit)
        part.loc[index, "n_validation"] = len(group)
        if "fit_seed" not in oof and fit_seed_rule == "split_seed+100*inner_fold+1":
            part.loc[index, "fit_seed"] = int(split_seed) + 100 * int(inner_fold) + 1
    return part


def probability_prediction_frame(frame, model, sigma, family, scenario, fold, holdout_group):
    part = frame.copy()
    if not np.isfinite(float(sigma)) or float(sigma) <= 0:
        raise ValueError("Probability scale must be positive and finite")
    part["model"] = model
    part["prob_location"] = part["pred_logN"].to_numpy(float)
    part["prob_scale"] = float(sigma)
    part["prob_family"] = family
    part["scenario"] = scenario
    part["fold"] = int(fold)
    part["holdout_group"] = str(holdout_group)
    return part


def main_probability_records(train, prediction_parts, oof_sources, native_results,
                             et_choice, walker_choice, selection_seed, trees, outer_fold):
    ordinary = ordinary_probability_oof(train, selection_seed, trees, et_choice, walker_choice)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    records, calibration = [], []
    selected = [spec.key for spec in PHYSICAL_SPECS] + [PROPOSED_MODEL] + list(EXTERNAL_MODELS) + list(CENSORED_MODELS) + list(SIMPLE_CONTROL_MODELS)
    for name in selected:
        rows = predictions.loc[predictions.model.eq(name)].copy()
        if rows.empty or not rows.row_id.is_unique or set(rows.row_id) & set(train.row_id):
            raise AssertionError("Invalid primary probability row coverage")
        if name in ordinary:
            oof, scale = ordinary[name]
            sigma, family = float(scale["sigma"]), "lognormal"
            configuration = {"ExtraTrees": et_choice, "ET-Walker": walker_choice,
                             "point_model": name, "ensemble_members": ["RF", "ExtraTrees", "GBDT", "SVR"] if name == "ML-Ens" else []}
            trace = probability_calibration_trace(train, oof, name, sigma, "Primary", outer_fold,
                "", selection_seed, configuration, "fixed_selected_configuration_oof_censored_scale",
                "split_seed+100*inner_fold+1", trees)
        elif name in native_results:
            result = native_results[name]
            sigma, family = float(result["sigma"]), str(result["family"])
            trace = probability_calibration_trace(train, None, name, sigma, "Primary", outer_fold,
                "", selection_seed + 1000 * list(CENSORED_MODELS).index(name),
                {"alpha": result["alpha"], "family": family}, "native_joint_location_scale_fit",
                "deterministic_native_aft_fit", family=family)
        else:
            oof, sigma, split_seed, configuration, seed_rule = oof_sources[name]
            family = "lognormal"
            trace = probability_calibration_trace(train, oof, name, sigma, "Primary", outer_fold,
                "", split_seed, configuration, "retained_development_oof_censored_scale", seed_rule, trees if name in (PROPOSED_MODEL, "ML-Strong") + SIMPLE_CONTROL_MODELS else None)
        records.append(probability_prediction_frame(rows, name, sigma, family, "Primary", outer_fold, ""))
        calibration.append(trace)
    if len(records) != 14 + len(SIMPLE_CONTROL_MODELS):
        raise AssertionError("Primary probability model coverage is incomplete")
    return records, calibration


def factor_probability_records(train, prediction_parts, proposed_oof, control_oof,
                               strong_scale, hybrid_choice, control_choice, et_choice,
                               walker_choice, selection_seed, trees, strategy, fold,
                               holdout_group, simple_results):
    ordinary = ordinary_probability_oof(train, selection_seed, trees, et_choice, walker_choice)
    physical_oof = inner_oof(train, CLASSIC_WB_SPEC, selection_seed)
    physical_scale = calibrate_aft_scale(physical_oof)
    proposed_scale = calibrate_aft_scale(proposed_oof)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    sources = {
        PROPOSED_MODEL: (proposed_oof, proposed_scale, hybrid_choice, "split_seed+100*inner_fold+1"),
        "ML-Strong": (control_oof, strong_scale, control_choice, "split_seed+100*inner_fold+1"),
        "WB": (physical_oof, physical_scale, {"spec": "WB", "gamma": GAMMA}, "deterministic_physical_fit"),
    }
    for name, result in simple_results.items():
        sources[name] = (result["oof"], result["scale"], result["choice"],
                         "split_seed+100*inner_fold+1")
    records, calibration = [], []
    for name in (PROPOSED_MODEL, "WB", "ExtraTrees", "ET-Walker", "ML-Ens", "ML-Strong") + SIMPLE_CONTROL_MODELS:
        rows = predictions.loc[predictions.model.eq(name)].copy()
        if rows.empty or not rows.row_id.is_unique or set(rows.row_id) & set(train.row_id):
            raise AssertionError("Invalid factor probability row coverage")
        if name in sources:
            oof, scale, config, seed_rule = sources[name]
            method = "retained_selected_oof_censored_scale" if name != "WB" else "physical_oof_censored_scale"
        else:
            oof, scale = ordinary[name]
            config = {"ExtraTrees": et_choice, "ET-Walker": walker_choice, "point_model": name}
            seed_rule = "split_seed+100*inner_fold+1"
            method = "fixed_selected_configuration_oof_censored_scale"
        sigma = float(scale["sigma"])
        records.append(probability_prediction_frame(rows, name, sigma, "lognormal", strategy, fold, holdout_group))
        calibration.append(probability_calibration_trace(train, oof, name, sigma, strategy, fold,
            holdout_group, selection_seed, config, method, seed_rule, trees if name != "WB" else None))
    for name in ("RF", "GBDT", "SVR"):
        oof, scale = ordinary[name]
        trace = probability_calibration_trace(train, oof, name, float(scale["sigma"]), strategy,
            fold, holdout_group, selection_seed, {"point_model": name},
            "fixed_selected_configuration_oof_censored_scale", "split_seed+100*inner_fold+1", trees)
        trace["calibration_only"] = True
        calibration.append(trace)
    return records, calibration


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
        {"model": "WB-CD", "role": 'right-censored competing-damage physical reference', "location_parameters": 6,
         "equation": 'mu=-log10(D_mech+D_env); exact density+runout survival; six location parameters; water enhancement estimated within training only',
         "used_for_selection": True},
        {"model": "WB-M", "role": "mechanical-proxy ablation", "location_parameters": 3,
         "equation": "a - b*log10(S_Walker) - c*M", "used_for_selection": False},
        {"model": "WB-Ox", "role": "oxidation-proxy ablation", "location_parameters": 3,
         "equation": "a - b*log10(S_Walker) - c*Dox", "used_for_selection": False},
        {"model": "WB-PIML-Anchor", "role": 'development-selected competing-damage residual branch', "location_parameters": 6,
         "equation": 'mu_phys+eta*soft250*residual(X); ExtraTrees, physics-feature ExtraTrees or Matérn; selected environmental-decrement penalty', "used_for_selection": True},
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
        "proposed_architecture": 'six-parameter censored-AFT competing-damage trunk with training-estimated water multiplier with selected environmental-decrement penalty, plus an exact-fracture ExtraTrees, physics-feature ExtraTrees, or Matérn residual',
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
        'revision_model_id': 'WB-PIML-learned-water-20260928',
        'environment_factors': {'vacuum':0.0,'air_or_oxygen':1.0,'water_oxygen':'10**beta, beta>=0; training-only fit'},
        'environment_factor_meaning': 'air reference; water direction-constrained nuisance parameter; no universal physical calibration claim',
        'revision_reason': 'replace fixed multiplier by training-only nonnegative log enhancement; post-review and post-sensitivity decision; identifiability assessed separately',
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


# ===== REVISED PRIMARY CODE END =====
