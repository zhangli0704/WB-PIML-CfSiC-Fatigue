

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
        "arrhenius_temperature_sensitivity", "n_physics_parameters",
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
        and physical_values["n_physics_parameters"].eq(5).all()
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
         "all reported WB-PIML folds have five finite physical parameters, positive Walker slope, bounded non-negative environmental exponents, and active physics"),
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
        "runouts_used_for_scale_and_selection": True, "n_physics_parameters": 5 if control_choice["family"]=="physics_feature_ET" else 0,
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
