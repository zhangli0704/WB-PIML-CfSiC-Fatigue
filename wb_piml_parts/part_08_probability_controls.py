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

WB-Residual uses the classic Walker-Basquin trunk (18 configurations); Basquin-Residual fixes gamma=0 (6 configurations). Each independently selects from the same three residual families and two gains on the original development campaign folds. The environmental penalty is absent. Basquin-Residual retains R and the same environment-independent residual inputs as WB-PIML; it removes Walker correction from the trunk. All residual families exclude environment labels and environmental derived features. Both use development OOF censored-scale calibration and empirical interval calibration.

The two controls enter primary validation and all four factor holdouts, with point, probability, and interval results in the existing tables. Trunk_Metrics, Trunk_Paired_CI, Trunk_Empirical_PI and Trunk_Method collect the comparisons; 10_trunk_comparison.png summarizes them. Trunk_External_Candidates retains all added factor-holdout candidate scores. Original fixed-configuration repeated validation is unchanged, so its new-control console entries are blank.

Positive gains in Trunk_Paired_CI favor the model named proposed, including the WB-Residual versus Basquin-Residual comparison. These are revision-stage comparisons on previously studied data. They do not select a new primary model automatically.
"""
    readme_path.write_text(text, encoding="utf-8")


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
                
                "strength": 0.0,
            })
    return tuple(candidates)


SIMPLE_HYBRID_CANDIDATES = simple_hybrid_candidate_grid()


def predict_simple_hybrid_configs(train, valid, seed, configs, trees=400):

    require_exact_training(train, "predict_simple_hybrid_configs")
    if not len(valid):
        raise ValueError("Simple-hybrid validation rows must be nonempty")
    exact = train.loc[train.is_exact].reset_index(drop=True)
    full = pd.concat([exact, valid], ignore_index=True)
    y = exact.logN.to_numpy(float)
    prep = make_residual_preprocessor()
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
                    aug["walker_stress"] = walker_log_stress(full, gamma)
                    extra = RESIDUAL_PHYSICS_NUMERIC
                    aug_prep = make_residual_preprocessor(extra)
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


        choice["censor_feasible"] = np.nan
        choice["censor_constraint_relaxed"] = False
        candidates["censor_feasible"] = np.nan
        candidates["censor_constraint_relaxed"] = False
        outputs[model] = (choice, stored[int(choice["candidate_index"])], candidates)
    return outputs


def fit_selected_simple_control(train, valid, choice, seed, trees):

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

