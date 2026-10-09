def claim_gate(metrics: pd.DataFrame, effects: pd.DataFrame, parameters: pd.DataFrame,
               df: pd.DataFrame, nested_selection: pd.DataFrame,
               nested_candidates: pd.DataFrame, runout: pd.DataFrame,
               external_summary: pd.DataFrame,
               external_groups: pd.DataFrame,
               external_effects: pd.DataFrame,
               external_uq: pd.DataFrame,
               external_selection: pd.DataFrame,
               temperature_gate_rows: pd.DataFrame) -> pd.DataFrame:

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

