"""Physical fatigue models, comparator models and residual learning."""

from __future__ import annotations

import math
from scipy.optimize import minimize
from scipy.special import log_ndtr
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from data import (
    CALIBRATION_MODES,
    CENSORED_MODELS,
    CLASSIC_WB_SPEC,
    COMPETING_WB_SPEC,
    CompetingDamageConfig,
    DAMAGE_CAP,
    DEFAULT_COMPETING_CONFIG,
    GAMMA,
    HYBRID_CANDIDATES,
    NESTED_CANDIDATES,
    PHYSICS_MIX_GRID,
    PRIMARY_GROUP_COLUMN,
    PhysicalSpec,
    RESIDUAL_TEMPERATURE_MATCH_ATOL_C,
    RMSE_TIE_TOLERANCE,
    TRAD_CATEGORICAL,
    TRAD_NUMERIC,
    LOTO_SHORTLIST,
    NEAR_TIE_RMSE,
    censored_hinge_rmse,
    grouped_folds,
    require_exact_training,
    sb_metrics,
    training_weights,
)


def joint_aft_nll(frame: pd.DataFrame, mu: np.ndarray, sigma: float,
                  weights: np.ndarray | None = None) -> dict[str, float]:
    """Lognormal AFT NLL with exact density and right-censored survival terms."""
    require_exact_training(frame, "joint_aft_nll")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("AFT sigma must be positive and finite")
    location = np.asarray(mu, float)
    if len(location) != len(frame) or not np.isfinite(location).all():
        raise ValueError("AFT location prediction is missing or non-finite")
    weight = training_weights(frame) if weights is None else np.asarray(weights, float)
    y = frame["logN"].to_numpy(float)
    exact_mask = frame["is_exact"].to_numpy(bool)
    runout_mask = ~exact_mask
    z = (y - location) / sigma
    losses = np.empty(len(frame), float)
    losses[exact_mask] = (
        0.5 * math.log(2.0 * math.pi) + math.log(sigma)
        + 0.5 * z[exact_mask] ** 2
    )

    losses[runout_mask] = -log_ndtr(-z[runout_mask])
    total = float(np.sum(weight * losses) / np.sum(weight))
    exact_nll = float(np.average(losses[exact_mask], weights=weight[exact_mask]))
    runout_nll = (
        float(np.average(losses[runout_mask], weights=weight[runout_mask]))
        if runout_mask.any() else np.nan
    )
    return {
        "joint_censored_NLL": total,
        "exact_density_NLL": exact_nll,
        "runout_survival_NLL": runout_nll,
    }


def walker_log_stress(df: pd.DataFrame, gamma: float = GAMMA) -> np.ndarray:
    factor = np.clip(0.5 * (1.0 - df["R"].to_numpy(float)), 1e-6, None)
    return df["logS"].to_numpy(float) + gamma * np.log10(factor)


def physical_design(df: pd.DataFrame, spec: PhysicalSpec, gamma: float = GAMMA) -> np.ndarray:
    if spec.mode != "linear":
        raise ValueError(f"{spec.key} uses a nonlinear competing-damage equation")
    columns = [np.ones(len(df)), -walker_log_stress(df, gamma)]
    columns += [-df[channel].to_numpy(float) for channel in spec.channels]
    return np.column_stack(columns)


def _fit_linear_physical(frame: pd.DataFrame, spec: PhysicalSpec,
                         gamma: float) -> np.ndarray:
    """Fit a physical location with joint exact/right-censored AFT likelihood."""
    require_exact_training(frame, f"_fit_linear_physical[{spec.key}]")
    x = physical_design(frame, spec, gamma)
    exact_mask = frame["is_exact"].to_numpy(bool)
    initial_coef = np.linalg.lstsq(
        x[exact_mask], frame.loc[exact_mask, "logN"].to_numpy(float), rcond=None
    )[0]
    initial_coef[1] = np.clip(initial_coef[1], 0.05, 20.0)
    if len(initial_coef) > 2:
        initial_coef[2:] = np.clip(initial_coef[2:], 0.0, DAMAGE_CAP)
    initial_sigma = np.clip(
        np.std(frame.loc[exact_mask, "logN"].to_numpy(float) - x[exact_mask] @ initial_coef),
        0.10, 3.0,
    )

    def loss(parameter: np.ndarray) -> float:
        coef = parameter[:-1]
        sigma = float(np.exp(parameter[-1]))
        nll = joint_aft_nll(frame, x @ coef, sigma)["joint_censored_NLL"]
        ridge = 1e-4 * float(np.sum(coef[2:] ** 2)) if len(coef) > 2 else 0.0
        return nll + ridge

    initial = np.r_[initial_coef, math.log(initial_sigma)]
    bounds = ([(None, None), (0.05, 20.0)]
              + [(0.0, DAMAGE_CAP)] * len(spec.channels)
              + [(math.log(0.08), math.log(4.0))])
    result = minimize(
        loss, initial, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success or not np.isfinite(result.fun):

        retry_start = (
            np.asarray(result.x, float)
            if np.size(result.x) == np.size(initial)
            and np.all(np.isfinite(result.x))
            else initial
        )
        fallback = minimize(
            loss, retry_start, method="Powell", bounds=bounds,
            options={"maxiter": 6000, "xtol": 1e-9, "ftol": 1e-10},
        )
        if fallback.success and np.isfinite(fallback.fun):
            result = fallback
        else:
            raise RuntimeError(
                f"{spec.key} censored-likelihood optimization failed: "
                f"L-BFGS-B={result.message}; Powell={fallback.message}"
            )
    return np.asarray(result.x[:-1], float)


def _predict_competing_damage(df: pd.DataFrame, coef: np.ndarray,
                              gamma: float) -> np.ndarray:
    """Predict log10 life from competing per-cycle damage rates."""
    if len(coef) != 5:
        raise ValueError("The competing-damage trunk requires five parameters")
    intercept, b, log_k, m, q_t = np.asarray(coef, float)
    log_sw = walker_log_stress(df, gamma)
    log_d_mech = -intercept + b * log_sw

    exposure = df["environment_exposure"].to_numpy(float)
    active = exposure > 0.0
    log_d_env = np.full(len(df), -np.inf, dtype=float)
    log_d_env[active] = (
        log_d_mech[active]
        + log_k
        + np.log10(exposure[active])
        + (q_t * df.loc[active, "arrhenius_temperature_drive"].to_numpy(float)
           / math.log(10.0))
        + m * log_sw[active]
        - df.loc[active, "logf"].to_numpy(float)
    )

    maximum = np.maximum(log_d_mech, log_d_env)
    log_total_damage = maximum + np.log10(
        np.power(10.0, log_d_mech - maximum)
        + np.power(10.0, log_d_env - maximum)
    )
    return -log_total_damage


def fit_physical(train: pd.DataFrame, spec: PhysicalSpec, gamma: float = GAMMA,
                 competing_config: CompetingDamageConfig = DEFAULT_COMPETING_CONFIG,
                 ) -> np.ndarray:
    require_exact_training(train, f"fit_physical[{spec.key}]")
    exact = train.loc[train["is_exact"]].reset_index(drop=True)
    if spec.mode == "linear":
        return _fit_linear_physical(train, spec, gamma)
    if spec.mode != "competing_damage":
        raise ValueError(f"Unknown physical mode: {spec.mode}")

    classic = _fit_linear_physical(train, CLASSIC_WB_SPEC, gamma)
    classic_intercept = float(classic[0])
    classic_b = float(classic[1])

    def loss(parameter: np.ndarray) -> float:
        coef = parameter[:-1]
        sigma = float(np.exp(parameter[-1]))
        prediction = _predict_competing_damage(train, coef, gamma)
        nll = joint_aft_nll(train, prediction, sigma)["joint_censored_NLL"]

        ridge = float(competing_config.ridge_strength) * (
            (coef[0] - classic_intercept) ** 2
            + 0.10 * (coef[1] - classic_b) ** 2
            + 0.01 * coef[3] ** 2
            + 0.01 * coef[4] ** 2
        )
        return nll + ridge

    bounds = [
        (classic_intercept - competing_config.intercept_half_width,
         classic_intercept + competing_config.intercept_half_width),
        (max(competing_config.walker_min,
             classic_b - competing_config.walker_half_width),
         min(competing_config.walker_max,
             classic_b + competing_config.walker_half_width)),
        (competing_config.environment_ratio_min,
         competing_config.environment_ratio_max),
        (competing_config.environment_stress_min,
         competing_config.environment_stress_max),
        (competing_config.arrhenius_min,
         competing_config.arrhenius_max),
    ]
    starts = [
        np.array([classic_intercept, classic_b, -2.0, 1.0, 4.0, math.log(0.8)]),
        np.array([classic_intercept, classic_b, -4.0, 2.0, 7.0, math.log(1.2)]),
        np.array([classic_intercept, classic_b, 0.0, 4.0, 2.0, math.log(1.0)]),
    ]
    bounds = bounds + [(math.log(0.08), math.log(4.0))]
    results = [
        minimize(loss, np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds]),
                 method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-8})
        for start in starts
    ]
    valid = [result for result in results if result.success and np.isfinite(result.fun)]
    if not valid:
        messages = "; ".join(str(result.message) for result in results)
        raise RuntimeError(f"{spec.key} optimization failed: {messages}")
    best = min(valid, key=lambda result: float(result.fun))
    return np.asarray(best.x[:-1], float)


def predict_physical(df: pd.DataFrame, spec: PhysicalSpec, coef: np.ndarray,
                     gamma: float = GAMMA) -> np.ndarray:
    if spec.mode == "competing_damage":
        return _predict_competing_damage(df, coef, gamma)
    return physical_design(df, spec, gamma) @ coef


def physical_parameter_values(spec: PhysicalSpec, coef: np.ndarray) -> dict[str, float]:
    """Return named parameters without pretending nonlinear coefficients are intercepts."""
    values = {
        "intercept": np.nan,
        "walker_slope": np.nan,
        "mechanical_log10_damage_scale": np.nan,
        "environment_log10_damage_ratio": np.nan,
        "environment_stress_exponent": np.nan,
        "arrhenius_temperature_sensitivity": np.nan,
        "M_coefficient": np.nan,
        "Ox_coefficient": np.nan,
        "Q_coefficient": np.nan,
    }
    if spec.mode == "competing_damage":
        values.update({
            "intercept": float(coef[0]),
            "mechanical_log10_damage_scale": float(-coef[0]),
            "walker_slope": float(coef[1]),
            "environment_log10_damage_ratio": float(coef[2]),
            "environment_stress_exponent": float(coef[3]),
            "arrhenius_temperature_sensitivity": float(coef[4]),
        })
    else:
        values["intercept"] = float(coef[0])
        values["walker_slope"] = float(coef[1])
        for channel, value in zip(spec.channels, coef[2:]):
            values[f"{channel}_coefficient"] = float(value)
    return values


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


def make_preprocessor(extra_numeric: tuple[str, ...] = ()) -> ColumnTransformer:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [("numeric", StandardScaler(), TRAD_NUMERIC + list(extra_numeric)),
         ("categorical", encoder, TRAD_CATEGORICAL)],
        sparse_threshold=0.0,
    )


def choose_rmse_first(candidates: pd.DataFrame) -> pd.Series:
    """Protect exact RMSE, then prefer the lowest joint censored NLL."""
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
    """Return the always-active physical hybrid; legacy arguments are audited."""
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
    """Finalize the selected residual branch without switching off physics."""
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
        max_features=max_features, random_state=seed, n_jobs=-1,
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
    """Exact-only direct ExtraTrees control with a Walker-stress feature."""
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
                max_features=max_features, random_state=seed, n_jobs=-1,
            )),
        ])

    pipe = new_pipe()
    pipe.fit(exact, exact["logN"], model__sample_weight=training_weights(exact))
    return np.asarray(pipe.predict(valid_aug), float)


def nested_select_et_walker(train: pd.DataFrame, seed: int, trees: int
                            ) -> tuple[dict[str, float], pd.DataFrame]:
    """Give ET-Walker exactly the WB-PIML inner folds and candidate budget."""
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
        "RF": RandomForestRegressor(n_estimators=trees, min_samples_leaf=3, max_features=0.75, random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=trees, min_samples_leaf=et_min_samples_leaf,
                                           max_features=et_max_features, random_state=seed + 1, n_jobs=-1),
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
        losses[~exact_mask] = -log_ndtr(-z[~exact_mask])
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


def make_smooth_residual_preprocessor() -> ColumnTransformer:
    """Low-variance residual branch with an explicit smooth temperature basis."""
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    temperature_pipe = Pipeline([
        ("spline", SplineTransformer(n_knots=4, degree=2, include_bias=False)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("numeric", StandardScaler(), TRAD_NUMERIC),
        ("temperature_spline", temperature_pipe, ["T_C"]),
        ("categorical", encoder, TRAD_CATEGORICAL),
    ], sparse_threshold=0.0)


def temperature_support_score(train: pd.DataFrame, valid: pd.DataFrame,
                              scale_c: float, unseen_arch_factor: float = 0.70,
                              unseen_env_factor: float = 0.70) -> np.ndarray:
    """Training-only support score; no validation response is consulted."""
    if scale_c <= 0:
        raise ValueError("temperature_scale must be positive")
    exact = train.loc[train["is_exact"]]
    support_t = np.unique(exact["T_C"].to_numpy(float))
    if not len(support_t):
        return np.zeros(len(valid), float)
    distance = np.min(
        np.abs(valid["T_C"].to_numpy(float)[:, None] - support_t[None, :]), axis=1
    )
    score = np.exp(-np.square(distance / float(scale_c)))
    seen_arch = set(exact["architecture"].astype(str))
    seen_env = set(exact["env_class"].astype(str))
    score *= np.where(
        valid["architecture"].astype(str).isin(seen_arch), 1.0,
        float(unseen_arch_factor),
    )
    score *= np.where(
        valid["env_class"].astype(str).isin(seen_env), 1.0,
        float(unseen_env_factor),
    )
    return np.clip(score, 0.0, 1.0)


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


def fit_hybrid_feature_model(train: pd.DataFrame, valid: pd.DataFrame,
                             spec: PhysicalSpec, seed: int, trees: int,
                             gamma: float = GAMMA, min_samples_leaf: int = 3,
                             max_features: float = 0.85, eta: float = 0.75,
                             residual_mode: str = "legacy_et",
                             ridge_alpha: float = 3.0,
                             temperature_scale: float = 250.0,
                             competing_config: CompetingDamageConfig = DEFAULT_COMPETING_CONFIG,
                             exact_temperature_gate_enabled: bool = True,
                             robust_tree_weight: float = 0.65,
                             unsupported_smooth_shrink: float = 0.25,
                             unseen_arch_factor: float = 0.70,
                             unseen_env_factor: float = 0.70,
                             ) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    """Always-physical trunk plus a predeclared exact-fracture residual."""
    require_exact_training(train, f"fit_hybrid_feature_model[{spec.key}]")
    if residual_mode not in {"legacy_et", "dual_support"}:
        raise ValueError(f"Unknown residual_mode: {residual_mode}")
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    for name, value in {
        "robust_tree_weight": robust_tree_weight,
        "unsupported_smooth_shrink": unsupported_smooth_shrink,
        "unseen_arch_factor": unseen_arch_factor,
        "unseen_env_factor": unseen_env_factor,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    exact = train.loc[train["is_exact"]].reset_index(drop=True)
    available_temperature_gate = exact_temperature_residual_gate(exact, valid)
    exact_temperature_gate = (
        available_temperature_gate
        if exact_temperature_gate_enabled
        else np.ones(len(valid), dtype=float)
    )
    temperature_gate_meta = summarize_temperature_gate(
        valid, available_temperature_gate
    )
    coef = fit_physical(train, spec, gamma, competing_config)
    base_valid = predict_physical(valid, spec, coef, gamma)
    exact_target = exact["logN"].to_numpy(float) - predict_physical(exact, spec, coef, gamma)
    if eta == 0.0 or not exact_temperature_gate.any():
        return np.asarray(base_valid, float), coef, {
            "eta": float(eta), "residual_mode": residual_mode,
            "exact_temperature_gate_enabled": bool(exact_temperature_gate_enabled),
            "robust_tree_weight": float(robust_tree_weight),
            "unsupported_smooth_shrink": float(unsupported_smooth_shrink),
            "unseen_arch_factor": float(unseen_arch_factor),
            "unseen_env_factor": float(unseen_env_factor),
            "active_runout_count": float(train["is_runout"].sum()),
            "active_runout_rate": float(train["is_runout"].mean()),
            "residual_correction_rms": 0.0,
            "residual_feature_importance_sum": 0.0,
            "mean_temperature_support": (
                float(np.mean(exact_temperature_gate)) if len(exact_temperature_gate) else np.nan
            ),
            **temperature_gate_meta,
        }

    et = Pipeline([
        ("preprocess", make_preprocessor()),
        ("model", ExtraTreesRegressor(
            n_estimators=trees, min_samples_leaf=min_samples_leaf,
            max_features=max_features, random_state=seed, n_jobs=-1,
        )),
    ])
    et.fit(exact, exact_target, model__sample_weight=training_weights(exact))
    tree_residual = np.asarray(et.predict(valid), float)
    support = np.ones(len(valid), float)
    smooth_residual = tree_residual
    if residual_mode == "dual_support":
        huber = Pipeline([
            ("preprocess", make_preprocessor()),
            ("model", GradientBoostingRegressor(
                n_estimators=160, learning_rate=0.025, max_depth=2,
                min_samples_leaf=max(3, min_samples_leaf), loss="huber",
                random_state=seed + 17,
            )),
        ])
        smooth = Pipeline([
            ("preprocess", make_smooth_residual_preprocessor()),
            ("model", Ridge(alpha=float(ridge_alpha))),
        ])
        weights = training_weights(exact)
        huber.fit(exact, exact_target, model__sample_weight=weights)
        smooth.fit(exact, exact_target, model__sample_weight=weights)
        robust_residual = (
            float(robust_tree_weight) * tree_residual
            + (1.0 - float(robust_tree_weight))
            * np.asarray(huber.predict(valid), float)
        )
        smooth_residual = np.asarray(smooth.predict(valid), float)
        support = temperature_support_score(
            exact, valid, temperature_scale,
            unseen_arch_factor=unseen_arch_factor,
            unseen_env_factor=unseen_env_factor,
        )

        learned_residual = (
            support * robust_residual
            + (1.0 - support) * float(unsupported_smooth_shrink) * smooth_residual
        )
    else:
        learned_residual = tree_residual

    learned_residual = exact_temperature_gate * learned_residual
    correction = float(eta) * learned_residual
    prediction = np.asarray(base_valid, float) + correction
    return prediction, coef, {
        "eta": float(eta), "residual_mode": residual_mode,
        "exact_temperature_gate_enabled": bool(exact_temperature_gate_enabled),
        "robust_tree_weight": float(robust_tree_weight),
        "unsupported_smooth_shrink": float(unsupported_smooth_shrink),
        "unseen_arch_factor": float(unseen_arch_factor),
        "unseen_env_factor": float(unseen_env_factor),
        "active_runout_count": float(train["is_runout"].sum()),
        "active_runout_rate": float(train["is_runout"].mean()),
        "residual_correction_rms": float(np.sqrt(np.mean(correction ** 2))) if len(correction) else 0.0,
        "residual_feature_importance_sum": float(np.sum(et.named_steps["model"].feature_importances_)),
        "mean_temperature_support": float(np.mean(support)) if len(support) else np.nan,
        **temperature_gate_meta,
    }


def _runout_fold_losses(oof: pd.DataFrame, sigma: float) -> pd.Series:
    runout = oof.loc[oof["is_runout"]].copy()
    if runout.empty:
        return pd.Series(dtype=float)
    z = (runout["mu"].to_numpy(float) - runout["logN"].to_numpy(float)) / float(sigma)
    runout["loss"] = -log_ndtr(z)
    return runout.groupby("inner_fold")["loss"].mean()


def _inner_loto_score(train: pd.DataFrame, candidate: dict[str, object],
                      seed: int, trees: int) -> float:
    pieces = []
    for index, level in enumerate(sorted(train["temp_bin"].astype(str).unique())):
        valid_mask = train["temp_bin"].astype(str).eq(level).to_numpy(bool)
        fit = train.loc[~valid_mask].reset_index(drop=True)
        valid = train.loc[valid_mask].reset_index(drop=True)
        if valid.empty or not fit["is_exact"].any() or not valid["is_exact"].any():
            continue
        mu, _, _ = fit_hybrid_feature_model(
            fit, valid, COMPETING_WB_SPEC, seed + 1000 * index + 31,
            min(trees, 160), float(candidate["gamma"]),
            int(candidate["min_samples_leaf"]), float(candidate["max_features"]),
            float(candidate["eta"]), str(candidate["residual_mode"]),
            float(candidate["ridge_alpha"]), float(candidate["temperature_scale"]),
        )
        part = valid.loc[valid["is_exact"], [
            "campaign_id", "record_equivalence_id", "logN"
        ]].copy()
        part["pred_logN"] = mu[valid["is_exact"].to_numpy(bool)]
        pieces.append(part)
    if not pieces:
        return np.inf
    pooled = pd.concat(pieces, ignore_index=True)
    return float(sb_metrics([
        group.reset_index(drop=True)
        for _, group in pooled.groupby(PRIMARY_GROUP_COLUMN)
    ])["SB_RMSE"])


def nested_select_hybrid(train: pd.DataFrame, seed: int, trees: int
                         ) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Censor-aware selection with a near-tie temperature robustness check."""
    require_exact_training(train, "nested_select_hybrid")
    splits, _ = grouped_folds(train, min(4, train[PRIMARY_GROUP_COLUMN].nunique()), seed)
    reference = inner_oof(train, COMPETING_WB_SPEC, seed)
    reference_scale = calibrate_aft_scale(reference)
    reference_runout = float(reference_scale["runout_survival_NLL"])
    reference_fold_loss = _runout_fold_losses(reference, float(reference_scale["sigma"]))
    rows: list[dict[str, object]] = []
    stored: dict[int, pd.DataFrame] = {}
    for candidate_index, candidate in enumerate(HYBRID_CANDIDATES):
        pieces = []
        for inner_fold, (fit_idx, valid_idx) in enumerate(splits, 1):
            fit = train.iloc[fit_idx].reset_index(drop=True)
            valid = train.iloc[valid_idx].reset_index(drop=True)
            mu, _, _ = fit_hybrid_feature_model(
                fit, valid, COMPETING_WB_SPEC,
                seed + 10000 * candidate_index + 100 * inner_fold + 1,
                min(trees, 300), float(candidate["gamma"]),
                int(candidate["min_samples_leaf"]), float(candidate["max_features"]),
                float(candidate["eta"]), str(candidate["residual_mode"]),
                float(candidate["ridge_alpha"]), float(candidate["temperature_scale"]),
            )
            part = valid[["row_id", "source_id", "campaign_id", "record_equivalence_id",
                          "logN", "is_exact", "is_runout", "temp_bin"]].copy()
            part["inner_fold"] = inner_fold
            part["mu"] = mu
            pieces.append(part)
        oof = pd.concat(pieces, ignore_index=True)
        exact = oof.loc[oof["is_exact"]].copy()
        exact["pred_logN"] = exact["mu"]
        exact_score = sb_metrics([
            group.reset_index(drop=True)
            for _, group in exact.groupby(PRIMARY_GROUP_COLUMN)
        ])["SB_RMSE"]
        scale = calibrate_aft_scale(oof)
        candidate_fold_loss = _runout_fold_losses(oof, float(scale["sigma"]))
        paired = pd.concat(
            [candidate_fold_loss.rename("candidate"), reference_fold_loss.rename("reference")],
            axis=1, join="inner",
        ).dropna()
        if len(paired) > 1:
            difference = paired["candidate"] - paired["reference"]
            margin = float(difference.std(ddof=1) / math.sqrt(len(difference)))
        else:
            margin = 0.0
        candidate_runout = float(scale["runout_survival_NLL"])
        rows.append({
            "candidate_index": candidate_index, **candidate,
            "inner_SB_RMSE": float(exact_score),
            "inner_runout_hinge_RMSE": censored_hinge_rmse(oof),
            "inner_joint_censored_NLL": float(scale["joint_censored_NLL"]),
            "inner_runout_survival_NLL": candidate_runout,
            "reference_WB_CD_runout_NLL": reference_runout,
            "runout_noninferiority_margin": margin,
            "censor_feasible": bool(candidate_runout <= reference_runout + margin + 1e-12),
            "within_RMSE_tolerance": False,
            "inner_LOTO_RMSE": np.nan,
        })
        stored[candidate_index] = oof
    candidates = pd.DataFrame(rows)
    best_rmse = float(candidates["inner_SB_RMSE"].min())
    candidates["within_RMSE_tolerance"] = (
        candidates["inner_SB_RMSE"] <= best_rmse + NEAR_TIE_RMSE
    )
    shortlist = candidates.loc[candidates["within_RMSE_tolerance"]].copy()
    feasible = shortlist.loc[shortlist["censor_feasible"]].copy()
    constraint_relaxed = feasible.empty
    pool = shortlist if constraint_relaxed else feasible
    pool = pool.sort_values(["inner_SB_RMSE", "inner_runout_survival_NLL"]).head(LOTO_SHORTLIST)
    for candidate_index in pool["candidate_index"].astype(int):
        candidate = dict(HYBRID_CANDIDATES[candidate_index])
        candidates.loc[candidates["candidate_index"].eq(candidate_index), "inner_LOTO_RMSE"] = (
            _inner_loto_score(train, candidate, seed + 700000 + 10000 * candidate_index, trees)
        )
    ranked = candidates.loc[candidates["candidate_index"].isin(pool["candidate_index"])].copy()
    ranked = ranked.sort_values(
        ["inner_LOTO_RMSE", "inner_SB_RMSE", "inner_runout_survival_NLL", "candidate_index"],
        na_position="last",
    )
    best = ranked.iloc[0]
    candidates["censor_constraint_relaxed"] = constraint_relaxed
    candidates = candidates.sort_values(
        ["within_RMSE_tolerance", "censor_feasible", "inner_SB_RMSE", "inner_LOTO_RMSE"],
        ascending=[False, False, True, True], na_position="last",
    ).reset_index(drop=True)
    chosen = {
        key: (int(best[key]) if key in {"candidate_index", "min_samples_leaf"}
              else float(best[key]) if key not in {"residual_mode"} else str(best[key]))
        for key in ["candidate_index", "gamma", "min_samples_leaf", "max_features",
                    "eta", "residual_mode", "ridge_alpha", "temperature_scale",
                    "inner_SB_RMSE", "inner_runout_hinge_RMSE",
                    "inner_joint_censored_NLL", "inner_runout_survival_NLL",
                    "reference_WB_CD_runout_NLL", "runout_noninferiority_margin",
                    "inner_LOTO_RMSE"]
    }
    chosen["censor_feasible"] = bool(best["censor_feasible"])
    chosen["censor_constraint_relaxed"] = bool(constraint_relaxed)
    chosen["RMSE_tolerance"] = NEAR_TIE_RMSE
    return chosen, stored[int(best["candidate_index"])], candidates
