def log_ndtr(x):
    return scipy_special.log_ndtr(x)


def ndtri(x):
    return scipy_special.ndtri(x)


import math
import numpy as np
import scipy.special as scipy_special
from scipy.optimize import minimize_scalar

REVIEW_FAMILIES = ("lognormal", "loglogistic", "weibull")


def review_row_losses(y, mu, scale, exact, family="lognormal"):
    y, mu, scale = np.broadcast_arrays(np.asarray(y, float),
                                      np.asarray(mu, float), np.asarray(scale, float))
    exact = np.asarray(exact, bool)
    if exact.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(mu).all():
        raise ValueError("Mismatched or nonfinite observations/locations")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Scale must be finite and positive")
    z = (y - mu) / scale
    if family == "lognormal":
        density = np.log(scale) + .5 * math.log(2 * math.pi) + .5 * z**2
        survival = -scipy_special.log_ndtr(-z)
    elif family == "loglogistic":
        density = np.log(scale) + np.logaddexp(0, z) + np.logaddexp(0, -z)
        survival = np.logaddexp(0, z)
    elif family == "weibull":
        
        with np.errstate(over="ignore"):
            survival = math.log(2) * np.exp(z)
        density = np.log(scale) - math.log(math.log(2)) - z + survival
    else:
        raise ValueError(f"Unknown family: {family}")
    return np.where(exact, density, survival)


def review_quantile(mu, scale, p, family="lognormal"):
    p = np.asarray(p, float)
    scale = np.asarray(scale, float)
    if np.any((p <= 0) | (p >= 1)) or np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("Require 0<p<1 and positive finite scale")
    if family == "lognormal":
        z = scipy_special.ndtri(p)
    elif family == "loglogistic":
        z = np.log(p) - np.log1p(-p)
    elif family == "weibull":
        z = np.log(-np.log1p(-p) / math.log(2))
    else:
        raise ValueError(f"Unknown family: {family}")
    return np.asarray(mu) + scale * z


def review_calibrate_scale(y, mu, exact, family="lognormal", bounds=(.08, 4.)):


    exact = np.asarray(exact, bool)
    if not exact.any():
        raise ValueError("Calibration requires at least one exact observation")
    result = minimize_scalar(lambda s: np.mean(review_row_losses(y, mu, s, exact, family)),
                             method="bounded", bounds=bounds,
                             options={"xatol": 1e-10})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Scale optimization failed: {result.message}")
    return {"scale": float(result.x), "calibration_nll": float(result.fun),
            "scale_at_bound": bool(min(result.x-bounds[0], bounds[1]-result.x) < 1e-4)}


@dataclass(frozen=True)
class review_CompetingDamageConfig:


    intercept_half_width: float = 0.75
    walker_half_width: float = 3.0
    walker_min: float = 0.05
    walker_max: float = 20.0
    environment_ratio_min: float = -8.0
    environment_ratio_max: float = 4.0
    environment_stress_min: float = 0.0
    environment_stress_max: float = 10.0
    arrhenius_min: float = 0.0
    arrhenius_max: float = 10.0
    ridge_strength: float = 0.01
    environment_strength: float = 0.0
    
    likelihood_family: str = "lognormal"
    runout_loss_power: float = 1.0


REVIEW_DEFAULT_COMPETING_CONFIG = review_CompetingDamageConfig()


def review_apply_physics_drivers(df: pd.DataFrame, water_factor: float | None = None,
                          arrhenius_reference_c: float = ARRHENIUS_REFERENCE_C,
                          arrhenius_scale: float = ARRHENIUS_SCALE,
                          air_factor: float = 1.0, inert_factor: float = 0.0,
                          frequency_exponent: float = 1.0,
                          temperature_enabled: bool = True,
                          stress_enabled: bool = True,
                          environment_enabled: bool = True,
                          water_log10_upper: float | None = None) -> pd.DataFrame:


    if (water_factor is not None and (not np.isfinite(water_factor) or water_factor <= 0)) or arrhenius_scale <= 0:
        raise ValueError("water_factor and arrhenius_scale must be positive")
    if not np.isfinite([1.0 if water_factor is None else water_factor, arrhenius_scale, air_factor, inert_factor, frequency_exponent]).all() or min(air_factor, inert_factor, frequency_exponent) < 0:
        raise ValueError("Environmental assumptions must be finite and nonnegative")
    result = df.copy()
    env = result["env_class"].astype(str).str.lower()
    inert = env.str.contains("vacuum|inert|argon", regex=True, na=False)
    water = env.str.contains("water|steam", regex=True, na=False)
    result["environment_exposure"] = np.where(
        inert.to_numpy(bool), float(inert_factor),
        np.where(water.to_numpy(bool), 1.0 if water_factor is None else float(water_factor), float(air_factor)),
    )
    if not environment_enabled:
        result["environment_exposure"] = 0.0
    
    result["environment_frequency_drive"] = frequency_exponent * result["logf"]
    result["environment_stress_multiplier"] = float(stress_enabled)
    result["fit_water_multiplier"] = water_factor is None and environment_enabled
    if water_log10_upper is not None and (not np.isfinite(water_log10_upper) or water_log10_upper <= 0):
        raise ValueError("water_log10_upper must be positive; diagnostic cap only")
    result["water_log10_upper"] = np.nan if water_log10_upper is None else float(water_log10_upper)
    temperature_k = result["T_C"].to_numpy(float) + 273.15
    reference_k = float(arrhenius_reference_c) + 273.15
    if reference_k <= 0 or np.any(temperature_k <= 0):
        raise ValueError("Arrhenius temperatures must be above absolute zero")
    result["arrhenius_temperature_drive"] = float(arrhenius_scale) * (
        1.0 / reference_k - 1.0 / temperature_k
    )
    if not temperature_enabled:
        result["arrhenius_temperature_drive"] = 0.0
    return result


def review_joint_aft_nll(frame: pd.DataFrame, mu: np.ndarray, sigma: float,
                  weights: np.ndarray | None = None, family: str = "lognormal",
                  runout_loss_power: float = 1.0) -> dict[str, float]:

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
    
    losses[runout_mask] = -scipy_special.log_ndtr(-z[runout_mask])
    if family != "lognormal":
        losses = review_row_losses(y, location, sigma, exact_mask, family)
    if not np.isfinite(runout_loss_power) or runout_loss_power < 0:
        raise ValueError("runout_loss_power must be finite and nonnegative")
    
    losses[runout_mask] *= runout_loss_power
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


def review_fit_linear_physical(frame: pd.DataFrame, spec: PhysicalSpec,
                         gamma: float, family: str = "lognormal",
                         runout_loss_power: float = 1.0) -> np.ndarray:

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
        nll = review_joint_aft_nll(frame, x @ coef, sigma, family=family,
                          runout_loss_power=runout_loss_power)["joint_censored_NLL"]
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


def review_predict_competing_damage(df: pd.DataFrame, coef: np.ndarray,
                              gamma: float) -> np.ndarray:


    if len(coef) not in (5,6):
        raise ValueError("Competing-damage parameters: five fixed-weight or six learned-weight coefficients")
    intercept, b, log_k, m, q_t = np.asarray(coef, float)[:5]
    log_sw = walker_log_stress(df, gamma)
    log_d_mech = -intercept + b * log_sw

    exposure = df["environment_exposure"].to_numpy(float)
    active = exposure > 0.0
    log_d_env = np.full(len(df), -np.inf, dtype=float)
    log_d_env[active] = (
        log_d_mech[active]
        + log_k
        + np.log10(exposure[active]) + water_log_adjustment(df,coef)[active]
        + (q_t * df.loc[active, "arrhenius_temperature_drive"].to_numpy(float)
           / math.log(10.0))
        + m * log_sw[active] * (df.loc[active, "environment_stress_multiplier"].to_numpy(float)
                               if "environment_stress_multiplier" in df else 1.0)
        - df.loc[active, "environment_frequency_drive" if "environment_frequency_drive" in df else "logf"].to_numpy(float)
    )
    
    maximum = np.maximum(log_d_mech, log_d_env)
    log_total_damage = maximum + np.log10(
        np.power(10.0, log_d_mech - maximum)
        + np.power(10.0, log_d_env - maximum)
    )
    return -log_total_damage


def review_environmental_decrement(frame, coefficients, gamma):
    _, _, log_k, stress_exponent, temperature_exponent = np.asarray(coefficients, float)[:5]
    exposure = frame["environment_exposure"].to_numpy(float)
    active = exposure > 0.0
    decrement = np.zeros(len(frame), dtype=float)
    log_sw = walker_log_stress(frame, gamma)
    log_ratio = (
        log_k + np.log10(exposure[active]) + water_log_adjustment(frame,coefficients)[active]
        + temperature_exponent * frame.loc[active, "arrhenius_temperature_drive"].to_numpy(float) / math.log(10.0)
        + stress_exponent * log_sw[active] * (frame.loc[active, "environment_stress_multiplier"].to_numpy(float)
                                             if "environment_stress_multiplier" in frame else 1.0)
        - frame.loc[active, "environment_frequency_drive" if "environment_frequency_drive" in frame else "logf"].to_numpy(float)
    )
    decrement[active] = np.logaddexp(0.0, math.log(10.0) * log_ratio) / math.log(10.0)
    return decrement


def review_fit_physical_uncached(train: pd.DataFrame, spec: PhysicalSpec, gamma: float = GAMMA,
                 competing_config: review_CompetingDamageConfig = REVIEW_DEFAULT_COMPETING_CONFIG,
                 ) -> np.ndarray:
    require_exact_training(train, f"fit_physical[{spec.key}]")
    exact = train.loc[train["is_exact"]].reset_index(drop=True)
    if spec.mode == "linear":
        return review_fit_linear_physical(train, spec, gamma, competing_config.likelihood_family,
                                    competing_config.runout_loss_power)
    if spec.mode != "competing_damage":
        raise ValueError(f"Unknown physical mode: {spec.mode}")

    classic = review_fit_linear_physical(train, CLASSIC_WB_SPEC, gamma,
        competing_config.likelihood_family,competing_config.runout_loss_power)
    return water_fit_competing(train,gamma,competing_config,classic,
        review_predict_competing_damage,review_joint_aft_nll,review_environmental_decrement,
        likelihood_options=dict(family=competing_config.likelihood_family,
                                runout_loss_power=competing_config.runout_loss_power))


review_fit_physical = cache_physical_fit(
    review_fit_physical_uncached, require_exact_training, lambda: DAMAGE_CAP
)


def review_predict_physical(df: pd.DataFrame, spec: PhysicalSpec, coef: np.ndarray,
                     gamma: float = GAMMA) -> np.ndarray:
    if spec.mode == "competing_damage":
        return review_predict_competing_damage(df, coef, gamma)
    return physical_design(df, spec, gamma) @ coef


def review_fit_hybrid_policy(
    train, valid, spec, seed, trees, gamma=GAMMA, min_samples_leaf=3,
    max_features=.85, eta=.75, residual_mode="ET_residual", ridge_alpha=1.0,
    temperature_scale=250.0, temperature_policy="soft250", strength=0.0,
    family=None, competing_config=REVIEW_DEFAULT_COMPETING_CONFIG,
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
    coef = review_fit_physical(train, spec, gamma, config)
    exact = train.loc[train.is_exact].reset_index(drop=True)
    full = pd.concat([exact, valid], ignore_index=True)
    base = review_predict_physical(full, spec, coef, gamma)
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


import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import platform
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits


REVIEW_ROOT = Path(__file__).resolve().parent
REVIEW_ID = ['row_id', 'source_id', 'campaign_id', 'record_equivalence_id', 'logN', 'is_exact', 'is_runout']
REVIEW_JOINT = ['architecture', 'temp_bin', 'env_class']
REVIEW_NUMERIC = ['T_C', 'S', 'R', 'logf']


def review_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review_save(frame, out, name):
    frame.to_csv(Path(out) / f'{name}.csv', index=False, encoding='utf-8-sig')


def review_require_disjoint(train, test):
    for key in ('row_id', 'campaign_id', 'record_equivalence_id'):
        if set(train[key]) & set(test[key]):
            raise AssertionError(f'Train/test overlap: {key}')


def review_counts(df, keys):
    result = df.groupby(keys, observed=True, dropna=False).agg(
        n_records=('row_id', 'size'), n_exact=('is_exact', 'sum'),
        n_runout=('is_runout', 'sum'), n_campaigns=('campaign_id', 'nunique'),
        n_sources=('source_id', 'nunique')).reset_index()
    for event in ('exact', 'runout'):
        sub = df.loc[df[f'is_{event}']].groupby(keys, observed=True).campaign_id.nunique()
        result = result.merge(sub.rename(f'n_{event}_campaigns'), on=keys, how='left')
        result[f'n_{event}_campaigns'] = result[f'n_{event}_campaigns'].fillna(0).astype(int)
    result['runout_fraction'] = result.n_runout / result.n_records
    result['single_campaign'] = result.n_campaigns.eq(1)
    result['few_records_lt10'] = result.n_records.lt(10)
    return result


def review_database_audit(df, out):
    for key in ['architecture', 'architecture_detail', 'env_class', 'T_C', 'temp_bin',
                'campaign_id', 'source_id']:
        review_save(review_counts(df, [key]), out, f'data_{key}')
    joint = review_counts(df, REVIEW_JOINT)
    
    grid = pd.DataFrame(itertools.product(*(sorted(df[k].unique()) for k in REVIEW_JOINT)), columns=REVIEW_JOINT)
    grid = grid.merge(joint, on=REVIEW_JOINT, how='left')
    for key in [c for c in grid if c.startswith('n_')]:
        grid[key] = grid[key].fillna(0).astype(int)
    grid['observed_cell'] = grid.n_records.gt(0)
    grid['grid_definition'] = 'observed marginal levels; empty cells not necessarily physically feasible'
    review_save(grid, out, 'data_joint_grid')
    review_save(review_counts(df, REVIEW_JOINT + ['T_C']), out, 'data_joint_exact_temperature')
    review_save(review_counts(df, ['architecture_detail','T_C','env_class']),out,'data_joint_detailed')
    pairs = []
    for a, b in itertools.combinations(REVIEW_JOINT, 2):
        table = review_counts(df, [a, b]).rename(columns={a: 'level_a', b: 'level_b'})
        table.insert(0, 'factor_b', b); table.insert(0, 'factor_a', a)
        pairs.append(table)
    review_save(pd.concat(pairs, ignore_index=True), out, 'data_factor_overlap')
    ranges = df.groupby(REVIEW_JOINT, observed=True).agg(
        stress_min=('S', 'min'), stress_max=('S', 'max'), frequency_min=('frequency_Hz', 'min'),
        frequency_max=('frequency_Hz', 'max'), R_min=('R', 'min'), R_max=('R', 'max')).reset_index()
    review_save(ranges, out, 'data_loading_ranges')
    run = df.loc[df.is_runout].copy()
    review_save(review_counts(run, ['Nf_cycles']), out, 'runout_stopping_limits')
    protocol = run[['row_id', 'source_id', 'campaign_id', 'Nf_cycles', 'env_class', 'T_C']].copy()
    protocol['documented_stop_type'] = 'unreviewed'
    protocol['source_page_or_figure'] = ''
    protocol['verbatim_stop_rule'] = ''
    protocol['reviewer_note'] = 'Round cycle counts alone do not establish administrative censoring'
    review_environment_overlap(df,out)
    
    if not (Path(out) / 'censor_protocol_TEMPLATE.csv').exists():
        review_save(protocol, out, 'censor_protocol_TEMPLATE')
    return joint


def review_primary_scores(frame):

    if frame.row_id.duplicated().any():
        raise AssertionError('Duplicate prediction row in a scoring population')
    exact = frame.loc[frame.is_exact.astype(bool)].copy()
    if exact.empty:
        point = dict(SB_RMSE=np.nan, SB_MAE=np.nan, SB_R2=np.nan, SB_F5=np.nan)
    else:
        w = analysis_weights(exact); w /= w.sum()
        error = exact.pred_logN.to_numpy(float) - exact.logN.to_numpy(float)
        mean = np.dot(w, exact.logN)
        variance = np.dot(w, (exact.logN - mean)**2)
        mse = np.dot(w, error**2)
        point = dict(SB_RMSE=float(np.sqrt(mse)), SB_MAE=float(np.dot(w, abs(error))),
                     SB_R2=float(1-mse/variance) if variance > 1e-15 else np.nan,
                     SB_F5=float(np.dot(w, abs(error) <= np.log10(5))),
                     SB_bias=float(np.dot(w, error)),
                     SB_overpredict_factor10=float(np.dot(w, error > 1)),
                     geometric_abs_error_factor=float(10**np.dot(w, abs(error))))
        for name, lo, hi in [('empirical90', 'lower90', 'upper90'),
                              ('parametric90', 'prob_lower90', 'prob_upper90')]:
            if {lo, hi}.issubset(exact) and exact[[lo, hi]].notna().all().all():
                point[f'{name}_exact_coverage'] = float(np.dot(w, (exact.logN >= exact[lo]) & (exact.logN <= exact[hi])))
                point[f'{name}_exact_width'] = float(np.dot(w, exact[hi]-exact[lo]))
    w = analysis_weights(frame); w /= w.sum()
    if 'row_nll' in frame:
        losses = frame.row_nll.to_numpy(float)
        if not np.isfinite(losses).all():
            raise FloatingPointError('Nonfinite probability loss')
        event = frame.is_exact.to_numpy(bool)
        point.update(joint_NLL=float(np.dot(w, losses)),
                     exact_component_NLL=float(np.dot(w, np.where(event, losses, 0))),
                     runout_component_NLL=float(np.dot(w, np.where(event, 0, losses))))
    return dict(n_records=len(frame), n_exact=len(exact), n_runout=len(frame)-len(exact),
                n_campaigns=frame.campaign_id.nunique(),
                n_exact_campaigns=exact.campaign_id.nunique(), **point)


def review_paired_ci(a, b, repetitions, seed=SEED):


    a = a.sort_values('row_id').reset_index(drop=True)
    b = b.sort_values('row_id').reset_index(drop=True)
    if not a[REVIEW_ID].equals(b[REVIEW_ID]):
        raise AssertionError('Paired predictions do not have identical observations')
    groups = list(a.groupby('campaign_id').indices.values())
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ['SB_RMSE', 'SB_MAE', 'SB_R2', 'joint_NLL']}
    
    def stats(x):
        values=[]
        for ind in groups:
            g=x.iloc[ind]; e=g.loc[g.is_exact]; err=e.pred_logN-e.logN
            values.append([len(e)>0, np.mean(err**2) if len(e) else 0,
                np.mean(abs(err)) if len(e) else 0, e.logN.mean() if len(e) else 0,
                np.mean(e.logN**2) if len(e) else 0, g.row_nll.mean()])
        return np.asarray(values, float)
    aa, bb = stats(a), stats(b)
    def score(arr, ind):
        z=arr[ind]; exact=z[z[:,0]>0]; mse=exact[:,1].mean()
        var=exact[:,4].mean()-exact[:,3].mean()**2
        return [np.sqrt(mse), exact[:,2].mean(), 1-mse/var if var>1e-15 else np.nan, z[:,5].mean()]
    for _ in range(repetitions):
        ind=rng.integers(0,len(groups),len(groups))
        delta=np.asarray(score(aa,ind))-np.asarray(score(bb,ind))
        for k,v in zip(draws,delta): draws[k].append(v)
    sa,sb=review_primary_scores(a),review_primary_scores(b)
    return pd.DataFrame([dict(metric=k, difference=sa[k]-sb[k],
        CI95_low=float(np.nanquantile(v,.025)), CI95_high=float(np.nanquantile(v,.975)),
        finite_bootstrap_draws=int(np.isfinite(v).sum()), bootstrap_draws=repetitions,
        direction='variant minus reference; negative favors variant for errors/NLL',
        uncertainty='campaign bootstrap of fixed OOF predictions; no refitting') for k,v in draws.items()])


def review_selection_audit(tables, df, out):
    selected = tables['Nested_Selection']
    selected = selected.loc[selected.family.notna()].copy()
    review_save(selected, out, 'selection_by_fold')
    candidates = tables['Nested_Candidates']
    frequencies = []
    hyper = []
    development = []
    outer,_=grouped_folds(df,5,SEED)
    for model, c in candidates.groupby('model'):
        s=selected.loc[selected.model.eq(model)]
        if s.empty: continue
        n_folds=s.outer_fold.nunique()
        for fold,(tr,te) in enumerate(outer,1):
            train=df.iloc[tr]
            development.append(dict(model=model,outer_fold=fold,n_training_records=len(train),
                n_training_exact=int(train.is_exact.sum()),n_training_runout=int(train.is_runout.sum()),
                n_training_campaigns=train.campaign_id.nunique(),
                n_training_exact_campaigns=train.loc[train.is_exact,'campaign_id'].nunique(),
                n_candidate_configurations=c.loc[c.outer_fold.eq(fold),'candidate_index'].nunique(),
                interpretation='configuration count is search multiplicity, not an independent sample count'))
        for cid,g in c.groupby('candidate_index'):
            f=s.candidate_index.eq(cid).sum()
            frequencies.append(dict(model=model,candidate_index=cid,family=g.family.iloc[0],
                selected_count=int(f),n_outer_folds=n_folds,selected_fraction=f/n_folds,
                n_candidate_configurations=c.candidate_index.nunique(),
                n_total_campaigns=df.campaign_id.nunique()))
        for parameter in ['family','gamma','strength','eta','min_samples_leaf','max_features',
                          'ridge_alpha','kernel_length_scale','temperature_policy']:
            if parameter not in s: continue
            active=s
            if parameter in ['min_samples_leaf','max_features']:
                active=s.loc[s.family.str.contains('ET|GBDT',regex=True)]
            if parameter in ['ridge_alpha','kernel_length_scale']:
                active=s.loc[s.family.str.contains('Matern',regex=False)]
            frequencies_for_parameter=active[parameter].dropna().value_counts()
            if parameter=='family':
                frequencies_for_parameter=frequencies_for_parameter.reindex(sorted(c.family.dropna().unique()),fill_value=0)
            for value,n in frequencies_for_parameter.items():
                searched=(c[parameter].nunique()>1 if parameter=='family'
                          else c.loc[c.family.isin(active.family),parameter].nunique()>1)
                hyper.append(dict(model=model,parameter=parameter,value=value,count=n,
                    applicable_folds=len(active),total_folds=n_folds,
                    role='searched' if searched else 'fixed_within_applicable_families'))
    review_save(pd.DataFrame(frequencies),out,'candidate_selection_frequency')
    review_save(pd.DataFrame(hyper),out,'hyperparameter_frequency')
    review_save(pd.DataFrame(development),out,'selection_development_support')
    replay=selection_rule_sensitivity(candidates,tables['Nested_Selection'])
    replay['outer_performance_recomputed']=False
    review_save(replay,out,'selection_rule_replay')


def review_support_audit(df, pred, out):

    representative=pred.loc[pred.model.eq('WB-PIML')]
    rows=[]; splits=[]
    for (scenario,fold),g in representative.groupby(['scenario','fold']):
        test=df.loc[df.row_id.isin(g.row_id)].copy()
        if scenario == 'Primary':
            train=df.loc[~df.campaign_id.isin(test.campaign_id)].copy()
            review_require_disjoint(train,test)
        else:
            column=str(g.holdout_column.iloc[0]); value=str(g.holdout_group.iloc[0])
            train=df.loc[~df[column].astype(str).eq(value)].copy()
            expected=set(df.loc[df[column].astype(str).eq(value),'row_id'])
            if expected!=set(test.row_id): raise AssertionError('Holdout definition mismatch')
        overlap=set(train.campaign_id)&set(test.campaign_id)
        splits.append(dict(scenario=scenario,fold=fold,holdout_group=g.holdout_group.iloc[0],
            n_train=len(train),n_test=len(test),n_train_campaigns=train.campaign_id.nunique(),
            n_test_campaigns=test.campaign_id.nunique(),n_overlapping_campaigns=len(overlap),
            campaign_overlap=bool(overlap),campaigns_to_purge=' | '.join(sorted(overlap)),
            n_train_after_campaign_purge=int((~train.campaign_id.isin(overlap)).sum())))
        sd=train[REVIEW_NUMERIC].std(ddof=0).replace(0,1).to_numpy(float)
        for _,r in test.iterrows():
            same=train.loc[(train[REVIEW_JOINT]==r[REVIEW_JOINT]).all(axis=1)]
            unseen=[k for k in ['architecture','architecture_detail','env_class','temp_bin'] if r[k] not in set(train[k])]
            outside=[k for k in REVIEW_NUMERIC if r[k]<train[k].min() or r[k]>train[k].max()]
            dist=np.sqrt(np.sum(((train[REVIEW_NUMERIC].to_numpy(float)-r[REVIEW_NUMERIC].to_numpy(float))/sd)**2,axis=1))
            
            label=('unseen_category' if unseen else 'outside_numeric_range' if outside
                   else 'unseen_joint_cell' if same.empty else 'within_observed_marginal_and_cell_support')
            rows.append(dict(scenario=scenario,fold=fold,row_id=r.row_id,
                train_joint_n_records=len(same),train_joint_n_campaigns=same.campaign_id.nunique(),
                train_joint_n_exact=int(same.is_exact.sum()),
                train_joint_n_exact_campaigns=same.loc[same.is_exact,'campaign_id'].nunique(),
                unseen_categories='|'.join(unseen),
                outside_numeric_ranges='|'.join(outside),nearest_numeric_standardized_distance=dist.min(),
                support_class=label,test_campaign_seen=r.campaign_id in overlap,
                novel_material_system_verified=False))
    support=pd.DataFrame(rows)
    review_save(support,out,'prediction_support_rows');review_save(pd.DataFrame(splits),out,'holdout_overlap')
    merged=pred.merge(support,on=['scenario','fold','row_id'],validate='many_to_one')
    scores=[]
    for (scenario,model,label),g in merged.groupby(['scenario','model','support_class']):
        scores.append(dict(scenario=scenario,model=model,support_class=label,**review_primary_scores(g)))
    review_save(pd.DataFrame(scores),out,'support_stratum_performance')
    merged['train_cell_campaign_support']=pd.cut(merged.train_joint_n_campaigns,
        bins=[-1,0,1,3,np.inf],labels=['0','1','2-3','4+'])
    sparse=[]
    for (scenario,model,level),g in merged.groupby(['scenario','model','train_cell_campaign_support'],observed=True):
        sparse.append(dict(scenario=scenario,model=model,training_cell_campaigns=level,**review_primary_scores(g)))
    review_save(pd.DataFrame(sparse),out,'sparse_cell_performance')
    group=[]
    for (scenario,holdout,model),g in pred.groupby(['scenario','holdout_group','model'],dropna=False):
        group.append(dict(scenario=scenario,holdout_group=holdout,model=model,**review_primary_scores(g)))
    review_save(pd.DataFrame(group),out,'holdout_performance_campaign_weighted')


def review_fixed_location_distribution_audit(df, pred, calibration, out):
    pieces=[]; logs=[]
    models=['WB-PIML','WB-CD','ML-Ens','WB-Residual','Basquin-Residual']
    primary=pred.loc[pred.scenario.eq('Primary') & pred.model.isin(models)]
    for (model,fold),test in primary.groupby(['model','fold']):
        oof=calibration.loc[calibration.scenario.eq('Primary') & calibration.model.eq(model)
                           & calibration.fold.eq(fold) & calibration.record_type.eq('calibration_oof')].copy()
        train=df.loc[~df.row_id.isin(test.row_id)]
        review_require_disjoint(train,test)
        if not oof.row_id.is_unique or set(oof.row_id)!=set(train.row_id):
            raise AssertionError('Saved development OOF coverage mismatch')
        if oof.outer_test_used_for_calibration.fillna(False).astype(bool).any():
            raise AssertionError('Baseline reports test calibration')
        for _,g in oof.groupby('inner_fold'):
            review_require_disjoint(train.loc[~train.row_id.isin(g.row_id)],g)
        for family in REVIEW_FAMILIES:
            fit=review_calibrate_scale(oof.logN,oof.mu,oof.is_exact,family)
            part=test[REVIEW_ID+['pred_logN']].copy()
            part['fold']=fold;part['model']=model;part['distribution']=family
            part['row_nll']=review_row_losses(part.logN,part.pred_logN,fit['scale'],part.is_exact,family)
            part['prob_lower90']=review_quantile(part.pred_logN,fit['scale'],.05,family)
            part['prob_upper90']=review_quantile(part.pred_logN,fit['scale'],.95,family)
            part['scale']=fit['scale'];pieces.append(part)
            logs.append(dict(model=model,fold=fold,distribution=family,**fit,
                outer_test_used=False,locations_refitted=False,
                development_hash=probability_row_id_hash(oof.row_id)))
    predictions=pd.concat(pieces,ignore_index=True)
    metrics=[dict(model=m,distribution=f,analysis='fixed_location_scale_recalibration',**review_primary_scores(g))
             for (m,f),g in predictions.groupby(['model','distribution'])]
    review_save(predictions,out,'distribution_fixed_location_predictions')
    review_save(pd.DataFrame(metrics),out,'distribution_fixed_location_metrics')
    review_save(pd.DataFrame(logs),out,'distribution_calibration_audit')


def review_variants():
    v=[dict(name='selected_pipeline')]
    v += [dict(name=f'water_{x:g}',drivers=dict(water_factor=x)) for x in [0.5,1.,1.25,1.5,1.75,2.,3.]]
    v += [dict(name=f'water_cap_{10**x:g}',drivers=dict(water_log10_upper=x)) for x in (1.,2.)]
    v += [dict(name=f'inert_{x:g}',drivers=dict(inert_factor=x)) for x in [.01,.05]]
    v += [dict(name=f'frequency_power_{x:g}',drivers=dict(frequency_exponent=x)) for x in [0.,.5,1.5]]
    v += [dict(name='no_temperature_in_damage',drivers=dict(temperature_enabled=False)),
          dict(name='no_stress_in_damage',drivers=dict(stress_enabled=False)),
          dict(name='no_environment_damage',drivers=dict(environment_enabled=False)),
          dict(name='no_residual',settings=dict(eta=0.))]
    v += [dict(name=f'environment_penalty_{x:g}',settings=dict(strength=x)) for x in [0.,.1,1.]]
    v += [dict(name=f'runout_loss_power_{x:g}',runout_loss_power=x) for x in [0.,.5,2.]]
    v += [dict(name=f'refit_{f}',distribution=f) for f in ['loglogistic','weibull']]
    v += [dict(name=f'WB_CD_{f}',distribution=f,physical_only=True) for f in REVIEW_FAMILIES]
    v += [dict(name=f'joint_water_{w:g}_frequency_{f:g}',drivers=dict(water_factor=w,frequency_exponent=f))
          for w in (.5,2.) for f in (.5,1.5)]
    return v


def review_refit_one(df, selected, variant, fold, trees, checkpoint, run_hash):

    cache=Path(checkpoint)/f'{variant["name"]}_fold{fold}.json'
    if cache.exists():
        payload=json.loads(cache.read_text(encoding='utf-8'))
        if payload['run_hash']==run_hash:
            return payload
    splits,_=grouped_folds(df,5,SEED)
    tr,te=splits[fold-1]
    train=review_apply_physics_drivers(df.iloc[tr].reset_index(drop=True),**variant.get('drivers',{}))
    test=review_apply_physics_drivers(df.iloc[te].reset_index(drop=True),**variant.get('drivers',{}))
    review_require_disjoint(train,test)
    family=variant.get('distribution','lognormal')
    base=selected.loc[fold]
    settings={k:base[k] for k in ['family','gamma','strength','eta','min_samples_leaf','max_features',
                                 'ridge_alpha','kernel_length_scale','temperature_policy']}
    settings['min_samples_leaf']=int(settings['min_samples_leaf'])
    settings.update(variant.get('settings',{}))
    config=replace(REVIEW_DEFAULT_COMPETING_CONFIG,likelihood_family=family,
                   runout_loss_power=variant.get('runout_loss_power',1.))
    seed=SEED+10000*fold
    def fit(a,b,fit_seed,capture=False):
        if variant.get('physical_only'):
            coef=review_fit_physical(a,COMPETING_WB_SPEC,GAMMA,config)
            return review_predict_physical(b,COMPETING_WB_SPEC,coef,GAMMA),coef,{}
        return review_fit_hybrid_policy(a,b,COMPETING_WB_SPEC,fit_seed,trees,
                                      competing_config=config,return_frozen_state=capture,**settings)
    pieces=[]; audit=[]


    calibration_seed=seed+100 if variant.get('physical_only') else seed+500
    inner,_=grouped_folds(train,4,calibration_seed)
    with threadpool_limits(limits=1):
        for k,(i,j) in enumerate(inner,1):
            a,b=train.iloc[i].reset_index(drop=True),train.iloc[j].reset_index(drop=True)
            review_require_disjoint(a,b)
            mu,inner_coef,_=fit(a,b,calibration_seed+100*k+1)
            p=b[REVIEW_ID].copy();p['mu']=mu;p['inner_fold']=k;pieces.append(p)
            audit.append(dict(inner_fold=k,fit_hash=probability_row_id_hash(a.row_id),
                validation_hash=probability_row_id_hash(b.row_id),
                n_fit=len(a),n_validation=len(b),campaign_overlap=False,
                water_fit=water_fit_metadata(a,inner_coef)))
        oof=pd.concat(pieces,ignore_index=True)
        if not oof.row_id.is_unique or set(oof.row_id)!=set(train.row_id):
            raise AssertionError('Refit OOF incomplete')
        cal=review_calibrate_scale(oof.logN,oof.mu,oof.is_exact,family)
        mu,coef,metadata=fit(train,test,seed+1,variant['name']=='selected_pipeline')
    order_audit=water_order_audit(metadata.pop('_frozen_state'),test) if '_frozen_state' in metadata else []
    p=test[REVIEW_ID].copy();p['pred_logN']=mu
    p['row_nll']=review_row_losses(p.logN,mu,cal['scale'],p.is_exact,family)
    p['prob_lower90']=review_quantile(mu,cal['scale'],.05,family)
    p['prob_upper90']=review_quantile(mu,cal['scale'],.95,family)
    p['scale']=cal['scale'];p['fold']=fold;p['variant']=variant['name'];p['distribution']=family
    if not np.isfinite(p.row_nll).all(): raise FloatingPointError('Nonfinite held-out NLL')
    
    payload=dict(run_hash=run_hash,variant=variant['name'],fold=fold,
        predictions=p.to_dict('records'),calibration=oof.to_dict('records'),
        audit=audit,scale_fit=cal,coefficients=np.asarray(coef).tolist(),settings=settings,
        water_fit=water_fit_metadata(train,coef),environment_order_audit=order_audit,
        assumption=variant,calibration_seed=calibration_seed,train_hash=probability_row_id_hash(train.row_id),
        test_hash=probability_row_id_hash(test.row_id),outer_test_used_for_calibration=False)
    cache.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    return payload


def review_refits(df,tables,out,args,fingerprint):
    selected=tables['Nested_Selection'].loc[lambda x:x.model.eq('WB-PIML-Anchor')].set_index('outer_fold')
    if not selected.index.is_unique or set(selected.index)!={1,2,3,4,5}:
        raise AssertionError('Expected five distinct selected folds')
    planned=review_variants()
    if args.variants:
        requested=set(args.variants.split(','));planned=[v for v in planned if v['name'] in requested]
        if requested!={v['name'] for v in planned}:raise ValueError('Unknown --variants name')
    checkpoint=out/'checkpoints';checkpoint.mkdir(exist_ok=True)
    review_save(pd.DataFrame([dict(variant=v['name'],settings_json=json.dumps(v,sort_keys=True),
        used_for_selection=False,scope='revised selected configurations; conditional refit, not reconstruction of submitted-model selection') for v in planned]),out,'sensitivity_plan')
    run_hash=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
    tasks=[(v,f) for v in planned for f in range(1,6)]
    results=[]
    with parallel_config(backend='loky',inner_max_num_threads=1):
        stream=Parallel(n_jobs=args.jobs,return_as='generator')(
            delayed(review_refit_one)(df,selected,v,f,args.trees,checkpoint,run_hash) for v,f in tasks)
        for n,r in enumerate(stream,1):
            results.append(r);print(f'Refit {n}/{len(tasks)}: {r["variant"]}, fold {r["fold"]}',flush=True)
    predictions=pd.concat([pd.DataFrame(r['predictions']) for r in results],ignore_index=True)
    review_save(predictions,out,'sensitivity_predictions')
    metrics=[dict(variant=v,**review_primary_scores(g)) for v,g in predictions.groupby('variant')]
    review_save(pd.DataFrame(metrics),out,'sensitivity_metrics')
    main_comparison=pd.DataFrame(metrics).loc[lambda x:x.variant.isin(['selected_pipeline','water_1','water_1.5','no_environment_damage'])].copy()
    main_comparison['comparison_scope']='fixed revised-selected hyperparameters; no model re-selection; all variant parameters refitted'
    review_save(main_comparison,out,'environment_model_comparison')
    pars=[dict(variant=r['variant'],fold=r['fold'],**physical_parameter_values(COMPETING_WB_SPEC,r['coefficients']),
               **r['scale_fit'],settings_json=json.dumps(r['settings']),
               assumption_json=json.dumps(r['assumption'],sort_keys=True)) for r in results]
    review_save(pd.DataFrame(pars),out,'sensitivity_parameters')
    baseline=predictions.loc[predictions.variant.eq('selected_pipeline')]
    if not baseline.empty:
        old=tables['Prob_Predictions'].loc[lambda x:x.scenario.eq('Primary') & x.model.eq('WB-PIML')]
        joined=baseline.merge(old[['row_id','pred_logN']],on='row_id',suffixes=('_new','_saved'),validate='one_to_one')
        diff=np.max(abs(joined.pred_logN_new-joined.pred_logN_saved))
        score_diff=abs(review_primary_scores(baseline)['joint_NLL']-review_primary_scores(old)['joint_NLL'])
        review_save(pd.DataFrame([dict(n_rows=len(joined),max_absolute_prediction_difference=diff,
             joint_NLL_absolute_difference=score_diff,tolerance=.0002,passed=diff<.0002 and score_diff<.0002,
             interpretation='numeric regression against supplied saved predictions and probability score')]),out,'baseline_prediction_regression')
        if diff>=.0002 or score_diff>=.0002:
            raise AssertionError(f'Baseline changed: prediction={diff}, jointNLL={score_diff}; investigate before reporting')
        pairs=[]
        for v,g in predictions.groupby('variant'):
            if v=='selected_pipeline':continue
            ci=review_paired_ci(g,baseline,args.bootstrap);ci.insert(0,'reference','selected_pipeline');ci.insert(0,'variant',v);pairs.append(ci)
        if pairs:review_save(pd.concat(pairs,ignore_index=True),out,'sensitivity_paired_CI')
    comparisons=[]
    for f in REVIEW_FAMILIES:
        a=predictions.loc[predictions.variant.eq('selected_pipeline' if f=='lognormal' else f'refit_{f}')]
        b=predictions.loc[predictions.variant.eq(f'WB_CD_{f}')]
        if len(a) and len(b):
            ci=review_paired_ci(a,b,args.bootstrap);ci.insert(0,'distribution',f);comparisons.append(ci)
    if comparisons:review_save(pd.concat(comparisons,ignore_index=True),out,'distribution_refit_PIML_vs_CD')
    merged=predictions.merge(df[['row_id']+REVIEW_JOINT],on='row_id',validate='many_to_one')
    subgroup=[]
    for col in REVIEW_JOINT:
        for (v,level),g in merged.groupby(['variant',col]):
            subgroup.append(dict(variant=v,factor=col,level=level,**review_primary_scores(g)))
    review_save(pd.DataFrame(subgroup),out,'sensitivity_subgroups')
    review_environment_diagnostics(df,predictions,results,out)
    water_analysis_diagnostics(df,results,out)
    return predictions


def review_report(tables,out,joint,refit_completed):
    m=tables['Primary_Campaign_Metrics'].set_index('model').loc['WB-PIML']
    p=tables['Prob_Metrics'];p=p.loc[p.scenario.eq('Primary') & p.aggregation_level.eq('overall')].set_index('model')
    delta=p.loc['WB-PIML',['SB_joint_censored_NLL','SB_exact_component_NLL','SB_runout_component_NLL']]-p.loc['WB-CD',['SB_joint_censored_NLL','SB_exact_component_NLL','SB_runout_component_NLL']]
    text=f'''# 返修代码运行结果

本次使用修订工作簿中的{EXPECTED_VERIFIED_ROWS}条记录。主分析结果来自本次指定的主结果工作簿（来源哈希见provenance.json）；新增分析不会据外层测试结果挑选新主模型。

## 可直接用于修改的证据

- 架构×温区×环境实际占用 {len(joint)} 个组合，其中 {(joint.n_campaigns==1).sum()} 个仅来自一次实验研究。记录数、失效数、runout数、独立campaign数量分别报告；没有把记录数称为统计独立的有效样本量。
- 修订主分析 R²={m.SB_R2:.3f}，95% CI [{m.SB_R2_lo:.3f}, {m.SB_R2_hi:.3f}]；RMSE={m.SB_RMSE:.3f} log10周期，MAE={m.SB_MAE:.3f}。10^MAE={10**m.SB_MAE:.2f} 是按原权重计算的绝对倍率误差的几何平均值。10^RMSE不可解释为平均倍率误差或可靠性界限。
- WB-PIML相对WB-CD的完整删失NLL差为 {delta.iloc[0]:+.6f}，失效分量 {delta.iloc[1]:+.6f}，runout分量 {delta.iloc[2]:+.6f}。三个分量采用同一完整测试集权重，前两分量相加严格等于总差。
- 选型频次表覆盖候选组合的零次选择情况，树参数只统计适用的树模型折。选型规则重放不产生新的外层性能。
- holdout_overlap.csv 显示已有因子保留验证中的campaign重叠；这些已保存预测不能冒充剔除重叠后重新训练的结果。prediction_support_rows.csv 使用训练协变量界定支持范围；“范围内”也不证明已知材料系统之外的泛化。

## 分析范围

- 重新拟合敏感性分析本次完成：{refit_completed}。具体完成的变体见 sensitivity_plan.csv；仅在使用 --refit 时生成。checkpoint保存每折训练和测试标识哈希、内层OOF预测和参数。
- 分布敏感性分为：固定位置的三分布尺度重校准，以及指定变体的全物理干线/残差重新拟合。尺度只使用开发集OOF预测校准。三种密度统一在Y=log10(N)坐标下评分，Weibull使用中位数位置参数。
- runout_loss_power是删失损失影响分析，不代表已拟合信息性删失机制。所有分布分析仍以条件独立删失为前提。全部runout保留为寿命下限观测，来源信息表为可选元数据，不影响训练。
- 主模型在训练内估计受约束水氧倍率，固定1和1.5只是条件对照，详见environment_weight_rationale.md和sensitivity_design.csv。空气为参考尺度；统一缩放E与K存在可辨识性混淆。新增4组权重×频率交叉检查不扩展主候选池；environment_response_summary.csv区分固定物理参数和重新拟合的影响。所有数值范围是分析假设，不是材料动力学常数的测量置信区间。
- 去掉环境正则惩罚与去掉环境损伤项分别命名。所有变体是返修阶段事后分析；不依最小测试RMSE改选主模型。置信区间是固定预测的campaign bootstrap，未覆盖模型重新选择的不确定性。
- 经验预测区间与参数分布区间分别命名；失效记录覆盖率不能解释为包含潜在runout寿命的无条件覆盖率。

正文引用与结论措辞仍需基于这些结果修改。DOI 10.1002/srin.202400850 属于加工优化研究，未强行变成此代码中的算法。材料系统外的真正外部验证仍需新增独立数据。
'''
    (out/'结果解读.md').write_text(text,encoding='utf-8')


def review_main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=REVIEW_ROOT/'data.xlsx')
    parser.add_argument('--baseline',type=Path,default=REVIEW_ROOT/'baseline_results.xlsx')
    parser.add_argument('--censor-evidence',type=Path)
    parser.add_argument('--out',type=Path,default=REVIEW_ROOT/'reviewer_results')
    parser.add_argument('--refit',action='store_true',help='Execute documented post-hoc conditional refits; never select a new main model')
    parser.add_argument('--variants',help='Comma-separated subset; omitted means the complete documented sensitivity plan')
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--trees',type=int,default=400)
    parser.add_argument('--bootstrap',type=int,default=5000)
    args=parser.parse_args(argv)
    if args.jobs<1 or args.bootstrap<100 or args.trees<1:parser.error('Positive jobs/trees and bootstrap>=100 required')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'run_status.json').write_text(json.dumps(dict(status='running',refits_requested=args.refit,
        variants=args.variants),indent=2),encoding='utf-8')
    started=time.perf_counter()
    review_write_design(out)
    df=load_data(args.data)
    names=['Protocol','Nested_Selection','Nested_Candidates','Prob_Predictions','Prob_Calibration',
           'Prob_Metrics','Primary_Campaign_Metrics','Paired_Bootstrap','Prob_Paired_CI',
           'Empirical_PI_Summary','External_PI_Summary','Sensitivity','External_Selection']
    print('Reading saved results and checking provenance...',flush=True)
    tables=pd.read_excel(args.baseline,sheet_name=names)
    protocol=tables['Protocol'].set_index('item').value
    single_check_primary(args.baseline)
    if str(protocol['data_sha256'])!=review_sha(args.data):raise ValueError('Data hash does not match saved results')
    fingerprint=dict(data_sha256=review_sha(args.data),baseline_sha256=review_sha(args.baseline),
        code_sha256={p.name:review_sha(p) for p in [Path(__file__)]},
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        scipy=scipy.__version__,sklearn=sklearn.__version__,trees=args.trees,
        seed=SEED,sensitivity_design_version=REVIEW_DESIGN_VERSION,
        sensitivity_plan_sha256=hashlib.sha256(json.dumps(review_design_rows(),sort_keys=True).encode('utf-8')).hexdigest(),
        strict_validation_version=STRICT_REVIEW_VERSION,
        primary_model_id=str(protocol.get('revision_model_id','unknown')),
        primary_code_provenance='current single-file source hash recorded above')
    (out/'provenance.json').write_text(json.dumps(fingerprint,indent=2),encoding='utf-8')
    primary=tables['Prob_Predictions'].loc[lambda x:x.scenario.eq('Primary')]
    regression=[]
    for model,g in primary.groupby('model'):
        computed=review_primary_scores(g)
        saved=tables['Prob_Metrics'].loc[lambda x:x.scenario.eq('Primary') & x.aggregation_level.eq('overall') & x.model.eq(model)].iloc[0]
        for key,source in [('SB_RMSE','SB_RMSE'),('SB_R2','SB_R2'),('joint_NLL','SB_joint_censored_NLL')]:
            delta=abs(computed[key]-saved[source])
            regression.append(dict(model=model,metric=key,absolute_difference=delta,passed=delta<1e-9))
            if delta>=1e-9:raise AssertionError(f'Baseline metric mismatch: {model} {key}')
    review_save(pd.DataFrame(regression),out,'baseline_metric_regression')
    external_water=tables['External_Selection'].loc[lambda x:x.model.eq('WB-PIML')].copy()
    review_save(external_water,out,'water_factor_holdout_estimation')
    joint=review_database_audit(df,out)
    review_selection_audit(tables,df,out)
    print('Scoring support regions and holdouts...',flush=True)
    review_support_audit(df,tables['Prob_Predictions'],out)
    review_fixed_location_distribution_audit(df,tables['Prob_Predictions'],tables['Prob_Calibration'],out)
    for n in ['Primary_Campaign_Metrics','Paired_Bootstrap','Prob_Paired_CI',
              'Empirical_PI_Summary','External_PI_Summary','Sensitivity','External_Selection']:
        review_save(tables[n],out,'retained_'+n)
    decomp=[]
    pm=tables['Prob_Metrics']
    for scenario,g in pm.loc[pm.aggregation_level.eq('overall')].groupby('scenario'):
        if 'WB-PIML' not in set(g.model):continue
        a=g.set_index('model').loc['WB-PIML']
        for _,b in g.loc[g.model.ne('WB-PIML')].iterrows():
            vals={k:float(a[k]-b[k]) for k in ['SB_joint_censored_NLL','SB_exact_component_NLL','SB_runout_component_NLL']}
            if not np.isclose(vals['SB_joint_censored_NLL'],vals['SB_exact_component_NLL']+vals['SB_runout_component_NLL'],atol=1e-12):raise AssertionError('NLL decomposition mismatch')
            decomp.append(dict(scenario=scenario,reference=b.model,direction='WB-PIML minus reference',**vals))
    review_save(pd.DataFrame(decomp),out,'probability_score_decomposition')
    if args.refit:review_refits(df,tables,out,args,fingerprint)
    evidence=getattr(args,'censor_evidence',None)
    if evidence is None:
        evidence=next((p for p in [Path(args.data).resolve().parent/'censor_protocol_reviewed.csv',Path(__file__).resolve().parent/'censor_protocol_reviewed.csv'] if p.is_file()),None)
    revision_censor_evidence(df,out,evidence)
    if args.refit:strict_holdout_review(df,tables,out,args)
    revision_interpretation_audit(df,tables,out)
    review_report(tables,out,joint,args.refit)
    (out/'run_status.json').write_text(json.dumps(dict(status='completed',refits_executed=args.refit,
        elapsed_seconds=time.perf_counter()-started,bootstrap=args.bootstrap,variants=args.variants),indent=2),encoding='utf-8')
    print(f'Completed in {time.perf_counter()-started:.1f}s: {out}',flush=True)

