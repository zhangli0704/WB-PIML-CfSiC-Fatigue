
# SciPy ufunc兼容适配：保持相同SciPy计算，避免以__main__.log_ndtr引用传递给worker。
# 原提交代码区不变；此适配仅改变函数传递方式，不改变公式、参数或返回值。
def log_ndtr(x):
    return scipy_special.log_ndtr(x)


def ndtri(x):
    return scipy_special.ndtri(x)


# ==================== 第二部分：补充分布与敏感性 ====================
"""Median-parameterized survival distributions for Y=log10(N).

All densities are on the same log10 coordinate. Right-censored observations
contribute survival, never density or an imputed exact failure. Weibull below
means Weibull N, hence minimum extreme-value Y, centered at its median.
"""
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
        # Overflow is an actual infinite loss, not silently clipped tail mass.
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
    """Unit-weight development-OOF calibration, with fixed location.

    No evaluation outcomes enter this fit. Calibration is always untempered,
    even when a sensitivity fit down/up-weights censored training losses.
    """
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
    """Predeclared weak-physics assumptions used by the competing-damage fit.

    The environmental life-decrement penalty is selected in inner folds.
    Other bounds and ridge settings are fixed in the primary analysis;
    their alternatives are confined to sensitivity audits.
    """

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
    # Revision sensitivities only. Defaults reproduce the primary analysis.
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
    """Return a copy with explicit environmental/Arrhenius physics drivers.

    These constants are covariate-only assumptions.  Exposing them here makes
    the primary definition and every sensitivity variant auditable without
    changing labels, weights, folds, or validation responses.
    """
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
    # Keep raw frequency/stress available to the residual learner unchanged.
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
    """AFT loss on log10 life; primary defaults remain lognormal/unit power."""
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
    # S(t|mu,sigma) = Phi((mu-t)/sigma); log_ndtr is stable in the tail.
    losses[runout_mask] = -scipy_special.log_ndtr(-z[runout_mask])
    if family != "lognormal":
        losses = review_row_losses(y, location, sigma, exact_mask, family)
    if not np.isfinite(runout_loss_power) or runout_loss_power < 0:
        raise ValueError("runout_loss_power must be finite and nonnegative")
    # A loss-power sensitivity, NOT a fitted informative-censoring mechanism.
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
        # Retry the same objective and bounds with Powell after L-BFGS-B failure.
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
    """Predict log10 life from competing per-cycle damage rates.

    ``coef = [a, b, log10_K_env, m, q_T, beta_water]`` (beta omitted for fixed-weight controls).  The two damage
    channels are strictly non-negative.  Positive ``b`` and ``m`` enforce a
    non-increasing stress-life relation, positive ``q_T`` increases the
    environmental damage rate with temperature, and ``1/f`` implements
    exposure time per cycle.
    """
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
    # Stable base-10 log-sum-exp: log10(10**x + 10**y).
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
    coef = review_fit_physical(train, spec, gamma, config)
    exact = train.loc[train.is_exact].reset_index(drop=True)
    full = pd.concat([exact, valid], ignore_index=True)
    base = review_predict_physical(full, spec, coef, gamma)
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


# ==================== 第三部分：审稿要求的统计与对照 ====================
"""Reproducible IJF reviewer analyses. CSV outputs are directly readable in Excel.

Standalone entry: python WB_PIML_revision_single.py (full fresh analysis)
Default (without --refit) audits the supplied frozen results only. Sensitivities
are post-review, fixed-configuration analyses; they never select a new winner.
"""
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
    # Cartesian product of OBSERVED marginal levels, not physically feasible combinations.
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
    # This is a TEMPLATE. Never overwrite a human-completed file.
    if not (Path(out) / 'censor_protocol_TEMPLATE.csv').exists():
        review_save(protocol, out, 'censor_protocol_TEMPLATE')
    return joint


def review_primary_scores(frame):
    """Equal campaigns, then equal records. Exact-subset weights recomputed."""
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
    """Positive differences are sensitivity minus reference, not 'improvement'.

    Cluster bootstrap of retained predictions, conditional on fitted models.
    Equal campaign weighting retained when a cluster is drawn multiple times.
    """
    a = a.sort_values('row_id').reset_index(drop=True)
    b = b.sort_values('row_id').reset_index(drop=True)
    if not a[REVIEW_ID].equals(b[REVIEW_ID]):
        raise AssertionError('Paired predictions do not have identical observations')
    groups = list(a.groupby('campaign_id').indices.values())
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ['SB_RMSE', 'SB_MAE', 'SB_R2', 'joint_NLL']}
    # Sufficient statistics give exactly equal campaign weights without relabeling copies.
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
    """Observed feature support; no claim of mechanistic/counterfactual identifiability."""
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
            # Known marginal ranges do not imply convex-hull membership.
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
    """All OOF locations, residual preprocessing and scale fitted on development only."""
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
    # Match the original primary hybrid development split. Physical-only WB-CD
    # uses the original physical comparator split (+100 for spec_index=1).
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
    # JSON, never pickle: safe resume with complete code/data/dependency/config fingerprint.
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

本次使用原始工作簿中的223条记录。主分析结果来自本次指定的主结果工作簿（来源哈希见provenance.json）；新增分析不会据外层测试结果挑选新主模型。

## 可直接用于修改的证据

- 架构×温区×环境实际占用 {len(joint)} 个组合，其中 {(joint.n_campaigns==1).sum()} 个仅来自一次实验研究。记录数、失效数、runout数、独立campaign数量分别报告；没有把记录数称为统计独立的有效样本量。
- 修订主分析 R²={m.SB_R2:.3f}，95% CI [{m.SB_R2_lo:.3f}, {m.SB_R2_hi:.3f}]；RMSE={m.SB_RMSE:.3f} log10周期，MAE={m.SB_MAE:.3f}。10^MAE={10**m.SB_MAE:.2f} 是按原权重计算的绝对倍率误差的几何平均值。10^RMSE不可解释为平均倍率误差或可靠性界限。
- WB-PIML相对WB-CD的完整删失NLL差为 {delta.iloc[0]:+.6f}，失效分量 {delta.iloc[1]:+.6f}，runout分量 {delta.iloc[2]:+.6f}。三个分量采用同一完整测试集权重，前两分量相加严格等于总差。
- 选型频次表覆盖候选组合的零次选择情况，树参数只统计适用的树模型折。选型规则重放不产生新的外层性能。
- holdout_overlap.csv 显示已有因子保留验证中的campaign重叠；这些已保存预测不能冒充剔除重叠后重新训练的结果。prediction_support_rows.csv 使用训练协变量界定支持范围；“范围内”也不证明已知材料系统之外的泛化。

## 分析范围

- 重新拟合敏感性分析本次完成：{refit_completed}。具体完成的变体见 sensitivity_plan.csv；仅在使用 --refit 时生成。checkpoint保存每折训练和测试标识哈希、内层OOF预测和参数。
- 分布敏感性分为：固定位置的三分布尺度重校准，以及指定变体的全物理干线/残差重新拟合。尺度只使用开发集OOF预测校准。三种密度统一在Y=log10(N)坐标下评分，Weibull使用中位数位置参数。
- runout_loss_power是删失损失影响分析，不代表已拟合信息性删失机制。所有分布分析仍以条件独立删失为前提。删失协议模板需要原文页码证据，不能从整百万的停止次数推定。
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




# ==================== 第四部分：结果完整性与隔离检查 ====================
"""Verify a completed full reviewer run without fitting any models."""
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def review_verify_main(argv=None):
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=root/'reviewer_results')
    parser.add_argument('--data',type=Path,default=root/'data.xlsx')
    parser.add_argument('--baseline',type=Path,default=root/'baseline_results.xlsx')
    args=parser.parse_args(argv)
    out=args.results.resolve()
    status=json.loads((out/'run_status.json').read_text(encoding='utf-8'))
    assert status['status']=='completed' and status['refits_executed'] and status['variants'] is None
    provenance=json.loads((out/'provenance.json').read_text(encoding='utf-8'))
    assert review_sha(args.data)==provenance['data_sha256']
    assert review_sha(args.baseline)==provenance['baseline_sha256']
    for name,digest in provenance['code_sha256'].items():assert review_sha(root/name)==digest,name
    run_hash=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    df=load_data(args.data)
    expected=set(df.row_id)
    all_predictions=pd.read_csv(out/'sensitivity_predictions.csv')
    assert set(all_predictions.variant)=={v['name'] for v in review_variants()}
    checked=0
    for name,group in all_predictions.groupby('variant'):
        assert len(group)==223 and group.row_id.is_unique and set(group.row_id)==expected,name
        assert group.is_exact.sum()==158 and group.is_runout.sum()==65
        for fold in range(1,6):
            p=json.loads((out/'checkpoints'/f'{name}_fold{fold}.json').read_text(encoding='utf-8'))
            assert p['run_hash']==run_hash
            assert p['outer_test_used_for_calibration'] is False
            test=pd.DataFrame(p['predictions']);oof=pd.DataFrame(p['calibration'])
            train=df.loc[~df.row_id.isin(test.row_id)]
            review_require_disjoint(train,test)
            assert set(oof.row_id)==set(train.row_id) and oof.row_id.is_unique
            assert set(test.row_id)|set(oof.row_id)==expected
            assert probability_row_id_hash(train.row_id)==p['train_hash']
            assert probability_row_id_hash(test.row_id)==p['test_hash']
            for audit in p['audit']:
                valid=oof.loc[oof.inner_fold.eq(audit['inner_fold'])]
                fit=train.loc[~train.row_id.isin(valid.row_id)]
                review_require_disjoint(fit,valid)
                assert probability_row_id_hash(fit.row_id)==audit['fit_hash']
                assert probability_row_id_hash(valid.row_id)==audit['validation_hash']
            family=p['assumption'].get('distribution','lognormal')
            np.testing.assert_allclose(test.row_nll,review_row_losses(test.logN,test.pred_logN,
                test.scale,test.is_exact,family),rtol=1e-12)
            np.testing.assert_allclose(p['scale_fit']['calibration_nll'],np.mean(review_row_losses(
                oof.logN,oof.mu,p['scale_fit']['scale'],oof.is_exact,family)),rtol=1e-12)
            check=test.merge(df[REVIEW_ID],on='row_id',suffixes=('_prediction','_data'),validate='one_to_one')
            np.testing.assert_array_equal(check.is_exact_prediction,check.is_exact_data)
            np.testing.assert_allclose(check.logN_prediction,check.logN_data,rtol=1e-12)
            checked+=1
    m=pd.read_csv(out/'sensitivity_metrics.csv')
    for name,g in all_predictions.groupby('variant'):
        exact=g.loc[g.is_exact].copy();exact['error']=exact.pred_logN-exact.logN
        rmse=float(np.sqrt(exact.groupby('campaign_id').error.apply(lambda x:np.mean(x*x)).mean()))
        mae=float(exact.groupby('campaign_id').error.apply(lambda x:np.mean(np.abs(x))).mean())
        row=m.set_index('variant').loc[name]
        np.testing.assert_allclose([row.SB_RMSE,row.SB_MAE],[rmse,mae],rtol=1e-12,atol=1e-12)
    if provenance.get('sensitivity_design_version')==REVIEW_DESIGN_VERSION:
        design=pd.read_csv(out/'sensitivity_design.csv')
        if set(design.variant)!=set(all_predictions.variant):raise AssertionError('Sensitivity plan coverage mismatch')
        if design.used_for_model_selection.any():raise AssertionError('Sensitivity must not choose primary model')
        expected_plan=hashlib.sha256(json.dumps(review_design_rows(),sort_keys=True).encode('utf-8')).hexdigest()
        if expected_plan!=provenance['sensitivity_plan_sha256']:raise AssertionError('Sensitivity plan changed during run')
        norm=pd.read_csv(out/'environment_scale_identifiability.csv')
        if len(norm)!=10 or norm.max_abs_difference.max()>1e-10:raise AssertionError('Scale identity diagnostics incomplete')
        response=pd.read_csv(out/'environment_response_rows.csv')
        if not np.isfinite(response[['frozen_trunk_delta_logN','refitted_pipeline_delta_logN']].to_numpy()).all():
            raise AssertionError('Nonfinite environmental response')
        expected_drivers={v['name'] for v in review_variants() if v.get('drivers')}
        if set(response.variant)!=expected_drivers:raise AssertionError('Environmental response coverage incomplete')
        for _,g in response.groupby('variant'):
            if len(g)!=len(df) or not g.row_id.is_unique:raise AssertionError('Environmental response row mismatch')
        interactions=pd.read_csv(out/'environment_interactions.csv')
        if interactions.variant.nunique()!=4:raise AssertionError('Interaction checks incomplete')
        detail=pd.read_csv(out/'data_joint_detailed.csv')
        if detail.n_records.sum()!=len(df):raise AssertionError('Detailed support counts incomplete')
    np.testing.assert_allclose(m.joint_NLL,m.exact_component_NLL+m.runout_component_NLL,atol=1e-12)
    assert pd.read_csv(out/'baseline_metric_regression.csv').passed.all()
    assert pd.read_csv(out/'baseline_prediction_regression.csv').passed.all()
    freq=pd.read_csv(out/'hyperparameter_frequency.csv')
    family=freq.loc[freq.model.eq('WB-PIML-Anchor') & freq.parameter.eq('family')].set_index('value')
    selection=pd.read_excel(args.baseline,sheet_name='Nested_Selection')
    expected_counts=selection.loc[selection.model.eq('WB-PIML-Anchor'),'family'].value_counts()
    for name,row in family.iterrows():assert row['count']==expected_counts.get(name,0)
    assert family['count'].sum()==5
    assert family.role.eq('searched').all()
    water_verify_outputs(out,df)
    strict_verify_outputs(out,df)
    summary=dict(status='passed',variants=len(review_variants()),outer_fits=checked,inner_OOF_fits=checked*4,
        prediction_rows=len(all_predictions),bootstrap_draws=status['bootstrap'],
        input_and_code_hashes='matched',campaign_leakage_in_refits=False,
        checks='coverage, train/test and inner campaign separation, likelihood/scale arithmetic, independent RMSE/MAE, baseline reproduction, selection frequencies and design diagnostics')
    summary['strict_holdout']=json.loads((out/'strict_holdout/summary.json').read_text(encoding='utf-8'))
    summary['censor_evidence']=json.loads((out/'censor_evidence_status.json').read_text(encoding='utf-8'))
    summary['status_meaning']='available-output arithmetic and isolation checks passed; see disclosed unavailable tasks and evidence gaps'
    (out/'verification_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))




# ==================== 第五部分：单文件统一入口 ====================
SINGLE_PRIMARY_SHA256 = '268f45521937cb6728e75a7f169e0125b9c25d483121a0d1d7a4b42166d1b700'
SINGLE_PRIMARY_GUARD = 'if __name__ == "__main__":\n    main()\n'
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import importlib.metadata
import sys
from types import SimpleNamespace

SINGLE_DATA_SHA256 = '8047f89eaee10db4bbb7b7919c31105886e59204202f52586133e4b3b18b9935'
SINGLE_DEPENDENCIES = {'numpy':'2.3.5','pandas':'2.2.3','scipy':'1.17.0',
    'scikit-learn':'1.8.0','openpyxl':'3.1.5','joblib':'1.5.3',
    'threadpoolctl':'3.6.0','matplotlib':'3.10.8','python-pptx':'1.0.2','Pillow':'12.3.0'}


def single_available_memory_gib():
    """读取当前可用物理内存；探测失败返回None，不增加额外依赖。"""
    try:
        if os.name=='nt':
            import ctypes
            class MemoryStatus(ctypes.Structure):
                _fields_=[('dwLength',ctypes.c_uint32),('dwMemoryLoad',ctypes.c_uint32),
                    ('ullTotalPhys',ctypes.c_uint64),('ullAvailPhys',ctypes.c_uint64),
                    ('ullTotalPageFile',ctypes.c_uint64),('ullAvailPageFile',ctypes.c_uint64),
                    ('ullTotalVirtual',ctypes.c_uint64),('ullAvailVirtual',ctypes.c_uint64),
                    ('ullAvailExtendedVirtual',ctypes.c_uint64)]
            memory=MemoryStatus();memory.dwLength=ctypes.sizeof(memory)
            query=ctypes.WinDLL('kernel32',use_last_error=True).GlobalMemoryStatusEx
            query.argtypes=[ctypes.POINTER(MemoryStatus)];query.restype=ctypes.c_int
            if query(ctypes.byref(memory)):return memory.ullAvailPhys/(1024**3)
        elif sys.platform.startswith('linux'):
            for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
                if line.startswith('MemAvailable:'):return int(line.split()[1])/(1024**2)
    except (OSError,ValueError,AttributeError):
        pass
    return None


def single_worker_budget(logical,physical,available_gib):
    """执行资源预算，不参与模型选择。内存预算为启发式，并非实测峰值。"""
    # 本版针对截图中的16核9950X，最多16进程；探测失败不盲目启用全部SMT线程。
    cores=max(1,min(int(logical),int(physical),16))
    if available_gib is None:return min(cores,16)
    # 留4 GiB给主进程/图表；其余按每个worker 1.5 GiB预留。
    memory_workers=max(1,int(max(0.0,float(available_gib)-4.0)/1.5))
    return min(cores,memory_workers)


def single_parallel_settings(requested=None):
    from joblib import cpu_count
    logical=max(1,cpu_count())
    try:physical=max(1,cpu_count(only_physical_cores=True))
    except (OSError,ValueError,NotImplementedError):physical=min(logical,16)
    available=single_available_memory_gib()
    jobs=single_worker_budget(logical,physical,available) if requested is None else requested
    return jobs,dict(mode='automatic' if requested is None else 'manual',
        available_logical_cpus=logical,available_physical_cores=physical,
        available_memory_GiB=None if available is None else round(available,2),
        workers=jobs,threads_per_worker=1,
        policy='physical cores with heuristic memory allowance; no speed benchmark or model training')


def single_environment_notes(version_info,installed):
    """区分参考版本和必要功能，不将参考环境误写成唯一可运行环境。"""
    if tuple(version_info[:2]) < (3,12):
        raise RuntimeError('此入口要求 Python 3.12 或更高版本。')
    notes=[]
    if tuple(version_info[:2]) != (3,12):
        notes.append('当前Python不是3.12参考环境；允许继续，完整运行尚需在当前环境验证。')
    notes.extend(f'{name}: 当前 {installed[name]}，参考 {wanted}'
        for name,wanted in SINGLE_DEPENDENCIES.items() if installed[name]!=wanted)
    if notes:
        notes.append('版本差异可能影响数值结果；保留实际环境记录和原有预测/概率一致性校验，不自动放宽结果容差。')
    return notes


def single_preflight(data):
    """只检查环境、文件和数据，不执行训练。"""
    source=Path(__file__).read_text(encoding='utf-8')
    start='# ===== REVISED PRIMARY CODE BEGIN =====\n'
    end='# ===== REVISED PRIMARY CODE END =====\n'
    preserved=source.split(start,1)[1].split(end,1)[0]+SINGLE_PRIMARY_GUARD
    if hashlib.sha256(preserved.encode('utf-8')).hexdigest()!=SINGLE_PRIMARY_SHA256:
        raise RuntimeError('修订主分析代码校验不一致，请使用完整原文件。')
    if not data.is_file():
        raise FileNotFoundError(f'未找到原始数据：{data}。请把 data.xlsx 放在本代码旁。')
    if review_sha(data)!=SINGLE_DATA_SHA256:
        raise RuntimeError('数据与本次返修所核对的原始数据不一致。')
    installed={name:importlib.metadata.version(name) for name in SINGLE_DEPENDENCIES}
    environment_notes=single_environment_notes(sys.version_info,installed)
    import inspect
    if 'return_as' not in inspect.signature(Parallel).parameters:
        raise RuntimeError('当前joblib缺少return_as接口，需安装支持此接口的版本（参考1.5.3）。')
    if any(value is None for value in (Font,Image,Presentation,RGBColor,PP_ALIGN,Inches,Pt)):
        raise ImportError('缺少 Excel/PPT 导出依赖，见文件开头。')
    frame=load_data(data)
    if (len(frame),int(frame.is_exact.sum()),int(frame.is_runout.sum()))!=(223,158,65):
        raise RuntimeError('数据记录/失效/runout数量与返修数据不符。')
    if not frame.row_id.is_unique or not frame.record_equivalence_id.is_unique:
        raise RuntimeError('存在重复分析标识。')
    return dict(python=sys.version,versions=installed,environment_notes=environment_notes,
        reference_python='3.12',reference_versions=SINGLE_DEPENDENCIES,
        data_sha256=review_sha(data),
        revised_primary_sha256=SINGLE_PRIMARY_SHA256,single_file_sha256=review_sha(__file__),
        records=223,exact=158,runout=65,framework_unchanged=True,environment_assumption_changed=True,primary_model_id=REVISION_MODEL_ID)


class SingleLogTee:
    def __init__(self,console,log):
        self.console,self.log=console,log
    def write(self,text):
        self.log.write(text);self.log.flush()
        try:return self.console.write(text)
        except UnicodeEncodeError:
            encoding=getattr(self.console,'encoding',None) or 'ascii'
            return self.console.write(text.encode(encoding,errors='replace').decode(encoding))
    def flush(self):
        self.console.flush();self.log.flush()
    def __getattr__(self,name):
        return getattr(self.console,name)


def single_check_primary(workbook):
    required={'Protocol','Nested_Selection','Nested_Candidates','Prob_Predictions',
              'Prob_Calibration','Prob_Metrics','Primary_Campaign_Metrics','Paired_Bootstrap',
              'Prob_Paired_CI','Empirical_PI_Summary','External_PI_Summary','Sensitivity','External_Selection'}
    with pd.ExcelFile(workbook) as excel:
        if not required.issubset(excel.sheet_names):raise RuntimeError('主工作簿缺少必要表。')
        protocol=pd.read_excel(excel,sheet_name='Protocol').set_index('item').value
    if str(protocol.get('revision_model_id',''))!=REVISION_MODEL_ID:
        raise RuntimeError('该主工作簿不是可学习水氧倍率版；请默认完整运行，不能复用旧固定权重主结果。')
    if str(protocol['data_sha256'])!=SINGLE_DATA_SHA256:
        raise RuntimeError('主工作簿与本次数据不一致。')


def single_main(argv=None):
    """默认一次完成修订主分析、返修补充、校验与独立验证准备。"""
    parser=argparse.ArgumentParser(description='WB-PIML 单文件返修版；默认完整运行，--check 只检查环境。')
    parser.add_argument('--check',action='store_true',help='只检查环境和数据，不训练')
    parser.add_argument('--data',type=Path,default=None,help='默认自动识别同目录data.xlsx或唯一data*.xlsx')
    parser.add_argument('--out',type=Path,help='新的结果目录；默认同目录 runs/full_时间')
    parser.add_argument('--jobs',type=int,default=None,help='默认自动按CPU及可用内存设置；也可手动指定正整数')
    parser.add_argument('--censor-evidence',type=Path,help='可选：逐条停止规则证据CSV；无此文件也可完整运行')
    parser.add_argument('--reviewer-only',action='store_true',help='只对已完成的主结果运行补充分析')
    parser.add_argument('--baseline',type=Path,help='与 --reviewer-only 一起使用的主结果xlsx路径')
    parser.add_argument('--predict',type=Path,help='新实验条件CSV/XLSX；仅用冻结模型预测')
    parser.add_argument('--model',type=Path,help='--predict使用的frozen_model.joblib')
    parser.add_argument('--evaluate',type=Path,help='独立实验结果CSV/XLSX；不训练')
    parser.add_argument('--predictions',type=Path,help='--evaluate使用的预先保存predictions.csv')
    args=parser.parse_args(argv)
    if args.predict or args.evaluate:
        revision_external_cli(args,parser)
        return
    if args.model or args.predictions:parser.error('--model/--predictions须与相应独立验证模式同时给出')
    if args.jobs is not None and args.jobs<1:parser.error('--jobs 必须大于0')
    if args.reviewer_only!=bool(args.baseline):parser.error('--reviewer-only 与 --baseline 必须同时给出')
    data=revision_resolve_data(args.data)
    args.jobs,parallel_settings=single_parallel_settings(args.jobs)
    # 在创建worker前设置线程上限，防止多进程再各自扩张BLAS线程。
    for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                     'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
        os.environ[variable]='1'
    environment=single_preflight(data)
    environment['parallel_settings']=parallel_settings
    print('启动检查：环境、数据和多进程函数传递（不训练模型）...',flush=True)
    environment['execution_preflight']=single_execution_preflight(args.jobs)
    if args.check:
        print(json.dumps(environment,ensure_ascii=False,indent=2))
        print('PREFLIGHT PASSED. 环境检查通过；未启动训练。')
        return
    output=(args.out or Path(__file__).resolve().parent/'runs'/('full_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))).resolve()
    if output.exists() and any(output.iterdir()):parser.error(f'结果目录非空，请使用新目录：{output}')
    output.mkdir(parents=True,exist_ok=True)
    review_write_design(output)
    (output/'analysis_plan.json').write_text(json.dumps(review_design_rows(),ensure_ascii=False,indent=2),encoding='utf-8')
    status=dict(status='running',mode='reviewer_only' if args.reviewer_only else 'full',
        environment=environment,completed_stages=[],source_file=Path(__file__).name,
        settings=dict(seed=SEED,trees=400,audit_trees=160,repeats=10,bootstrap=5000,jobs=args.jobs),
        primary_model_id=REVISION_MODEL_ID,supplement_role='固定修订主模型已选配置的敏感性及诊断；不据此再改选')
    def write_status():
        (output/'workflow_status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    write_status()
    global WORKERS,TREE_JOBS,BLAS_THREADS
    WORKERS,TREE_JOBS,BLAS_THREADS=args.jobs,1,1
    try:
        with (output/'run.log').open('w',encoding='utf-8') as log:
            with redirect_stdout(SingleLogTee(sys.stdout,log)),redirect_stderr(SingleLogTee(sys.stderr,log)):
                print(json.dumps(environment,ensure_ascii=False,indent=2),flush=True)
                print(f'自动/手动并行设置：{args.jobs} 个进程，每进程 1 个计算线程。',flush=True)
                print('每个阶段按实际任务数使用进程；5折主验证最多同时处理5折。',flush=True)
                if args.reviewer_only:
                    baseline=args.baseline.resolve();single_check_primary(baseline)
                else:
                    primary=output/'primary'
                    print('阶段 1/4：可学习水氧倍率主分析（原候选池及验证规模）',flush=True)
                    primary_args=SimpleNamespace(data=str(data),out=str(primary),trees=400,
                        audit_trees=160,repeats=10,bootstrap=5000,export_csv=False)
                    with threadpool_limits(limits=1):_run_analysis(primary_args)
                    baseline=primary/'WB_PIML_results.xlsx';single_check_primary(baseline)
                    if not (primary/'WB_PIML_figures.pptx').is_file() or len(list(primary.glob('*.png')))<10:
                        raise RuntimeError('修订主分析图件/PPT输出不完整。')
                    status['completed_stages'].append('primary');write_status()
                status['primary_workbook']=str(baseline)
                status['primary_workbook_sha256']=review_sha(baseline);write_status()
                reviewer=output/'reviewer'
                print(f'阶段 2/4：审稿补充统计及{len(review_variants())}组具名敏感性分析（不改选主模型）',flush=True)
                review_main(['--refit','--data',str(data),'--baseline',str(baseline),
                    '--out',str(reviewer),'--jobs',str(args.jobs),'--trees','400','--bootstrap','5000']
                    +(['--censor-evidence',str(args.censor_evidence)] if args.censor_evidence else []))
                status['completed_stages'].append('reviewer')
                status['strict_holdout']=json.loads((reviewer/'strict_holdout/summary.json').read_text(encoding='utf-8'))
                status['censor_evidence']=json.loads((reviewer/'censor_evidence_status.json').read_text(encoding='utf-8'))
                write_status()
                print('阶段 3/4：预测、概率计算和训练/测试隔离检查',flush=True)
                review_verify_main(['--results',str(reviewer),'--data',str(data),'--baseline',str(baseline)])
                status['completed_stages'].append('verification')
                write_status()
                print('阶段 4/4：按原内层规则选择并冻结全数据模型，生成独立实验模板',flush=True)
                with threadpool_limits(limits=1):
                    revision_export_frozen(data,output/'prospective',trees=400)
                status['completed_stages'].append('prospective')
                status['status']='completed_with_strict_fit_failures' if status['strict_holdout']['n_fit_failed'] else 'completed'
                status['scientific_readiness']='requires_manuscript_and_source_evidence_review'
                write_status()
                print(f'计算流程结束（状态：{status["status"]}）；请阅读返修关键结论与缺口.md。\n结果目录：{output}',flush=True)
    except BaseException as exc:
        status['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        status['error']=str(exc);write_status()
        raise


# ==================== 返修分析设计（不是模型优化配置） ====================
REVIEW_DESIGN_VERSION = 'reviewer-audit-20260928-v4-learned-water'
REVIEW_ENVIRONMENT_BASIS = '''# 可学习水氧倍率的假设与验证

主模型：E(vacuum)=0，E(air/oxygen)=1，E(water/oxygen)=10**beta，beta>=0。
只增加一个物理位置参数；原残差候选池、种子、删失似然及组隔离验证规模保留。
每个内层训练子集重新估计beta，外层测试寿命不用于参数、范围或候选选择。
真空0为理想化边界；空气1固定尺度；水氧方向约束是适用条件下的物理先验，不是全条件实验证明。
主拟合无有限beta上界，无新增beta正则项。原有物理参数边界、ridge和环境惩罚仍适用。
计算域对照10/100仅检验一个/两个数量级限制是否改变估计，不是物理上限或待选超参数。
初值中的1/1.5/5只用于同一目标的数值多起点，不是允许权重的离散集合。

数据限制：水氧12条记录仅1300℃、3Hz、R=0.1、2D编织，来自两个试验组，无匹配空气数据。
参数可能与温度、K和试验来源混杂。可拟合不等于可辨识，更不等于测得独立环境反应速率。
若训练无水氧或无空气参考，beta固定0且标记未辨识；不能利用留出环境寿命拟合。
水氧LOEO中的参考倍率1只是未辨识情况下的公开假设，不能称为学习了未知气氛效果。

对照：selected_pipeline为训练内估计；water_1和water_1.5为固定权重条件对照。
所有对照保留修订模型选定残差配置，但重新拟合其训练物理参数和OOF尺度。
原0.5/1.25/1.75/2/3情景仍保留，0.5是允许方向反转的压力测试，不是主模型先验。
no_environment_damage关闭显式物理项但保留残差环境输入；environment_penalty_0仅取消惩罚。
温度/应力/频率驱动消融、真空小非零情景及有限权重×频率交互全部保留。
水氧系数不由环境样本比例或不同环境原始平均寿命直接计算。

water_identifiability_profiles.csv同时报告训练集原惩罚目标剖面和保留干扰参数边界的无惩罚似然剖面。
不从剖面挑更高外测精度，不自动给95%区间；若范围内曲线平坦或自由拟合存在局部极值，明确输出。
water_air_counterfactual_order.csv检查物理主干和完整模型；前者约束有保证，后者无约束残差可能逆转。
这些环境切换是模型情景检查，不是实测空气/水氧配对验证。

本次是审稿和已查看结果之后的模型修订，不称为预注册。敏感性区间是固定OOF预测的试验组bootstrap，
不包含模型重选不确定性，也未作多重比较校正。主模型不根据外层误差继续换权重。

机制资料只支持环境作用及条件依赖，不支持通用1.5倍率：
Liu C.-D. et al. Fatigue Behavior of 2D C/SiC Composites in Water-vapor Containing Environment at 1300℃.
Journal of Inorganic Materials, 2008, 23(4), 729. https://doi.org/10.3724/SP.J.1077.2008.00729
C/SiC Life Prediction for Propulsion Applications. NASA NTRS 20040033966.
https://ntrs.nasa.gov/citations/20040033966
参数诊断方法参考：Raue A. et al. Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood.
Bioinformatics 2009;25(15):1923–1929. https://doi.org/10.1093/bioinformatics/btp358
本程序没有直接使用该文的渐近置信阈值，因本模型含约束、惩罚和少量异质试验组。

后续先冻结预测再记录实验结果；空气试验不能标定水氧倍率。倍率需同批材料同工况气氛对照及独立验证组。
替代生存分布仍假设条件独立删失；runout权重分析不证明该假设成立。停止规则仍需文献证据。
预测区间尚非工程设计寿命界限。
'''


def review_design_rows():
    rows=[]
    for v in review_variants():
        n=v['name'];d=v.get('drivers',{})
        if n=='selected_pipeline':
            role,reason='primary_reference','训练内估计水氧倍率参照；不按外层误差改选'
        elif n.startswith('water_cap_'):
            role,reason='computational_boundary_test','log10倍率上界1/2即10/100，仅为一/两个数量级计算域依赖检查；不是物理上界或待选超参数'
        elif n.startswith('water_'):
            x=d['water_factor']
            role=('stress_test' if x in (.5,3.) else 'fixed_equal_control' if x==1 else 'legacy_weight_control' if x==1.5 else 'local_perturbation')
            reason=f'相对于原水氧1.5的比值={x/1.5:g}；相对于空气1的比值={x:g}；范围是分析假设，不是实测置信区间'
        elif n.startswith('inert_'):
            role,reason='zero_boundary_test',f'真空显式通道从0改为参考空气的{100*d["inert_factor"]:g}%；不是残氧测量值'
        elif n.startswith('joint_water_'):
            role,reason='limited_interaction','水氧0.5/2与频率指数0.5/1.5的2×2交叉；保留对应单因素对照'
        elif n.startswith('frequency_'):
            role,reason='exposure_time_test','指数1对应每循环暴露时间1/f；0关闭该显式频率驱动，0.5/1.5为对称扰动'
        elif n.startswith('runout_'):
            role,reason='censor_loss_influence','删失损失乘子0/0.5/2；评价用标准NLL；不代表改变真实停止机制'
        elif n.startswith('refit_') or n.startswith('WB_CD_'):
            role,reason='distribution_assumption','替代生存分布；参数仅由开发集估计；全部在log10寿命坐标评分'
        elif n.startswith('environment_penalty_'):
            role,reason='regularization_audit','原候选池已有惩罚水平；与关闭环境损伤通道严格区分'
        else:
            role,reason='component_ablation','具名组件消融；不删除残差模型原始协变量，不改变主模型'
        rows.append(dict(variant=n,role=role,rationale=reason,settings_json=json.dumps(v,sort_keys=True),
            water_to_air=d.get('water_factor','learned_training_only'),inert_to_air=d.get('inert_factor',0.),
            frequency_exponent=d.get('frequency_exponent',1.),
            numeric_weights_measured=False,used_for_model_selection=False,
            design_timing='revision-stage post-hoc; previously inspected data',
            scope='conditional sensitivity, not global sensitivity or new external validation'))
    return rows


def review_write_design(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    review_save(pd.DataFrame(review_design_rows()),out,'sensitivity_design')
    (out/'运行与修改说明.txt').write_text(REVISION_RUN_GUIDE,encoding='utf-8')
    (out/'environment_weight_rationale.md').write_text(REVIEW_ENVIRONMENT_BASIS,encoding='utf-8')
    mapping=[
        ('R2.1','Novelty','manuscript_required','Established components must be distinguished from integration contribution'),
        ('R2.2/R2.8','Suggested literature','manuscript_required','10.3390/machines13100906; 10.1002/srin.202400850; no automatic algorithm adoption'),
        ('R2.3','Database support','data_joint_grid.csv; data_joint_detailed.csv; sparse_cell_performance.csv','Counts are not formal independent effective sample size'),
        ('R2.4','Low R2','retained_Primary_Campaign_Metrics.csv; retained_Paired_Bootstrap.csv','Absolute errors, intervals, stronger baselines; no design-life claim'),
        ('R2.5/R3.1','Environment weights','sensitivity_design.csv; sensitivity_metrics.csv; environment_improvement_attribution.csv; water_fallback_attribution.csv; water_air_counterfactual_order.csv','No calibrated universal weights; limited interactions; no outcome-driven winner'),
        ('R2.6','Selection overfitting','candidate_selection_frequency.csv; hyperparameter_frequency.csv','Nested validation does not eliminate small-campaign uncertainty'),
        ('R2.7','Runouts','distribution_refit_PIML_vs_CD.csv; censor_protocol_audit.csv; censor_evidence_gaps.csv','Stopping mechanism requires source evidence; no fabricated completed audit'),
        ('R2.9','Generalization','holdout_overlap.csv; strict_holdout/task_status.csv; strict_holdout/metrics_matched_population.csv; support_stratum_performance.csv','Factor holdouts with campaign overlap are not independent material validation'),
        ('R2.10/R3.5','Conclusions and intervals','retained_Empirical_PI_Summary.csv; retained_External_PI_Summary.csv; ../prospective/model_manifest.json; ../prospective/独立实验使用说明.txt','Future independent matched-condition validation required'),
        ('R3.2','NLL tradeoff','probability_score_decomposition.csv; primary_NLL_difference_CI.csv','Exact and runout contributions use common population weights'),
        ('R3.3','Weak holdouts','holdout_performance_campaign_weighted.csv','Report vacuum/water and temperature failures, do not tune them away'),
        ('R3.4','Penalty ablation','sensitivity_paired_CI.csv; sensitivity_subgroups.csv','Zero regularization is not zero environmental damage'),
    ]
    review_save(pd.DataFrame(mapping,columns=['reviewer_comment','topic','evidence_or_action','limitation']),out,'reviewer_response_map')


def review_environment_diagnostics(df,predictions,results,out):
    """No fitting or selection: fixed-coefficient mechanism shifts and paired prediction changes."""
    base=predictions.loc[predictions.variant.eq('selected_pipeline')].set_index('row_id')
    if base.empty:return
    reference={int(r['fold']):r for r in results if r['variant']=='selected_pipeline'}
    details=[];normalization=[]
    for fold,r in reference.items():
        ids=[p['row_id'] for p in r['predictions']]
        raw=df.set_index('row_id').loc[ids].reset_index()
        reference_frame=review_apply_physics_drivers(raw)
        coef=np.asarray(r['coefficients'],float);gamma=float(r['settings']['gamma'])
        original=review_predict_competing_damage(reference_frame,coef,gamma)
        for scale in (.5,2.):
            frame=reference_frame.copy();frame['environment_exposure']*=scale
            adjusted=coef.copy();adjusted[2]-=math.log10(scale)
            difference=np.max(np.abs(review_predict_competing_damage(frame,adjusted,gamma)-original))
            if difference>1e-10:raise AssertionError('Environment scale/K identity failed')
            normalization.append(dict(fold=fold,common_E_multiplier=scale,max_abs_difference=difference,
                refitted=False,note='algebraic invariance; transformed coefficients need not satisfy fit bounds'))
        for v in review_variants():
            if not v.get('drivers'):continue
            frame=review_apply_physics_drivers(raw,**v['drivers'])
            shift=review_predict_competing_damage(frame,coef,gamma)-original
            alt=predictions.loc[predictions.variant.eq(v['name'])].set_index('row_id')
            if alt.empty:continue
            for i,row in raw.iterrows():
                rid=row.row_id
                details.append(dict(variant=v['name'],fold=fold,row_id=rid,campaign_id=row.campaign_id,
                    source_id=row.source_id,record_equivalence_id=row.record_equivalence_id,
                    env_class=row.env_class,is_exact=bool(row.is_exact),
                    frozen_trunk_delta_logN=float(shift[i]),
                    refitted_pipeline_delta_logN=float(alt.loc[rid,'pred_logN']-base.loc[rid,'pred_logN'])))
    detail=pd.DataFrame(details);review_save(detail,out,'environment_response_rows')
    summary=[]
    for (variant,env),g in detail.groupby(['variant','env_class']):
        w=analysis_weights(g);w=w/w.sum()
        row=dict(variant=variant,env_class=env,n_records=len(g),n_campaigns=g.campaign_id.nunique())
        for c in ['frozen_trunk_delta_logN','refitted_pipeline_delta_logN']:
            row[c+'_weighted_mean']=float(np.dot(w,g[c]))
            row[c+'_weighted_mean_abs']=float(np.dot(w,np.abs(g[c])))
            row[c+'_max_abs']=float(np.abs(g[c]).max())
        summary.append(row)
    review_save(pd.DataFrame(summary),out,'environment_response_summary')
    review_save(pd.DataFrame(normalization),out,'environment_scale_identifiability')
    interaction=[]
    pivot=predictions.pivot(index='row_id',columns='variant',values='pred_logN')
    for water in (.5,2.):
        for freq in (.5,1.5):
            name=f'joint_water_{water:g}_frequency_{freq:g}'
            names=[name,f'water_{water:g}',f'frequency_power_{freq:g}','selected_pipeline']
            if not set(names).issubset(pivot.columns):continue
            delta=pivot[name]-pivot[names[1]]-pivot[names[2]]+pivot[names[3]]
            part=df.set_index('row_id').loc[delta.index].copy();part['interaction_logN']=delta
            for env,g in part.groupby('env_class'):
                w=analysis_weights(g);w/=w.sum()
                interaction.append(dict(variant=name,env_class=env,n_records=len(g),n_campaigns=g.campaign_id.nunique(),
                    weighted_mean_interaction=float(np.dot(w,g.interaction_logN)),
                    weighted_mean_absolute_interaction=float(np.dot(w,np.abs(g.interaction_logN))),
                    meaning='combined-minus-water-minus-frequency-plus-baseline; refitted pipeline, not causal interaction'))
    review_save(pd.DataFrame(interaction,columns=['variant','env_class','n_records','n_campaigns',
        'weighted_mean_interaction','weighted_mean_absolute_interaction','meaning']),out,'environment_interactions')


def single_worker_probe(functions):
    """Test worker loading of complete function graphs, without invoking any fitter."""
    names=[fn.__name__ for fn in functions]
    y=np.array([4.,5.]);mu=np.array([4.2,4.8]);exact=np.array([True,False])
    losses=review_row_losses(y,mu,.7,exact)
    quantiles=review_quantile(mu,.7,np.array([.05,.95]))
    return dict(functions=names,losses=losses.tolist(),quantiles=quantiles.tolist())


def single_execution_preflight(jobs):
    """Actual subprocess transport and likelihood checks; no scientific model is fitted."""
    if sys.flags.optimize:raise RuntimeError('请不要使用python -O/-OO；该选项会关闭结果校验断言。')
    funcs=[review_refit_one,_run_analysis,review_verify_main,water_profile_one,revision_export_frozen,strict_holdout_task]
    expected=single_worker_probe(funcs)
    if jobs>1:
        with parallel_config(backend='loky',inner_max_num_threads=1):
            rows=Parallel(n_jobs=2)(delayed(single_worker_probe)(funcs) for _ in range(2))
        if any(r!=expected for r in rows):raise RuntimeError('Worker startup self-check failed')
    return dict(status='passed',worker_processes_checked=2 if jobs>1 else 0,
                model_training_executed=False,checks='function transport, likelihood, quantiles')


# ==================== 独立实验：冻结预测与评价（不改主验证） ====================
REVISION_MODEL_ID = 'WB-PIML-learned-water-20260928'
REVISION_RUN_GUIDE = """本次版本：WB-PIML-learned-water-20260928
把本Python文件与data.xlsx（或唯一data*.xlsx）放在同一目录，VS Code运行Python文件即可。
无需旧代码和历史结果。完整运行四阶段：主分析、34组条件敏感性、结果校验、冻结模型导出。
自动按物理核心和可用内存安排多进程，每进程限制1个数值线程。

本次追加修改不改变主模型；新增严格因素留出、误差改善归因、NLL配对区间和停止证据审计。
优先阅读 reviewer/返修关键结论与缺口.md；严格验证见 reviewer/strict_holdout/。
同一轮主分析中原本隔离的因素任务复用，有重叠任务重新选型/校准；数据不足或失败明确报告。
原主模型设置：主干环境因子为0/1/10**beta，beta>=0且在训练内拟合；温度/应力/频率关系、环境类别残差输入、候选池、种子与验证规模保留。
所有主分析与因素保留验证重新计算。原0/1/1.5在补充分析中作为固定修订配置的条件对照。
不按敏感性分数寻找最优环境权重，不把模型估计当作已独立标定的环境物理常数。

输出：
primary/WB_PIML_results.xlsx：修订主结果、图件和PPT在primary目录。
reviewer/environment_model_comparison.csv：可学习倍率/固定1/固定1.5/关闭环境项的条件比较。
reviewer/environment_matched_support.csv：跨气氛可比条件数量，非因果辨识证据。
reviewer/verification_summary.json：完成全部34组以后生成的计算和隔离校验。
prospective/frozen_model.joblib及model_manifest.json：未来实验用冻结模型，不等于已经完成外部验证。
prospective/独立实验使用说明.txt及空白CSV模板：实验前预测与实验后评价方法。
workflow_status.json和run.log：当前状态和运行日志。

所有结果写入本Python文件旁runs/full_时间目录，不覆盖旧结果。
旧固定权重结果不能通过--reviewer-only当作新版主结果复用。
完整运行由用户执行，本版本未声称已在用户Windows/Python环境全量训练通过。
启动--check可验证环境和两进程传递，但不会计算论文预测结果。
"""

PROSPECTIVE_INPUT_COLUMNS = ['specimen_id','campaign_id','architecture','arch_group',
    'env_class','T_C','R','frequency_Hz','sigma_max_MPa_before_normalization','UTS_MPa']


def revision_resolve_data(argument=None):
    if argument is not None:
        p=Path(argument).expanduser().resolve()
        if not p.is_file():raise FileNotFoundError(f'指定的数据文件不存在：{p}')
        return p
    here=Path(__file__).resolve().parent
    default=here/'data.xlsx'
    if default.is_file():return default
    candidates=sorted(p for p in here.glob('data*.xlsx') if p.is_file() and not p.name.startswith('~$'))
    if len(candidates)==1:return candidates[0]
    if not candidates:raise FileNotFoundError('请把本Python文件与data.xlsx放在同一文件夹；data(9).xlsx也可自动识别。')
    raise ValueError('同目录存在多个data*.xlsx，请保留需要运行的一份，或用 --data 指定文件。')


def review_environment_overlap(df,out):
    """Exact covariate overlap only; it is not evidence of identical material batches."""
    keys=['architecture_detail','T_C','R','frequency_Hz']
    cells=review_counts(df,keys+['env_class'])
    review_save(cells,out,'environment_condition_counts')
    rows=[]
    for scope,matched in [('same_recorded_conditions',keys),
            ('same_conditions_and_stress',keys+['sigma_max_MPa_before_normalization']),
            ('same_campaign_conditions_and_stress',['campaign_id']+keys+['sigma_max_MPa_before_normalization'])]:
        for a,b in itertools.combinations(sorted(df.env_class.unique()),2):
            left=df.loc[df.env_class.eq(a)];right=df.loc[df.env_class.eq(b)]
            common=left[matched].drop_duplicates().merge(right[matched].drop_duplicates(),on=matched,validate='one_to_one')
            na=len(left.merge(common,on=matched,validate='many_to_one'))
            nb=len(right.merge(common,on=matched,validate='many_to_one'))
            rows.append(dict(scope=scope,environment_a=a,environment_b=b,
                matched_condition_cells=len(common),records_a=na,records_b=nb,
                interpretation='recorded covariate match only; material-batch/atmosphere equivalence requires source evidence'))
    review_save(pd.DataFrame(rows),out,'environment_matched_support')


def revision_json_default(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def revision_json(path,payload):
    Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2,
        default=revision_json_default),encoding='utf-8')


def revision_read_table(path):
    path=Path(path)
    if path.suffix.lower()=='.csv':return pd.read_csv(path,encoding='utf-8-sig')
    if path.suffix.lower() in ('.xlsx','.xlsm'):return pd.read_excel(path,sheet_name=0)
    raise ValueError('独立实验输入仅支持UTF-8 CSV或XLSX。')


def revision_prediction_frame(raw):
    missing=sorted(set(PROSPECTIVE_INPUT_COLUMNS)-set(raw.columns))
    if missing:raise ValueError(f'独立实验条件缺少列：{missing}')
    if raw.empty:raise ValueError('实验条件模板尚未填写。')
    d=raw[PROSPECTIVE_INPUT_COLUMNS].copy()
    for key in ['specimen_id','campaign_id','architecture','arch_group','env_class']:
        if d[key].isna().any():raise ValueError(f'{key}不能留空')
        d[key]=d[key].astype(str).str.strip()
        if d[key].eq('').any():raise ValueError(f'{key}不能留空')
    if d.specimen_id.duplicated().any():raise ValueError('specimen_id必须唯一')
    for key in ['T_C','R','frequency_Hz','sigma_max_MPa_before_normalization','UTS_MPa']:
        d[key]=pd.to_numeric(d[key],errors='raise')
        if not np.isfinite(d[key]).all():raise ValueError(f'{key}包含非有限值')
    if (d[['frequency_Hz','sigma_max_MPa_before_normalization','UTS_MPa']]<=0).any().any():
        raise ValueError('频率、最大应力和UTS必须为正数')
    if (d.R>=1).any() or (d.T_C<=-273.15).any():raise ValueError('R必须小于1，温度必须高于绝对零度')
    d['architecture_detail']=d.architecture.str.lower()
    d['architecture']=d.arch_group.str.lower();d['env_class']=d.env_class.str.lower()
    allowed={'air_or_oxygen','water_oxygen','vacuum'}
    if not set(d.env_class).issubset(allowed):raise ValueError(f'env_class请使用{sorted(allowed)}')
    d['S']=d.sigma_max_MPa_before_normalization/d.UTS_MPa
    d['stress_level_sigma_max_over_UTS']=d.S
    d['logS']=np.log10(d.S.clip(lower=1e-8));d['logf']=np.log10(d.frequency_Hz.clip(lower=1e-8))
    d['logUTS']=np.log10(d.UTS_MPa.clip(lower=1e-8));d['Tn']=(d.T_C-25.)/1000.
    d['Sa']=.5*d.S*np.clip(1.-d.R,0.,2.);d['Smean']=.5*d.S*(1.+d.R)
    return apply_physics_drivers(d)


def revision_predict_state(state,frame):
    coef=np.asarray(state['coefficients']);gamma=state['gamma']
    base=predict_physical(frame,COMPETING_WB_SPEC,coef,gamma)
    augmented=frame.copy()
    if state['family']=='physics_feature_ET_residual':
        mech=coef[0]-coef[1]*walker_log_stress(frame,gamma)
        augmented['mechanical_prediction']=mech
        augmented['environment_decrement']=mech-base
        augmented['walker_stress']=walker_log_stress(frame,gamma)
    xv=state['preprocessor'].transform(augmented)
    if state['eta']==0.0:correction=np.zeros(len(frame))
    elif state['family']=='Matern_residual':
        correction=Matern(length_scale=state['kernel_length_scale'],nu=1.5)(xv,state['kernel_x'])@state['kernel_weights']
    else:correction=state['estimator'].predict(xv)
    distance=np.abs(frame.T_C.to_numpy(float)[:,None]-np.asarray(state['training_temperatures'])[None,:]).min(axis=1)
    if not state['temperature_support_enabled']:support=np.ones(len(frame))
    elif state['temperature_policy'] in ('soft','soft250'):support=np.exp(-(distance/250.)**2)
    else:raise ValueError('独立预测只支持本研究既定soft250温度支持，不默默替代其他规则')
    mu=base+state['eta']*support*correction
    if not np.isfinite(mu).all():raise FloatingPointError('独立预测出现非有限值')
    return mu


def revision_export_frozen(data,out,trees=400):
    """Fit deployment model using the unchanged training-only selection rule.

    This is an additional final fit for future experiments, not a replacement
    of the five-fold generalization estimate. No new experimental response is read.
    """
    import joblib
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d=load_data(data)
    selection_seed=SEED+900000
    choice,oof,candidates=nested_select_hybrid(d,selection_seed,trees)
    keys=['family','gamma','strength','eta','min_samples_leaf','max_features',
          'ridge_alpha','kernel_length_scale','temperature_policy']
    settings={key:choice[key] for key in keys};settings['min_samples_leaf']=int(settings['min_samples_leaf'])
    mu,coef,meta=fit_hybrid_policy(d,d,COMPETING_WB_SPEC,selection_seed+1,trees,
        return_frozen_state=True,**settings)
    state=meta.pop('_frozen_state')
    # Equality checks run on the user's real fit: serialized inference must match fitting inference.
    np.testing.assert_allclose(revision_predict_state(state,d),mu,rtol=1e-11,atol=1e-11)
    scale=calibrate_aft_scale(oof);h80,h90=empirical_interval_halfwidths(oof)
    from datetime import timezone
    support_cols=['architecture','architecture_detail','env_class','T_C','R','frequency_Hz','S','UTS_MPa']
    manifest=dict(model_id=REVISION_MODEL_ID,created_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=review_sha(__file__),data_sha256=review_sha(data),seed=selection_seed,trees=trees,
        environment_factors={'vacuum':0.,'air_or_oxygen':1.,'water_oxygen':water_multiplier_value(coef[5])},
        water_fit=water_fit_metadata(d,coef),
        settings=settings,training_campaigns=sorted(d.campaign_id.unique().tolist()),
        training_records=len(d),interval_calibration='selected-candidate development OOF; selection and calibration share development records; not independent validation',
        claim='Frozen for prospective evaluation; not engineering design-life bounds',
        versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','pandas','joblib']})
    bundle=dict(manifest=manifest,state=state,scale=float(scale['sigma']),
        empirical_halfwidth80=float(h80),empirical_halfwidth90=float(h90),support=d[support_cols].copy())
    model_path=out/'frozen_model.joblib';joblib.dump(bundle,model_path,compress=3)
    loaded=joblib.load(model_path)
    np.testing.assert_allclose(revision_predict_state(loaded['state'],d),mu,rtol=1e-11,atol=1e-11)
    manifest['model_sha256']=review_sha(model_path)
    revision_json(out/'model_manifest.json',manifest)
    review_save(candidates,out,'final_training_selection');review_save(oof,out,'final_training_OOF')
    pd.DataFrame(columns=PROSPECTIVE_INPUT_COLUMNS+['material_batch','coating','waveform',
        'water_partial_pressure_kPa','oxygen_partial_pressure_kPa','gas_flow_rate','planned_stop_cycles']).to_csv(out/'new_conditions_TEMPLATE.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(columns=['specimen_id','Nf_cycles','is_runout','stop_reason','result_date']).to_csv(out/'new_outcomes_TEMPLATE.csv',index=False,encoding='utf-8-sig')
    (out/'独立实验使用说明.txt').write_text(PROSPECTIVE_GUIDE,encoding='utf-8')
    revision_json(out/'export_verification.json',dict(status='passed',
        checks='fit-vs-frozen and serialized-reload prediction equality',training_records=len(d),
        independent_experiments_evaluated=0))
    print(f'已保存冻结模型和空白实验模板：{out}',flush=True)


def revision_domain_flags(bundle,frame):
    support=bundle['support'];flags=[]
    for _,row in frame.iterrows():
        issues=[]
        for c in ['architecture','architecture_detail','env_class']:
            if row[c] not in set(support[c]):issues.append('unseen_'+c)
        for c in ['T_C','R','frequency_Hz','S','UTS_MPa']:
            if row[c]<support[c].min() or row[c]>support[c].max():issues.append('outside_'+c)
        matched=support.loc[(support.architecture_detail==row.architecture_detail)&
            (support.env_class==row.env_class)&np.isclose(support.T_C,row.T_C)&
            np.isclose(support.R,row.R)&np.isclose(support.frequency_Hz,row.frequency_Hz)]
        if matched.empty:issues.append('unseen_joint_conditions')
        elif row.S<matched.S.min() or row.S>matched.S.max():issues.append('outside_matched_stress_range')
        flags.append('|'.join(issues) if issues else 'within_recorded_covariate_support_not_material_validation')
    return flags


def revision_write_predictions(model,input_path,out):
    import joblib
    from datetime import timezone
    model=Path(model);side=model.with_name('model_manifest.json')
    if not side.is_file():raise FileNotFoundError('冻结模型缺少同目录model_manifest.json')
    manifest=json.loads(side.read_text(encoding='utf-8'))
    if manifest.get('model_id')!=REVISION_MODEL_ID or manifest.get('model_sha256')!=review_sha(model):
        raise ValueError('模型版本或文件哈希与说明不一致')
    for name,version in manifest['versions'].items():
        if importlib.metadata.version(name)!=version:
            raise RuntimeError(f'冻结模型要求{name}=={version}；请在原训练环境预测，避免反序列化差异。')
    # Only load models generated by this script in the user's trusted workspace.
    bundle=joblib.load(model)
    raw=revision_read_table(input_path)
    for c in ('Nf_cycles','is_runout','logN','event_observed'):
        if c in raw and raw[c].notna().any():raise ValueError('预测输入包含实验结果，请仅提供实验前条件表')
    frame=revision_prediction_frame(raw)
    old=set(frame.campaign_id)&set(manifest['training_campaigns'])
    if old:raise ValueError(f'新试验组ID与训练数据重叠：{sorted(old)}')
    mu=revision_predict_state(bundle['state'],frame)
    result=raw.copy();result['specimen_id']=frame.specimen_id;result['campaign_id']=frame.campaign_id
    result['model_id']=REVISION_MODEL_ID;result['pred_logN']=mu;result['scale']=bundle['scale']
    result['support_status']=revision_domain_flags(bundle,frame)
    for key,value in bundle['state'].get('water_fit',{}).items():result[key]=value
    result['predicted_median_cycles']=np.power(10.,np.clip(mu,-300,300))
    for level in (80,90):
        half=bundle[f'empirical_halfwidth{level}']
        result[f'empirical_lower{level}_logN']=mu-half;result[f'empirical_upper{level}_logN']=mu+half
        p=(1-level/100)/2
        result[f'prob_lower{level}_logN']=review_quantile(mu,bundle['scale'],p)
        result[f'prob_upper{level}_logN']=review_quantile(mu,bundle['scale'],1-p)
    for c in list(result.columns):
        if c.endswith('_logN') and ('lower' in c or 'upper' in c):
            result[c.replace('_logN','_cycles')]=np.power(10.,np.clip(result[c],-300,300))
    result['cycle_conversion_clipped']=np.abs(mu)>300
    for c in result:
        if c.endswith('_logN') and ('lower' in c or 'upper' in c):result['cycle_conversion_clipped']|=result[c].abs()>300
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    prediction_path=out/'predictions.csv';result.to_csv(prediction_path,index=False,encoding='utf-8-sig')
    revision_json(out/'prediction_manifest.json',dict(model_id=REVISION_MODEL_ID,
        created_utc=datetime.now(timezone.utc).isoformat(),model_sha256=review_sha(model),
        conditions_sha256=review_sha(input_path),predictions_sha256=review_sha(prediction_path),
        specimen_count=len(result),outcomes_used=False,
        note='Local timestamp/hash is a record, not external certification; archive before experiments.'))
    print(f'独立预测完成，未训练：{prediction_path}',flush=True)


def revision_evaluate(predictions,outcomes,out):
    predictions=Path(predictions);side=predictions.with_name('prediction_manifest.json')
    if not side.is_file():raise FileNotFoundError('需使用实验前生成的预测及同目录prediction_manifest.json')
    manifest=json.loads(side.read_text(encoding='utf-8'))
    if manifest['predictions_sha256']!=review_sha(predictions) or manifest['model_id']!=REVISION_MODEL_ID:
        raise ValueError('预测文件与保存时的哈希或模型版本不一致')
    pred=revision_read_table(predictions);obs=revision_read_table(outcomes)
    required={'specimen_id','Nf_cycles','is_runout'}
    if not required.issubset(obs) or obs.empty:raise ValueError('实验结果需填写specimen_id、Nf_cycles、is_runout')
    for d in (pred,obs):
        if d.specimen_id.isna().any():raise ValueError('specimen_id不能空缺')
        d['specimen_id']=d.specimen_id.astype(str).str.strip()
        if d.specimen_id.duplicated().any() or d.specimen_id.eq('').any():raise ValueError('specimen_id需非空且唯一')
    if not set(obs.specimen_id).issubset(pred.specimen_id):raise ValueError('存在没有预先预测的试样')
    obs['Nf_cycles']=pd.to_numeric(obs.Nf_cycles,errors='raise')
    if not np.isfinite(obs.Nf_cycles).all() or (obs.Nf_cycles<=0).any():raise ValueError('寿命或停止循环数必须为有限正数')
    obs['is_runout']=parse_bool(obs.is_runout,'is_runout')
    # Prediction covariates are authoritative; only observed endpoint and documentation are joined.
    keep=['specimen_id','Nf_cycles','is_runout']+[c for c in ['stop_reason','result_date'] if c in obs]
    joined=pred.merge(obs[keep],on='specimen_id',validate='one_to_one')
    joined['is_exact']=~joined.is_runout;joined['logN']=np.log10(joined.Nf_cycles)
    joined['row_nll']=review_row_losses(joined.logN,joined.pred_logN,joined.scale,joined.is_exact)
    joined['exact_error_logN']=np.where(joined.is_exact,joined.pred_logN-joined.logN,np.nan)
    for kind in ('empirical','prob'):
        for level in (80,90):
            joined[f'{kind}_covered{level}_exact_only']=np.where(joined.is_exact,
                (joined.logN>=joined[f'{kind}_lower{level}_logN'])&(joined.logN<=joined[f'{kind}_upper{level}_logN']),np.nan)
    summary=[]
    for aggregation in ('record_equal','campaign_equal'):
        w=np.ones(len(joined)) if aggregation=='record_equal' else 1/joined.groupby('campaign_id').specimen_id.transform('size').to_numpy(float)
        w=w/w.sum();exact=joined.loc[joined.is_exact].copy()
        row=dict(aggregation=aggregation,n_evaluated=len(joined),n_pending=len(pred)-len(joined),
            n_campaigns=joined.campaign_id.nunique(),n_exact=len(exact),n_runout=int(joined.is_runout.sum()),
            joint_NLL=float(np.dot(w,joined.row_nll)),
            coverage_scope='exact failures only; runout latent-life coverage unknown')
        if len(exact):
            ew=np.ones(len(exact)) if aggregation=='record_equal' else 1/exact.groupby('campaign_id').specimen_id.transform('size').to_numpy(float)
            ew=ew/ew.sum();err=exact.exact_error_logN.to_numpy(float);y=exact.logN.to_numpy(float)
            sse=float(np.dot(ew,err**2));sst=float(np.dot(ew,(y-np.dot(ew,y))**2))
            row.update(RMSE=sse**.5,MAE=float(np.dot(ew,np.abs(err))),R2=1-sse/sst if sst>0 else np.nan,
                F5=float(np.dot(ew,np.abs(err)<=LOG5)))
            for kind in ('empirical','prob'):
                for level in (80,90):row[f'{kind}_coverage{level}_exact_only']=float(np.dot(ew,exact[f'{kind}_covered{level}_exact_only']))
        summary.append(row)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    review_save(joined,out,'independent_evaluation_rows');review_save(pd.DataFrame(summary),out,'independent_evaluation_summary')
    revision_json(out/'evaluation_manifest.json',dict(model_id=REVISION_MODEL_ID,
        predictions_sha256=review_sha(predictions),outcomes_sha256=review_sha(outcomes),
        model_refitted=False,n_pending=len(pred)-len(joined),
        limitation='No automatic certification of blinded timing, censoring independence or engineering design reliability.'))
    print(f'独立实验评价完成，未训练：{out}',flush=True)


def revision_external_cli(args,parser):
    if args.predict and args.evaluate:parser.error('--predict与--evaluate不可同时使用')
    if args.check or args.reviewer_only or args.baseline or args.data or args.censor_evidence:parser.error('独立预测/评价不能与主分析参数混用')
    if args.predict and (not args.model or args.predictions):parser.error('--predict须配合--model')
    if args.evaluate and (not args.predictions or args.model):parser.error('--evaluate须配合--predictions')
    out=(args.out or Path(__file__).resolve().parent/'runs'/('predict_' if args.predict else 'evaluate_')/
         datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if out.exists() and any(out.iterdir()):parser.error('为保护预先保存的预测，请指定新的空输出目录')
    if args.predict:revision_write_predictions(args.model,args.predict,out)
    else:revision_evaluate(args.predictions,args.evaluate,out)


PROSPECTIVE_GUIDE = '''独立实验使用说明

本目录的模型由原候选池和原内层选择规则，在223条文献记录上最终拟合。
它与主分析的五折泛化评价不同。新增的最终拟合不用于改写五折结果。
区间由选中配置的开发集OOF校准，选型与校准共用开发数据，不能称为独立验证或工程设计下限。
环境倍率在训练内估计，物理主干有水氧损伤不弱于空气的方向约束；残差仍可能逆转最终预测顺序。

1. 实验前：复制new_conditions_TEMPLATE.csv并填写，不填寿命和runout结果。
architecture填写详细编织方式（与训练表原architecture字段一致），arch_group填粗类别（如2D、2.5D、3D）。
env_class只能填写air_or_oxygen、water_oxygen、vacuum。T_C单位℃，应力/UTS为MPa，频率为Hz。
UTS应来自独立拉伸表征并记录采用的温度/批次依据，不能从本次疲劳结果反推。
新specimen_id、campaign_id必须非空。记录涂层、材料批次、波形、分压、气流和停止规则。
下面命令路径请换成你的实际路径；保持同一Python依赖环境，仅加载自己生成的模型。
python WB_PIML_revision_single.py --predict "新实验条件.csv" --model "本目录/frozen_model.joblib" --out "新预测目录"
保留生成的predictions.csv及prediction_manifest.json。不要覆盖。先保存预测，再获得实验结果。
本地时间戳和哈希便于核对，但不是第三方的盲测时间证明。

2. 实验后：复制new_outcomes_TEMPLATE.csv，填specimen_id、Nf_cycles、is_runout、停止原因和日期。
断裂填False；到停止循环数仍未断填True，Nf_cycles填实际停止循环数。
python WB_PIML_revision_single.py --evaluate "新实验结果.csv" --predictions "新预测目录/predictions.csv" --out "新评价目录"
程序不训练或调参，支持部分完成试样，并报告尚未完成数量。
runout只进入生存评分，不进入断裂误差，也不冒充双侧预测区间覆盖率。

3. 试验分工：空气试验检验新材料批次上的温度/频率/应力预测，不能标定水氧倍率。
空气—水氧差异需要同批材料、相同温度/频率/R/应力的对照，并记录气氛条件。
1300℃、3Hz、R=0.1可作为衔接已有水氧数据的候选条件，不是强制试验方案。
多个应力级、重复件及跨温度/批次数据用于区分离散性和环境效应；样本量另按资源和目标精度确定。
如果用新实验重新标定参数，必须更新模型版本，并另留独立试验组评价，不能继续称其为原模型盲测。
当前覆盖率不构成工程设计资格；需跨批次独立验证、覆盖率不确定性及目标可靠度下的验证。
'''


# ==================== 训练内水氧倍率估计与可辨识性诊断 ====================
def water_mask(frame):
    return frame.env_class.astype(str).str.lower().str.contains('water|steam',regex=True).to_numpy(bool)


def water_multiplier_value(beta):
    # Keep log10 beta as the authoritative number if a ratio cannot be represented.
    beta=float(beta)
    return float(10.**beta) if np.isfinite(beta) and beta<=300 else None


def water_training_policy(train):
    requested=bool(train['fit_water_multiplier'].iloc[0]) if 'fit_water_multiplier' in train else False
    if 'fit_water_multiplier' in train and train.fit_water_multiplier.nunique()!=1:
        raise ValueError('A training fit must use one water-parameter policy')
    wet=water_mask(train)
    air=train.env_class.astype(str).str.lower().isin(['air_or_oxygen','air','oxygen']).to_numpy(bool)
    active_channel=train.environment_exposure.to_numpy(float)>0
    wet=wet&active_channel;air=air&active_channel
    upper=None
    if 'water_log10_upper' in train:
        bounds=train.water_log10_upper.dropna().unique()
        if len(bounds)>1:raise ValueError('Mixed computational bounds within a training fit')
        if len(bounds):upper=float(bounds[0])
    active=requested and bool(wet.any()) and bool(air.any())
    status=('estimated_training_only_identifiability_not_guaranteed' if active else
            'unidentified_no_water_reference_beta0' if requested and not wet.any() else
            'unidentified_no_air_reference_beta0' if requested and not air.any() else 'fixed_weight_control')
    return dict(requested=requested,active=active,status=status,upper=upper,
        n_water=int(wet.sum()),n_air=int(air.sum()),
        n_water_campaigns=int(train.loc[wet,'campaign_id'].nunique()) if 'campaign_id' in train else 0,
        n_air_campaigns=int(train.loc[air,'campaign_id'].nunique()) if 'campaign_id' in train else 0)


def water_log_adjustment(frame,coef):
    if len(coef)<6:return np.zeros(len(frame))
    enabled=(frame.fit_water_multiplier.to_numpy(bool) if 'fit_water_multiplier' in frame else np.ones(len(frame),bool))
    return np.where(water_mask(frame)&enabled,float(coef[5]),0.)


def water_fit_metadata(train,coef):
    policy=water_training_policy(train)
    beta=float(coef[5]) if len(coef)==6 else 0.
    if not policy['requested']:
        wet=water_mask(train)
        fixed=float(train.loc[wet,'environment_exposure'].iloc[0]) if wet.any() else None
        multiplier=fixed
    else:multiplier=water_multiplier_value(beta)
    return dict(water_parameter_status=policy['status'],water_parameter_estimated=policy['active'],
        water_log10_multiplier=beta if policy['requested'] else None,water_multiplier=multiplier,
        water_training_records=policy['n_water'],water_training_campaigns=policy['n_water_campaigns'],
        air_training_records=policy['n_air'],air_training_campaigns=policy['n_air_campaigns'],
        water_lower_boundary=bool(policy['active'] and beta<=1e-6),
        water_log10_computational_upper=policy['upper'],
        water_upper_boundary=bool(policy['active'] and policy['upper'] is not None and beta>=policy['upper']-1e-6),
        water_outer_responses_used=False)


def water_fit_competing(train,gamma,config,classic,predictor,nll_function,decrement_function,
                        likelihood_options=None,fixed_beta=None,return_diagnostics=False,
                        penalty_free=False):
    """Estimate a nonnegative log10 water multiplier on the supplied training rows only.

    No upper physical bound or new beta penalty is imposed in the main fit.
    Existing physical bounds/ridge/environmental penalty are retained. Optional
    caps and penalty-free profiles are diagnostics, never winner selection.
    """
    policy=water_training_policy(train)
    if fixed_beta is not None and (not np.isfinite(fixed_beta) or fixed_beta<0):
        raise ValueError('Fixed profile beta must be finite and nonnegative')
    free_beta=policy['active'] and fixed_beta is None
    emit_beta=policy['requested'] or fixed_beta is not None
    ca,cb=float(classic[0]),float(classic[1])
    bounds=[(ca-config.intercept_half_width,ca+config.intercept_half_width),
        (max(config.walker_min,cb-config.walker_half_width),min(config.walker_max,cb+config.walker_half_width)),
        (config.environment_ratio_min,config.environment_ratio_max),
        (config.environment_stress_min,config.environment_stress_max),(config.arrhenius_min,config.arrhenius_max)]
    if free_beta:bounds.append((0.,policy['upper']))
    bounds.append((math.log(.08),math.log(4.)))
    def decode(parameter):
        coef=np.asarray(parameter[:5],float)
        if emit_beta:
            beta=parameter[5] if free_beta else 0. if fixed_beta is None else fixed_beta
            coef=np.r_[coef,beta]
        return coef,float(np.exp(parameter[-1]))
    def components(parameter):
        coef,sigma=decode(parameter)
        mu=predictor(train,coef,gamma)
        if not np.isfinite(mu).all():return 1e100,0.,0.
        nll=float(nll_function(train,mu,sigma,**(likelihood_options or {}))['joint_censored_NLL'])
        ridge=0. if penalty_free else float(config.ridge_strength)*((coef[0]-ca)**2+.1*(coef[1]-cb)**2+.01*coef[3]**2+.01*coef[4]**2)
        env=0. if penalty_free or config.environment_strength==0 else float(config.environment_strength)*float(np.mean(decrement_function(train,coef,gamma)**2))
        return nll,ridge,env
    def loss(parameter):
        values=components(parameter);total=sum(values)
        return total if np.isfinite(total) else 1e100
    starts=[]
    for k,m,q,sigma,beta in [(-2.,1.,4.,.8,0.),(-4.,2.,7.,1.2,math.log10(1.5)),(0.,4.,2.,1.,math.log10(5.))]:
        start=[ca,cb,k,m,q]+([beta] if free_beta else [])+[math.log(sigma)]
        starts.append(np.clip(start,[b[0] if b[0] is not None else -np.inf for b in bounds],
                                   [b[1] if b[1] is not None else np.inf for b in bounds]))
    results=[minimize(loss,x,method='L-BFGS-B',bounds=bounds,
        options={'maxiter':1200,'ftol':1e-12,'gtol':1e-8}) for x in starts]
    valid=[r for r in results if r.success and np.isfinite(r.fun) and r.fun<1e99]
    if not valid:
        # Same objective/bounds; no loosening of constraints or label changes.
        finite=[r for r in results if np.isfinite(r.fun) and np.isfinite(r.x).all()]
        start=min(finite,key=lambda r:r.fun).x if finite else starts[0]
        retry=minimize(loss,start,method='Powell',bounds=bounds,
            options={'maxiter':6000,'xtol':1e-9,'ftol':1e-10})
        if retry.success and np.isfinite(retry.fun) and retry.fun<1e99:valid=[retry]
        else:raise RuntimeError('Water-enhancement fit did not converge: '+'; '.join(str(r.message) for r in results)+f'; retry={retry.message}')
    best=min(valid,key=lambda r:float(r.fun));coef,sigma=decode(best.x)
    if return_diagnostics:
        nll,ridge,env=components(best.x)
        return dict(coefficients=coef,sigma=sigma,objective=float(best.fun),mean_NLL=nll,
            ridge=ridge,environment_penalty=env,free_water_parameter=free_beta,
            policy=policy,successful_starts=len(valid),start_objective_spread=float(max(r.fun for r in valid)-min(r.fun for r in valid)))
    return coef


def water_order_audit(state,test):
    """Counterfactual covariates only: check trunk direction and report residual reversals.

    These synthetic environment switches are not observed environmental contrasts.
    """
    air=test.copy();air['env_class']='air_or_oxygen';air=apply_physics_drivers(air)
    wet=test.copy();wet['env_class']='water_oxygen';wet=apply_physics_drivers(wet)
    a=revision_predict_state(state,air);w=revision_predict_state(state,wet)
    pa=predict_physical(air,COMPETING_WB_SPEC,state['coefficients'],state['gamma'])
    pw=predict_physical(wet,COMPETING_WB_SPEC,state['coefficients'],state['gamma'])
    if np.any(pw>pa+1e-10):raise AssertionError('Constrained trunk reverses water/air direction')
    return [dict(row_id=int(row.row_id),campaign_id=str(row.campaign_id),
        water_minus_air_trunk_logN=float(pw[i]-pa[i]),water_minus_air_total_logN=float(w[i]-a[i]),
        total_order_reversal=bool(w[i]>a[i]+1e-10),
        interpretation='counterfactual covariates; no observed paired lifetimes; residual not constrained')
        for i,row in test.reset_index(drop=True).iterrows()]


def water_profile_one(df,result):
    """Five outer TRAINING subsets only; response-free test rows identify partition.

    Profiles reoptimize nuisance physics and scale, not the residual learner.
    Penalized objective profiles are not converted into chi-square confidence
    intervals. Unpenalized diagnostic profiles are also supplied, conditional
    on the existing nuisance bounds. Neither proves causal identifiability.
    """
    held={p['row_id'] for p in result['predictions']}
    train=df.loc[~df.row_id.isin(held)].reset_index(drop=True)
    policy=water_training_policy(train);gamma=float(result['settings']['gamma'])
    config=replace(DEFAULT_COMPETING_CONFIG,environment_strength=float(result['settings']['strength']))
    rows=[]
    if not policy['active']:
        return [dict(fold=result['fold'],status=policy['status'],outer_test_responses_used=False,confidence_interval_claim=False)]
    classic=_fit_linear_physical(train,CLASSIC_WB_SPEC,gamma)
    estimated_beta=float(result['coefficients'][5])
    # Log-spaced diagnostic span, not a fitted search range or physical bounds.
    grid=sorted(set([0.,math.log10(1.25),math.log10(1.5),math.log10(2.),math.log10(3.),
                     math.log10(5.),1.,math.log10(30.),2.,estimated_beta]))
    for penalty_free in (False,True):
        reference=water_fit_competing(train,gamma,config,classic,_predict_competing_damage,
            joint_aft_nll,environmental_decrement,return_diagnostics=True,penalty_free=penalty_free)
        reference_beta=float(reference['coefficients'][5])
        for beta in sorted(set(grid+[reference_beta])):
            fitted=water_fit_competing(train,gamma,config,classic,_predict_competing_damage,
                joint_aft_nll,environmental_decrement,fixed_beta=beta,
                return_diagnostics=True,penalty_free=penalty_free)
            rows.append(dict(fold=result['fold'],profile_type='unpenalized_likelihood_with_existing_bounds' if penalty_free else 'original_penalized_objective',
                status='computed',beta=beta,water_multiplier=water_multiplier_value(beta),
                objective=fitted['objective'],mean_NLL=fitted['mean_NLL'],sigma=fitted['sigma'],
                reference_objective=reference['objective'],reference_beta=reference_beta,
                delta_mean_objective=fitted['objective']-reference['objective'],
                fixed_profile_better_than_free_fit=fitted['objective']<reference['objective']-1e-6,
                n_training=len(train),n_water=policy['n_water'],n_water_campaigns=policy['n_water_campaigns'],
                nuisance_log10_K=float(fitted['coefficients'][2]),nuisance_temperature=float(fitted['coefficients'][4]),
                nuisance_stress=float(fitted['coefficients'][3]),outer_test_responses_used=False,
                confidence_interval_claim=False,scope='conditional diagnostic; finite scan not a physical upper bound'))
    return rows


def water_analysis_diagnostics(df,results,out):
    if not any(r['variant']=='selected_pipeline' for r in results):return
    audit=[];orders=[]
    for r in results:
        audit.append(dict(variant=r['variant'],fold=r['fold'],**r['water_fit']))
        for row in r.get('environment_order_audit',[]):orders.append(dict(fold=r['fold'],**row))
    review_save(pd.DataFrame(audit),out,'water_parameter_by_fold')
    inner=[dict(variant=r['variant'],outer_fold=r['fold'],inner_fold=a['inner_fold'],**a['water_fit'])
           for r in results for a in r['audit']]
    review_save(pd.DataFrame(inner),out,'water_parameter_inner_folds')
    order=pd.DataFrame(orders);review_save(order,out,'water_air_counterfactual_order')
    summary=order.groupby('fold').agg(n_conditions=('row_id','size'),
        total_order_reversals=('total_order_reversal','sum'),
        maximum_trunk_water_minus_air=('water_minus_air_trunk_logN','max'),
        maximum_total_water_minus_air=('water_minus_air_total_logN','max')).reset_index()
    review_save(summary,out,'water_air_order_summary')
    reference=[r for r in results if r['variant']=='selected_pipeline']
    print('水氧参数诊断：只使用5个外层训练集，剖面拟合不参与主模型选择...',flush=True)
    with parallel_config(backend='loky',inner_max_num_threads=1):
        parts=Parallel(n_jobs=min(WORKERS,len(reference)),return_as='generator')(
            delayed(water_profile_one)(df,r) for r in reference)
        profiles=[]
        for part in parts:
            profiles.extend(part);print(f'水氧剖面完成：外层折 {part[0]["fold"]}',flush=True)
    review_save(pd.DataFrame(profiles),out,'water_identifiability_profiles')
    a=pd.DataFrame(audit);base=a.loc[a.variant.eq('selected_pipeline')].set_index('fold')
    bounds=[]
    for n in ('water_cap_10','water_cap_100'):
        for _,row in a.loc[a.variant.eq(n)].iterrows():
            b=base.loc[row['fold']]
            bounds.append(dict(variant=n,fold=row['fold'],unbounded_beta=b.water_log10_multiplier,
                capped_beta=row.water_log10_multiplier,
                beta_change=row.water_log10_multiplier-b.water_log10_multiplier,
                cap_reached=row.water_upper_boundary,scope='computational cap dependency, not parameter selection'))
    review_save(pd.DataFrame(bounds),out,'water_computational_bound_sensitivity')
    (Path(out)/'水氧参数诊断说明.txt').write_text(WATER_DIAGNOSTIC_GUIDE,encoding='utf-8')


def water_verify_outputs(out,df):
    a=pd.read_csv(Path(out)/'water_parameter_by_fold.csv')
    selected=a.loc[a.variant.eq('selected_pipeline')]
    if len(selected)!=5 or selected.water_log10_multiplier.isna().any():raise AssertionError('Missing learned water estimates')
    if selected.water_log10_multiplier.lt(-1e-10).any():raise AssertionError('Water direction constraint failed')
    if a.water_outer_responses_used.any():raise AssertionError('Water estimation used outer responses')
    profile=pd.read_csv(Path(out)/'water_identifiability_profiles.csv')
    if profile.fold.nunique()!=5 or profile.outer_test_responses_used.any():raise AssertionError('Profile partition mismatch')
    if profile.confidence_interval_claim.any():raise AssertionError('Profiles must not claim unsupported confidence intervals')
    order=pd.read_csv(Path(out)/'water_air_counterfactual_order.csv')
    if set(order.row_id)!=set(df.row_id) or not order.row_id.is_unique:raise AssertionError('Counterfactual audit incomplete')
    if order.water_minus_air_trunk_logN.gt(1e-10).any():raise AssertionError('Trunk order violation')
    # Do not fail merely because unconstrained residuals reverse the total order: report this scientific limitation.


WATER_DIAGNOSTIC_GUIDE = '''水氧系数说明
主模型：E(vacuum)=0，E(air)=1，E(water)=10**beta，beta>=0。
它与exp(delta)等价（delta=ln(10)*beta），只增加一个物理参数，不扩大残差候选池。
该倍率是相同其他工况下水氧/空气环境损伤通道的比值，不是总损伤比或寿命缩短倍数。
air_or_oxygen是原数据库合并类别，当前编码不能单独识别空气与纯氧的差异。
beta在每个对应训练子集内与原物理参数及删失尺度一起拟合；外层寿命不用于选择或拟合。
没有新增beta正则化，也不假定1.5为中心先验。原有物理参数约束和惩罚仍保留。
主拟合beta无有限上界。10和100分别代表一个、两个数量级的额外计算域对照，不是物理边界。
水氧训练样本/空气参考缺失时，beta不可估计，固定为0作为明确的参考假设，绝不称作学得倍率。
包括无水氧训练数据的LOEO水氧留出，必须结合情景分析解释，不能称作学会了水氧外推。

water_parameter_by_fold.csv：每折估计及训练支持、边界状态。
water_identifiability_profiles.csv：只使用外层训练集，固定beta后重拟合其他物理系数和尺度。
同时报告原惩罚目标剖面，以及保留原干扰参数边界但去掉惩罚的似然剖面；不重新拟合残差。
扫描点只是诊断坐标，不是主模型可选权重。平坦或广阔的低目标区域提示弱辨识；不自动换算95%区间。
若fixed_profile_better_than_free_fit为True，说明某固定点找到了更低目标，需要关注局部极值；程序不偷偷替换主模型。
水氧只有两个试验组且温度/来源混杂，折间离散性和剖面不能证明独立因果环境效应。
water_computational_bound_sensitivity.csv：限制计算域后beta变化及触顶情况。
water_air_counterfactual_order.csv：同一协变量下改变环境类别的情景检查，不是真实配对试验。
物理主干保证水氧寿命不长于空气（正的环境通道下）；无约束残差可能逆转最终预测顺序。
出现逆转会如实输出，不能声称完整模型满足全局环境单调性，也不能裁剪结果以掩盖问题。

固定1、1.5及其他旧权重对照全部保留，不挑外层分数最好的倍率。
估计等于1不等于证明真实损伤相同；估计很大或依赖上界，也不等于发现巨大真实损伤。
独立的同批材料、同温度/应力/频率/R气氛对照才是校准环境倍率的后续证据。
'''


# ==================== 严格因素留出及返修证据审计（不更改主模型） ====================
STRICT_REVIEW_VERSION='strict-campaign-holdout-20260928-v1'
STRICT_MODELS={'WB-PIML','WB','ExtraTrees','ET-Walker','ML-Ens','ML-Strong',*SIMPLE_CONTROL_MODELS}


def strict_holdout_plan(df,overlap):
    """Retain original test sets/seeds; exclude their entire campaigns from training.

    Four inner campaign folds are required for this additional diagnostic.
    Infeasible tasks are disclosed, not rescued with row-wise validation.
    """
    columns={'LOSO':'source_id','LOAO':'architecture','LOTO':'temp_bin','LOEO':'env_class'}
    rows=[]
    for task_index,rec in enumerate(overlap.loc[overlap.scenario.isin(columns)].to_dict('records'),1):
        scenario=rec['scenario'];column=columns[scenario];level=str(rec['holdout_group'])
        te=np.flatnonzero(df[column].astype(str).eq(level).to_numpy())
        original=np.flatnonzero(~df[column].astype(str).eq(level).to_numpy())
        excluded=set(df.iloc[te].campaign_id)
        tr=np.array([i for i in original if df.iloc[i].campaign_id not in excluded],dtype=int)
        train=df.iloc[tr].reset_index(drop=True);test=df.iloc[te].reset_index(drop=True)
        review_require_disjoint(train,test)
        seed=SEED+700000+1000000*list(columns).index(scenario)
        selection_seed=seed+10000*int(rec['fold'])+500
        reason='';inner=[]
        if train.campaign_id.nunique()<4:reason='fewer_than_four_training_campaigns'
        elif not train.is_exact.any() or not test.is_exact.any():reason='no_exact_failures_for_existing_protocol'
        else:
            try:
                splits,_=grouped_folds(train,4,selection_seed)
                for k,(a,b) in enumerate(splits,1):
                    review_require_disjoint(train.iloc[a],train.iloc[b])
                    inner.append(dict(inner_fold=k,fit_ids=train.iloc[a].row_id.astype(int).tolist(),
                        validation_ids=train.iloc[b].row_id.astype(int).tolist()))
            except (RuntimeError,ValueError) as exc:reason='inner_partition_not_feasible: '+str(exc)
        # Unchanged partitions may be reused only if the same four-fold protocol is feasible.
        action='not_estimable' if reason else 'refit' if len(tr)!=len(original) else 'reuse_identical_partition'
        rows.append(dict(task_id=f'{scenario}_{int(rec["fold"]):02d}',scenario=scenario,
            fold=int(rec['fold']),holdout_group=level,holdout_column=column,seed=seed,
            action=action,reason=reason,n_train_original=len(original),n_train=len(tr),n_test=len(te),
            n_train_campaigns=int(train.campaign_id.nunique()),n_removed=len(original)-len(tr),
            train_indices=tr.tolist(),test_indices=te.tolist(),train_ids=train.row_id.astype(int).tolist(),
            test_ids=test.row_id.astype(int).tolist(),inner_partitions=inner,
            train_campaigns=sorted(train.campaign_id.unique().tolist()),test_campaigns=sorted(test.campaign_id.unique().tolist())))
    return rows


def strict_holdout_task(df,task,trees):
    import traceback
    try:
        tr=np.asarray(task['train_indices'],int);te=np.asarray(task['test_indices'],int)
        review_require_disjoint(df.iloc[tr],df.iloc[te])
        with threadpool_limits(limits=1):
            point,choices,prob,cal=_external_fold_task(df,task['fold'],tr,te,task['holdout_group'],
                task['holdout_column'],task['seed'],trees,task['scenario'],1)
        return dict(task_id=task['task_id'],status='refitted',point=pd.concat(point,ignore_index=True),
            selection=pd.DataFrame(choices),probability=pd.concat(prob,ignore_index=True),
            calibration=pd.concat(cal,ignore_index=True),error='')
    except Exception:
        # Do not silently treat numerical failure as data insufficiency or valid performance.
        return dict(task_id=task['task_id'],status='fit_failed',error=traceback.format_exc())


def strict_scoring_frame(frame):
    d=frame.copy()
    d['row_nll']=distribution_nll(d.prob_family,d.logN,d.prob_location,d.prob_scale,d.is_exact)
    for level in (80,90):
        tail=(1-level/100)/2
        d[f'prob_lower{level}']=distribution_quantile(d.prob_family,d.prob_location,d.prob_scale,tail)
        d[f'prob_upper{level}']=distribution_quantile(d.prob_family,d.prob_location,d.prob_scale,1-tail)
    return d


def strict_accept_result(df,task,result):
    p=result['probability'];c=result['calibration'];s=result['selection']
    if set(p.model)!=STRICT_MODELS:raise AssertionError('Strict comparison model coverage differs')
    train=set(task['train_ids']);test=set(task['test_ids'])
    if not set(c.row_id).issubset(train) or set(c.row_id)&test:
        raise AssertionError('Strict interval calibration includes held-out rows')
    if not STRICT_MODELS.issubset(set(c.model)):raise AssertionError('Strict calibration models missing')
    if c.outer_test_used_for_calibration.astype(bool).any():raise AssertionError('Strict calibration admits outer responses')
    for _,part in c.groupby('model'):
        if part.inner_fold.nunique()!=4:raise AssertionError('Strict calibration must retain four campaign folds')
        if set(part.row_id)!=train or part.row_id.duplicated().any():
            raise AssertionError('Strict calibration does not cover development set exactly once')
    for _,part in p.groupby('model'):
        if set(part.row_id)!=test or part.row_id.duplicated().any():raise AssertionError('Strict prediction coverage mismatch')
    for f in (p,c,s):
        f['strict_task_id']=task['task_id'];f['validation_scope']='factor_and_campaign_disjoint'
        f['strict_result_origin']=result['status'];f['campaign_disjoint_by_design']=True
    s['campaign_overlap']=False;s['holdout_campaign_disjoint']=True;s['n_overlapping_campaigns']=0
    return strict_scoring_frame(p),c,s


def strict_holdout_review(df,tables,out,args):
    out=Path(out);dest=out/'strict_holdout';dest.mkdir(exist_ok=True)
    overlap=pd.read_csv(out/'holdout_overlap.csv')
    plan=strict_holdout_plan(df,overlap);revision_json(dest/'partition_plan.json',plan)
    public=[{k:v for k,v in t.items() if k not in {'train_indices','test_indices','inner_partitions','train_ids','test_ids','train_campaigns','test_campaigns'}} for t in plan]
    review_save(pd.DataFrame(public),dest,'task_plan')
    indexed={t['task_id']:t for t in plan};results=[];status=[]
    for t in plan:
        if t['action']=='not_estimable':
            status.append(dict(task_id=t['task_id'],status='not_estimable',reason=t['reason']));continue
        if t['action']=='reuse_identical_partition':
            def subset(key):
                f=tables[key];return f.loc[f.scenario.eq(t['scenario']) & f.fold.eq(t['fold'])].copy()
            results.append(dict(task_id=t['task_id'],status='reused_identical_partition',
                probability=subset('Prob_Predictions'),calibration=subset('Prob_Calibration'),
                selection=subset('External_Selection'),error=''))
    todo=[t for t in plan if t['action']=='refit']
    print(f'严格因素留出：{len(todo)}项重新训练；{len(results)}项原划分已隔离；{len(status)}项条件不足。',flush=True)
    if todo:
        with parallel_config(backend='loky',inner_max_num_threads=1):
            stream=Parallel(n_jobs=min(args.jobs,len(todo)),return_as='generator')(
                delayed(strict_holdout_task)(df,t,args.trees) for t in todo)
            for res in stream:
                results.append(res);print(f'严格留出 {res["task_id"]}: {res["status"]}',flush=True)
    predictions=[];calibrations=[];selections=[]
    for res in results:
        t=indexed[res['task_id']]
        if res['status']=='fit_failed':
            status.append(dict(task_id=t['task_id'],status='fit_failed',reason=res['error']))
            (dest/(t['task_id']+'_error.txt')).write_text(res['error'],encoding='utf-8');continue
        p,c,s=strict_accept_result(df,t,res)
        predictions.append(p);calibrations.append(c);selections.append(s)
        status.append(dict(task_id=t['task_id'],status=res['status'],reason=''))
    review_save(pd.DataFrame(status),dest,'task_status')
    if not predictions:raise RuntimeError('No strict holdout task could be evaluated; see task_status.csv')
    pred=pd.concat(predictions,ignore_index=True);cal=pd.concat(calibrations,ignore_index=True)
    review_save(pred,dest,'predictions');review_save(cal,dest,'calibration');review_save(pd.concat(selections,ignore_index=True),dest,'selection')
    metrics=[];comparisons=[];paired=[]
    base=strict_scoring_frame(tables['Prob_Predictions'].loc[lambda x:x.scenario.ne('Primary')])
    for (scenario,model),g in pred.groupby(['scenario','model']):
        b=base.loc[base.scenario.eq(scenario)&base.model.eq(model)&base.row_id.isin(g.row_id)].copy()
        for label,d in [('strict',g),('original_same_test_rows',b)]:
            metrics.append(dict(scenario=scenario,model=model,evaluation=label,n_tasks=d.fold.nunique(),
                population='estimable strict tasks only; excluded tasks remain in task_status',**review_primary_scores(d)))
        for metric in ('SB_RMSE','SB_MAE','SB_R2','joint_NLL'):
            comparisons.append(dict(scenario=scenario,model=model,metric=metric,
                strict_minus_original=review_primary_scores(g)[metric]-review_primary_scores(b)[metric],
                n_test_records=len(g),same_test_population=True))
    for scenario,g in pred.groupby('scenario'):
        a=g.loc[g.model.eq('WB-PIML')]
        for model,b in g.loc[g.model.ne('WB-PIML')].groupby('model'):
            if a.campaign_id.nunique()<2:continue
            ci=review_paired_ci(a,b,args.bootstrap,SEED+7100000+len(paired))
            ci['scenario']=scenario;ci['proposed']='WB-PIML';ci['comparator']=model
            ci['direction']='WB-PIML minus comparator; negative favors WB-PIML for errors/NLL'
            paired.append(ci)
    review_save(pd.DataFrame(metrics),dest,'metrics_matched_population')
    review_save(pd.DataFrame(comparisons),dest,'strict_vs_original')
    review_save(pd.concat(paired,ignore_index=True) if paired else pd.DataFrame(columns=['scenario','metric']),dest,'paired_model_CI')
    groupmetrics=[]
    for (scenario,level,model),g in pred.groupby(['scenario','holdout_group','model']):
        groupmetrics.append(dict(scenario=scenario,holdout_group=level,model=model,**review_primary_scores(g)))
    review_save(pd.DataFrame(groupmetrics),dest,'holdout_metrics')
    # Add both empirical coverage levels, including the 80% LOTO weakness.
    coverage=[]
    for (scenario,model),g in pred.groupby(['scenario','model']):
        exact=g.loc[g.is_exact]
        for level in (80,90):
            lo,hi=f'lower{level}',f'upper{level}'
            if lo not in exact or hi not in exact:continue
            valid=exact[lo].notna()&exact[hi].notna();value=np.nan
            if len(exact) and valid.all():
                w=analysis_weights(exact);w/=w.sum();value=float(np.dot(w,(exact.logN>=exact[lo])&(exact.logN<=exact[hi])))
            coverage.append(dict(scenario=scenario,model=model,nominal=level/100,coverage=value,
                n_exact=len(exact),n_with_interval=int(valid.sum()),scope='exact failures in estimable strict tasks only'))
    review_save(pd.DataFrame(coverage),dest,'empirical_coverage')
    summary=dict(version=STRICT_REVIEW_VERSION,n_tasks=len(plan),n_refitted=sum(s['status']=='refitted' for s in status),
        n_reused=sum(s['status']=='reused_identical_partition' for s in status),
        n_not_estimable=sum(s['status']=='not_estimable' for s in status),n_fit_failed=sum(s['status']=='fit_failed' for s in status),
        main_model_changed=False,selection_and_calibration='remaining training campaigns only',
        claim='Strictness of this split does not establish independent material-system or causal environmental validation.')
    summary['status']='incomplete_numerical_failures' if summary['n_fit_failed'] else 'completed_with_disclosed_ineligible_tasks' if summary['n_not_estimable'] else 'completed'
    revision_json(dest/'summary.json',summary)
    (dest/'使用说明.txt').write_text(STRICT_GUIDE,encoding='utf-8')


def strict_verify_outputs(out,df):
    dest=Path(out)/'strict_holdout';plan=json.loads((dest/'partition_plan.json').read_text(encoding='utf-8'))
    status=pd.read_csv(dest/'task_status.csv').set_index('task_id');p=pd.read_csv(dest/'predictions.csv');c=pd.read_csv(dest/'calibration.csv')
    if set(status.index)!={t['task_id'] for t in plan}:raise AssertionError('Strict tasks silently missing')
    for t in plan:
        train=df.loc[df.row_id.isin(t['train_ids'])];test=df.loc[df.row_id.isin(t['test_ids'])]
        review_require_disjoint(train,test)
        if train[t['holdout_column']].astype(str).eq(t['holdout_group']).any():raise AssertionError('Factor leaked into strict training')
        q=p.loc[p.strict_task_id.eq(t['task_id'])];s=str(status.loc[t['task_id'],'status'])
        if s in ('fit_failed','not_estimable'):
            if len(q):raise AssertionError('Unavailable task has fabricated predictions')
            continue
        for _,g in q.groupby('model'):
            if set(g.row_id)!=set(t['test_ids']) or g.row_id.duplicated().any():raise AssertionError('Strict test coverage incomplete')
        if set(q.model)!=STRICT_MODELS:raise AssertionError('Strict comparators missing')
        cc=c.loc[c.strict_task_id.eq(t['task_id'])]
        if not STRICT_MODELS.issubset(set(cc.model)):raise AssertionError('Strict calibration models missing')
        for _,g in cc.groupby('model'):
            if set(g.row_id)!=set(t['train_ids']) or g.row_id.duplicated().any():raise AssertionError('Strict calibration coverage incorrect')
            if g.inner_fold.nunique()!=4:raise AssertionError('Strict actual calibration is not four-fold')
            for _,v in g.groupby('inner_fold'):
                fitting=train.loc[~train.row_id.isin(v.row_id)]
                valid=train.loc[train.row_id.isin(v.row_id)]
                review_require_disjoint(fitting,valid)
                if not v.fit_row_ids_sha256.eq(probability_row_id_hash(fitting.row_id)).all():raise AssertionError('Strict actual fit trace mismatch')
                if not v.validation_row_ids_sha256.eq(probability_row_id_hash(valid.row_id)).all():raise AssertionError('Strict actual validation trace mismatch')
        for inner in t['inner_partitions']:
            if set(inner['fit_ids'])|set(inner['validation_ids'])!=set(t['train_ids']):raise AssertionError('Strict inner partition incomplete')
            review_require_disjoint(df.loc[df.row_id.isin(inner['fit_ids'])],df.loc[df.row_id.isin(inner['validation_ids'])])
    np.testing.assert_allclose(p.row_nll,distribution_nll(p.prob_family,p.logN,p.prob_location,p.prob_scale,p.is_exact),rtol=1e-10)
    revision_json(dest/'verification.json',dict(status='passed_for_available_tasks',
        n_fit_failed=int(status.status.eq('fit_failed').sum()),n_not_estimable=int(status.status.eq('not_estimable').sum()),
        meaning='Arithmetic and partition checks; not a certificate of predictive or scientific adequacy'))


def revision_censor_evidence(df,out,evidence_path=None):
    out=Path(out);run=df.loc[df.is_runout,['row_id','source_id','campaign_id','Nf_cycles','env_class','T_C']].copy()
    fields=['documented_stop_type','source_page_or_figure','verbatim_stop_rule','evidence_document','evidence_scope','reviewer_note']
    for f in fields:run[f]=''
    run['documented_stop_type']='unreviewed'
    evidence_hash=None
    if evidence_path is not None:
        evidence_path=Path(evidence_path)
        supplied=pd.read_csv(evidence_path).fillna('')
        if 'row_id' not in supplied or supplied.row_id.duplicated().any():raise ValueError('停止证据表需唯一row_id')
        if not set(supplied.row_id).issubset(set(run.row_id)):raise ValueError('停止证据表含非runout记录')
        allowed={'unreviewed','unknown','administrative_cycle_limit','damage_threshold','equipment_or_interruption','other_documented'}
        for _,row in supplied.iterrows():
            i=run.index[run.row_id.eq(row.row_id)][0]
            for identity in ('source_id','campaign_id'):
                if identity in supplied and str(row[identity]) and str(row[identity])!=str(run.loc[i,identity]):raise ValueError('停止证据表来源与记录不一致')
            for f in fields:
                if f in supplied:run.loc[i,f]=str(row[f]).strip()
            if run.loc[i,'documented_stop_type'] not in allowed:raise ValueError('未知停止规则类别；请保留unreviewed或unknown')
        evidence_hash=review_sha(evidence_path)
    complete=run[['source_page_or_figure','verbatim_stop_rule','evidence_document','evidence_scope']].ne('').all(axis=1)
    documented=~run.documented_stop_type.isin(['unreviewed','unknown'])
    run['evidence_fields_complete']=complete
    run['audit_status']=np.where(documented&complete,'documented_pending_source_authenticity_review',
        np.where(documented,'incomplete_evidence',run.documented_stop_type))
    run['conditional_independence_established']=False
    review_save(run,out,'censor_protocol_audit')
    review_save(run.groupby(['source_id','documented_stop_type','audit_status'],dropna=False).size().rename('n_records').reset_index(),out,'censor_evidence_gaps')
    if not (out/'censor_protocol_reviewed_TEMPLATE.csv').exists():review_save(run,out,'censor_protocol_reviewed_TEMPLATE')
    revision_json(out/'censor_evidence_status.json',dict(n_runout=len(run),n_documented_with_fields=int((complete&documented).sum()),
        n_unresolved=int((~(complete&documented)).sum()),evidence_sha256=evidence_hash,
        source_content_automatically_verified=False,conditional_independence_established=False))


def revision_interpretation_audit(df,tables,out):
    out=Path(out);path=out/'sensitivity_predictions.csv'
    lines=['# 返修证据与未完成事项','本报告区分计算完成与科学主张得到支持；不按外层分数改选模型。']
    if path.exists():
        p=pd.read_csv(path);a=p.loc[p.variant.eq('selected_pipeline')];b=p.loc[p.variant.eq('water_1.5')]
        if len(a) and len(b):
            d=a.merge(b[['row_id','pred_logN']],on='row_id',suffixes=('_new','_fixed15'),validate='one_to_one')
            d=d.merge(df[['row_id','env_class']],on='row_id',validate='one_to_one');d=d.loc[d.is_exact].copy()
            d['MSE_gain']=(d.pred_logN_fixed15-d.logN)**2-(d.pred_logN_new-d.logN)**2
            counts=d.groupby('campaign_id').row_id.transform('size');ng=d.campaign_id.nunique()
            d['weighted_MSE_gain']=d.MSE_gain/(counts*ng)
            total=float(d.weighted_MSE_gain.sum());fit=pd.read_csv(out/'water_parameter_by_fold.csv').query("variant=='selected_pipeline'").set_index('fold')
            d['water_training_records']=d['fold'].map(fit.water_training_records)
            d['water_test_without_water_training']=d.env_class.eq('water_oxygen')&d.water_training_records.eq(0)
            d['net_improvement_share']=d.weighted_MSE_gain/total if abs(total)>1e-12 else np.nan
            review_save(d,out,'environment_improvement_attribution_rows')
            summary=d.groupby(['campaign_id','env_class','fold'],dropna=False).agg(n_exact=('row_id','size'),
                mean_MSE_gain=('MSE_gain','mean'),weighted_MSE_gain=('weighted_MSE_gain','sum'),
                net_improvement_share=('net_improvement_share','sum'),water_training_records=('water_training_records','first')).reset_index()
            review_save(summary,out,'environment_improvement_attribution')
            share=float(d.loc[d.water_test_without_water_training,'weighted_MSE_gain'].sum()/total) if abs(total)>1e-12 else np.nan
            lines.append(f'固定1.5条件对照→当前方案：试验组平衡MSE净改善={total:.8f}；无水氧训练支持的水氧测试记录贡献比例={share:.2%}。这是净MSE变化分解，不是RMSE改善率，不是相对所有基线，更不是机制学习证据。')
            rows=[]
            for f in sorted(fit.index):
                a0=json.loads((out/'checkpoints'/f'selected_pipeline_fold{f}.json').read_text(encoding='utf-8'))
                b0=json.loads((out/'checkpoints'/f'water_1.5_fold{f}.json').read_text(encoding='utf-8'))
                rows.append(dict(fold=f,n_water_train=int(fit.loc[f,'water_training_records']),
                    physical_first5_max_difference=float(np.max(np.abs(np.array(a0['coefficients'][:5])-np.array(b0['coefficients'][:5])))),
                    residual_settings_equal=a0['settings']==b0['settings'],
                    interpretation='no water in training: held-out multiplier is a reference assumption, not an estimated water effect' if fit.loc[f,'water_training_records']==0 else 'training estimate subject to confounding'))
            review_save(pd.DataFrame(rows),out,'water_fallback_attribution')
    if (out/'water_air_counterfactual_order.csv').exists():
        q=pd.read_csv(out/'water_air_counterfactual_order.csv')
        lines.append(f'环境情景检查：{int(q.total_order_reversal.sum())}/{len(q)}个条件在完整预测中出现水氧寿命更长；不是实测错误率。环境方向保证只适用于物理主干。')
    ci=tables['Prob_Paired_CI'];ci=ci.loc[ci.scenario.eq('Primary')].copy()
    ci['PIML_minus_comparator']=-ci.estimate;ci['difference_CI95_low']=-ci.CI95_hi;ci['difference_CI95_high']=-ci.CI95_lo
    ci['CI_crosses_zero']=(ci.difference_CI95_low<=0)&(ci.difference_CI95_high>=0)
    review_save(ci,out,'primary_NLL_difference_CI')
    for name in ('WB-CD','WB-Residual','Basquin-Residual'):
        row=ci.loc[ci.comparator.eq(name)].iloc[0]
        lines.append(f'主验证NLL：WB-PIML减{name}={row.PIML_minus_comparator:+.6f}，95% CI [{row.difference_CI95_low:+.6f}, {row.difference_CI95_high:+.6f}]；负值有利于WB-PIML。条件bootstrap不包含反复模型开发不确定性，跨零不证明等效。')
    for _,row in tables['External_PI_Summary'].query("scenario=='LOTO'").iterrows():
        lines.append(f'原温区留出：名义{row.nominal:.0%}经验区间，断裂记录覆盖率{row.coverage:.2%}；不能解释为所有潜在寿命的无条件覆盖率。')
    if (out/'strict_holdout/summary.json').exists():
        v=json.loads((out/'strict_holdout/summary.json').read_text(encoding='utf-8'))
        lines.append(f'严格因素留出：重新拟合{v["n_refitted"]}项，复用相同隔离划分{v["n_reused"]}项，条件不足{v["n_not_estimable"]}项，数值失败{v["n_fit_failed"]}项。严格/原方案仅在相同可评估测试记录上比较，不以减少评价样本冒充性能提升。')
    c=json.loads((out/'censor_evidence_status.json').read_text(encoding='utf-8'))
    lines.append(f'runout停止规则：{c["n_runout"]}条，填写完整证据字段{c["n_documented_with_fields"]}条，未完成{c["n_unresolved"]}条。程序不能认证原文真实性或证明删失独立性；分布/损失权重敏感性不能替代停止机制证据。')
    lines.append('主模型及可学习环境设置保留；本次仅增加审稿验证和证据解释。不会把固定1.5或学习倍率宣布为已验证物理常数。严格留出也不自动等于独立材料系统实验。')
    (out/'返修关键结论与缺口.md').write_text('\n\n'.join(lines),encoding='utf-8')


STRICT_GUIDE='''严格因素留出说明
主模型、权重拟合、候选池和主验证不变。本目录是额外的因素+试验组隔离对照。
先确定原因素测试集，再从训练中排除这些测试试验组的所有记录。
有重叠且可评估任务重新内层选型、拟合并以剩余训练OOF校准区间；原本隔离的相同任务复用保存预测。
所有八个比较模型共用任务和训练/测试划分。不得仅删除旧结果中的重叠记录当成重训。
本额外对照要求至少四个训练试验组，并检查四折内层都有断裂记录；不足时明确不估计。
任务失败单独保存错误，不会自动放宽隔离或切换算法，也不伪造结果。
metrics_matched_population中的原/严格指标使用相同可评估测试记录。汇总不包含不可评估任务，必须一起报告覆盖范围。
外层同源异工况记录被排除后，差异包含训练量减少和材料支持变化，不能解释为纯粹的泄漏偏差估计。
本次验证设计为看到旧结果后的返修分析，不称为预注册。
停止规则证据可使用主脚本同目录或数据同目录的censor_protocol_reviewed.csv；也可--censor-evidence指定。
证据类别：unreviewed、unknown、administrative_cycle_limit、damage_threshold、equipment_or_interruption、other_documented。
必须填写row_id、documented_stop_type、source_page_or_figure、verbatim_stop_rule、evidence_document、evidence_scope。
记录页码和适用样本范围，不能由整百万循环数反推停止协议；文件字段完整并不代表原文已认证。
无证据表仍可完整计算，结果会明确列出未完成的文献证据，不会误报已解决删失独立性。
'''

if __name__=='__main__':
    from multiprocessing import freeze_support
    freeze_support()
    single_main()
