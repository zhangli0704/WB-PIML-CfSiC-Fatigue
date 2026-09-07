"""Temperature-supported residual models and nested candidate selection."""

from __future__ import annotations

import math

from scipy.special import log_ndtr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from .baselines import make_preprocessor
from .config import (
    COMPETING_WB_SPEC, CompetingDamageConfig, DEFAULT_COMPETING_CONFIG, GAMMA, HYBRID_CANDIDATES,
    PRIMARY_GROUP_COLUMN, PhysicalSpec, RESIDUAL_TEMPERATURE_MATCH_ATOL_C, TRAD_CATEGORICAL,
    TRAD_NUMERIC, V23_LOTO_SHORTLIST, V23_NEAR_TIE_RMSE,
)
from .data import grouped_folds, require_exact_training, training_weights
from .metrics import censored_hinge_rmse, sb_metrics
from .physics import calibrate_aft_scale, fit_physical, inner_oof, predict_physical


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
        candidates["inner_SB_RMSE"] <= best_rmse + V23_NEAR_TIE_RMSE
    )
    shortlist = candidates.loc[candidates["within_RMSE_tolerance"]].copy()
    feasible = shortlist.loc[shortlist["censor_feasible"]].copy()
    constraint_relaxed = feasible.empty
    pool = shortlist if constraint_relaxed else feasible
    pool = pool.sort_values(["inner_SB_RMSE", "inner_runout_survival_NLL"]).head(V23_LOTO_SHORTLIST)
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
    chosen["RMSE_tolerance"] = V23_NEAR_TIE_RMSE
    return chosen, stored[int(best["candidate_index"])], candidates
