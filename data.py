"""Analysis settings, data preparation and statistical summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.special import log_ndtr


SEED = 20260824


GAMMA = 0.85


DAMAGE_CAP = 0.40


ARRHENIUS_REFERENCE_C = 800.0


ARRHENIUS_SCALE = 1000.0


LOG5 = math.log10(5.0)


SHEET = "Plot_Data_Verified"


EXPECTED_FILE = "data.xlsx"


EXPECTED_VERIFIED_ROWS = 222


EXPECTED_EXACT_ROWS = 160


EXPECTED_RUNOUT_ROWS = 62


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


PPTX_OUTPUT_NAME = "WB_PIML_figures.pptx"


PRIMARY_GROUP_COLUMN = "campaign_id"


PROXIES = ["Dm", "Di", "Df", "Dox"]


TRAD_NUMERIC = ["logS", "R", "Tn", "logf", "logUTS", "Sa", "Smean"]


TRAD_CATEGORICAL = ["architecture", "env_class"]


NESTED_CANDIDATES = (

    (0.75, 2, 0.65, 0.50),
    (0.75, 3, 0.85, 1.00),
    (0.75, 5, 1.00, 0.50),
    (0.85, 2, 0.85, 1.00),
    (0.85, 3, 0.65, 0.50),
    (0.85, 3, 0.85, 0.00),
    (0.85, 3, 0.85, 0.50),
    (0.85, 3, 0.85, 1.00),
    (0.85, 5, 0.85, 1.00),
    (0.95, 2, 0.85, 1.00),
    (0.95, 3, 0.85, 0.50),
    (0.95, 3, 1.00, 1.00),
)


HYBRID_CANDIDATES = tuple(
    {
        "gamma": gamma,
        "min_samples_leaf": min_leaf,
        "max_features": max_features,
        "eta": eta,
        "residual_mode": "legacy_et",
        "ridge_alpha": 3.0,
        "temperature_scale": 250.0,
    }
    for gamma, min_leaf, max_features, eta in NESTED_CANDIDATES
) + (
    {"gamma": 0.75, "min_samples_leaf": 3, "max_features": 0.85,
     "eta": 0.75, "residual_mode": "dual_support", "ridge_alpha": 3.0,
     "temperature_scale": 180.0},
    {"gamma": 0.85, "min_samples_leaf": 2, "max_features": 0.85,
     "eta": 0.75, "residual_mode": "dual_support", "ridge_alpha": 3.0,
     "temperature_scale": 180.0},
    {"gamma": 0.85, "min_samples_leaf": 3, "max_features": 0.85,
     "eta": 0.50, "residual_mode": "dual_support", "ridge_alpha": 6.0,
     "temperature_scale": 250.0},
    {"gamma": 0.95, "min_samples_leaf": 3, "max_features": 0.85,
     "eta": 0.75, "residual_mode": "dual_support", "ridge_alpha": 6.0,
     "temperature_scale": 250.0},
)


PHYSICS_MIX_GRID = (1.00,)


CALIBRATION_MODES = ("none",)


RMSE_TIE_TOLERANCE = 0.001


NEAR_TIE_RMSE = 0.020


LOTO_SHORTLIST = 3


RESIDUAL_TEMPERATURE_MATCH_ATOL_C = 1e-9


CENSORED_MODELS = ("Lognormal-AFT", "Weibull-AFT")


REPEATED_VALIDATION_SEEDS = tuple(SEED + 101 * index for index in range(10))


@dataclass(frozen=True)
class PhysicalSpec:
    key: str
    channels: tuple[str, ...]
    mode: str = "linear"


@dataclass(frozen=True)
class CompetingDamageConfig:
    """Predeclared weak-physics assumptions used by the competing-damage fit."""

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


DEFAULT_COMPETING_CONFIG = CompetingDamageConfig()


CLASSIC_WB_SPEC = PhysicalSpec("WB", ())


COMPETING_WB_SPEC = PhysicalSpec("WB-CD", (), "competing_damage")


PHYSICAL_SPECS = (
    CLASSIC_WB_SPEC,
    COMPETING_WB_SPEC,
    PhysicalSpec("WB-M", ("M",)),
    PhysicalSpec("WB-Ox", ("Ox",)),
)


EXTERNAL_MODELS = ("RF", "ExtraTrees", "ET-Walker", "GBDT", "SVR", "ML-Ens")


PRIMARY_COMPARATORS = ("ML-Ens", "ExtraTrees", "ET-Walker")


HYBRID_SPECS = (
    ("WB-PIML-Anchor", COMPETING_WB_SPEC),
    ("WB-PIML-M", PhysicalSpec("WB-M", ("M",))),
    ("WB-PIML-Ox", PhysicalSpec("WB-Ox", ("Ox",))),
)


PROPOSED_MODEL = "WB-PIML"


AUDIT_CONTEXT: dict[str, object] = {}


ACTIVE_DATA_PATH: Path | None = None


def parse_bool(series: pd.Series, name: str = "boolean field") -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True, "y": True,
         "false": False, "0": False, "no": False, "n": False}
    )
    if mapped.isna().any():
        raise ValueError(f"{name} contains unrecognized values")
    return mapped.astype(bool)


def resolve_data(argument: str | None) -> Path:
    here = Path(__file__).resolve().parent
    candidates = [] if argument is None else [Path(argument)]
    candidates += [
        here / EXPECTED_FILE,
        Path.cwd() / EXPECTED_FILE,
        here / "upload" / EXPECTED_FILE,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Cannot find {EXPECTED_FILE}; pass --data")


def require_exact_training(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    """Validate a joint exact/right-censored likelihood frame."""
    if frame.empty:
        raise ValueError(f"{context}: empty training frame")
    if "is_runout" not in frame or "is_exact" not in frame:
        raise KeyError(f"{context}: missing exact/runout labels")
    if not frame["is_exact"].eq(~frame["is_runout"]).all():
        raise AssertionError(f"{context}: exact/runout labels are not complementary")
    if not frame["is_exact"].any():
        raise AssertionError(f"{context}: at least one exact failure is required")
    if "sample_weight" in frame.columns:
        supplied = pd.to_numeric(frame["sample_weight"], errors="coerce").to_numpy(float)
        if not np.isfinite(supplied).all() or not np.allclose(supplied, 1.0):
            raise ValueError(f"{context}: every physically unique record must have weight 1")
    return frame


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def apply_physics_drivers(df: pd.DataFrame, water_factor: float = 1.5,
                          arrhenius_reference_c: float = ARRHENIUS_REFERENCE_C,
                          arrhenius_scale: float = ARRHENIUS_SCALE) -> pd.DataFrame:
    """Return a copy with explicit environmental/Arrhenius physics drivers."""
    if water_factor <= 0 or arrhenius_scale <= 0:
        raise ValueError("water_factor and arrhenius_scale must be positive")
    result = df.copy()
    env = result["env_class"].astype(str).str.lower()
    inert = env.str.contains("vacuum|inert|argon", regex=True, na=False)
    water = env.str.contains("water|steam", regex=True, na=False)
    result["environment_exposure"] = np.where(
        inert.to_numpy(bool), 0.0,
        np.where(water.to_numpy(bool), float(water_factor), 1.0),
    )
    temperature_k = result["T_C"].to_numpy(float) + 273.15
    reference_k = float(arrhenius_reference_c) + 273.15
    if reference_k <= 0 or np.any(temperature_k <= 0):
        raise ValueError("Arrhenius temperatures must be above absolute zero")
    result["arrhenius_temperature_drive"] = float(arrhenius_scale) * (
        1.0 / reference_k - 1.0 / temperature_k
    )
    return result


def load_data(path: Path) -> pd.DataFrame:
    if SHEET not in pd.ExcelFile(path).sheet_names:
        raise KeyError(f"Missing sheet {SHEET!r}")
    configure_audit_context(path)
    df = pd.read_excel(path, sheet_name=SHEET).copy()
    required = {
        "row_id", "source_id", "architecture", "arch_group", "env_class", "R", "T_C",
        "frequency_Hz", "Nf_cycles", "UTS_MPa", "stress_level_sigma_max_over_UTS",
        "sigma_max_MPa_before_normalization", "is_runout", "is_exact_failure",
        "record_equivalence_id", "campaign_id", "training_role", "sample_weight",
        "event_observed", "censoring_type", "likelihood_weight", "point_metric_eligible",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    df["row_id"] = pd.to_numeric(df["row_id"], errors="raise").astype(int)
    if df["row_id"].duplicated().any():
        raise ValueError("row_id must be unique")
    df["source_id"] = df["source_id"].fillna("").astype(str).str.strip()
    if df["source_id"].eq("").any():
        raise ValueError("source_id contains blanks")
    df["architecture_detail"] = df["architecture"].fillna("unknown").astype(str).str.strip().str.lower()
    df["architecture"] = (df["arch_group"].fillna(df["architecture_detail"]).fillna("unknown")
                          .astype(str).str.strip().str.lower())
    df["env_class"] = df["env_class"].fillna("unknown").astype(str).str.strip().str.lower()
    if "temp_bin" in df.columns:
        df["temp_bin"] = df["temp_bin"].fillna("unknown").astype(str).str.strip().str.lower()
    else:

        df["temp_bin"] = pd.to_numeric(df["T_C"], errors="coerce").map(lambda value: f"{value:g}c")
    for col in ["R", "T_C", "frequency_Hz", "Nf_cycles", "UTS_MPa",
                "stress_level_sigma_max_over_UTS", "sigma_max_MPa_before_normalization"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric = ["R", "T_C", "frequency_Hz", "Nf_cycles", "UTS_MPa",
               "stress_level_sigma_max_over_UTS", "sigma_max_MPa_before_normalization"]
    if df[numeric].isna().any().any() or not np.isfinite(df[numeric].to_numpy(float)).all():
        raise ValueError("Required numeric columns contain missing/non-numeric values")
    if (df[["frequency_Hz", "Nf_cycles", "UTS_MPa", "stress_level_sigma_max_over_UTS"]] <= 0).any().any():
        raise ValueError("Frequency, life, UTS and normalized stress must be positive")
    if (df["R"] >= 1.0).any():
        raise ValueError("Walker correction requires every stress ratio R < 1")
    df["is_runout"] = parse_bool(df["is_runout"], "is_runout")
    supplied_exact = parse_bool(df["is_exact_failure"], "is_exact_failure")
    mismatch = supplied_exact.ne(~df["is_runout"])
    if mismatch.any():
        bad_rows = df.loc[mismatch, "row_id"].astype(int).tolist()
        raise ValueError(
            "is_exact_failure must equal ~is_runout; mismatched row_id values: "
            f"{bad_rows[:20]}"
        )
    df["supplied_is_runout"] = df["is_runout"].astype(bool)
    df["supplied_is_exact_failure"] = supplied_exact.astype(bool)
    df["label_override_applied"] = False
    df["label_review_status"] = "accepted_from_Plot_Data_Verified"
    if "evidence_source" in df.columns:
        df["label_review_evidence"] = df["evidence_source"].fillna("").astype(str)
    else:
        df["label_review_evidence"] = "verified workbook label retained without code-side override"
    df["reviewed_is_exact_failure"] = ~df["is_runout"]
    df["is_exact"] = ~df["is_runout"]
    df["label_consistent"] = df["reviewed_is_exact_failure"].eq(~df["is_runout"])
    df["training_role"] = df["training_role"].fillna("").astype(str).str.strip()
    df["sample_weight"] = pd.to_numeric(df["sample_weight"], errors="coerce")
    expected_role = np.where(
        df["is_exact"], "exact_likelihood_training", "right_censored_likelihood_training"
    )
    role_mismatch = df["training_role"].ne(expected_role)
    if role_mismatch.any():
        bad = df.loc[role_mismatch, ["row_id", "training_role", "is_runout"]]
        raise ValueError(
            "training_role is inconsistent with the joint-likelihood contract: "
            f"{bad.to_dict('records')[:20]}"
        )
    df["likelihood_weight"] = pd.to_numeric(df["likelihood_weight"], errors="coerce")
    if not np.allclose(df["sample_weight"], 1.0) or not np.allclose(df["likelihood_weight"], 1.0):
        bad = df.loc[
            ~np.isclose(df["sample_weight"], 1.0) | ~np.isclose(df["likelihood_weight"], 1.0),
            ["row_id", "training_role", "sample_weight", "likelihood_weight"],
        ]
        raise ValueError(f"Every unique record must have unit likelihood weight: {bad.to_dict('records')[:20]}")
    event = pd.to_numeric(df["event_observed"], errors="coerce")
    point = pd.to_numeric(df["point_metric_eligible"], errors="coerce")
    expected_event = df["is_exact"].astype(int)
    expected_censoring = np.where(df["is_exact"], "exact", "right_censored")
    if not event.eq(expected_event).all() or not point.eq(expected_event).all():
        raise ValueError("event_observed/point_metric_eligible are inconsistent with exact/runout labels")
    if not df["censoring_type"].astype(str).eq(expected_censoring).all():
        raise ValueError("censoring_type is inconsistent with exact/runout labels")
    if "include_in_verified_model" in df.columns:
        included = parse_bool(df["include_in_verified_model"], "include_in_verified_model")
        if not included.all():
            bad = df.loc[~included, "row_id"].astype(int).tolist()
            raise ValueError(
                "Plot_Data_Verified may contain only retained unique records; "
                f"excluded row_id values were found: {bad[:20]}"
            )
    df["logN"] = np.log10(df["Nf_cycles"].clip(lower=1.0))
    df["S"] = df["stress_level_sigma_max_over_UTS"]
    df["logS"] = np.log10(df["S"].clip(lower=1e-8))
    df["logf"] = np.log10(df["frequency_Hz"].clip(lower=1e-8))
    df["logUTS"] = np.log10(df["UTS_MPa"].clip(lower=1e-8))
    df["Tn"] = (df["T_C"] - 25.0) / 1000.0
    df["Sa"] = 0.5 * df["S"] * np.clip(1.0 - df["R"], 0.0, 2.0)
    df["Smean"] = 0.5 * df["S"] * (1.0 + df["R"])

    df = apply_physics_drivers(df)
    env = df["env_class"].str.lower()
    inert = env.str.contains("vacuum|inert|argon", regex=True, na=False)
    water = env.str.contains("water|steam", regex=True, na=False)
    thermal = sigmoid((df["T_C"].to_numpy(float) - 800.0) / 200.0)
    slow = sigmoid(-df["logf"].to_numpy(float))
    oxidation = (~inert).to_numpy(float) * thermal * (0.5 + 0.5 * slow)
    oxidation = np.maximum(oxidation, water.to_numpy(float) * thermal)
    df["oxidation"] = np.clip(oxidation, 0.0, 1.0)
    stress = np.clip(df["S"].to_numpy(float), 0.0, 1.0)
    df["Dm"] = stress
    df["Di"] = np.clip(stress * (0.60 + 0.40 * df["oxidation"]), 0.0, 1.0)
    df["Df"] = np.clip(stress**2, 0.0, 1.0)
    df["Dox"] = df["oxidation"]
    df["M"] = df[["Dm", "Di", "Df"]].mean(axis=1)
    df["Ox"] = df["Dox"]
    df["Q"] = df[PROXIES].mean(axis=1)
    series_columns = ["source_id", "architecture_detail", "env_class", "T_C", "R", "UTS_MPa"]
    df["series_group_id"] = df[series_columns].astype(str).agg("|".join, axis=1)
    if "raw_Nf" in df.columns:
        raw_nf = df["raw_Nf"].fillna("").astype(str).str.strip().str.lower()
    else:
        raw_nf = pd.Series("", index=df.index)
    round_limit = np.isclose(df["Nf_cycles"], 1e6) | np.isclose(df["Nf_cycles"], 1e7)
    df["qa_exact_round_limit"] = df["is_exact"] & round_limit & ~raw_nf.str.contains("~|未断|runout|>", regex=True)
    df["qa_sstar_gt_1"] = df["S"] > 1.0
    return add_campaign_metadata(df.reset_index(drop=True))


def source_weights(df: pd.DataFrame) -> np.ndarray:
    count = df.groupby("source_id")["source_id"].transform("size").to_numpy(float)
    weight = 1.0 / count
    return weight / np.mean(weight)


def training_weights(df: pd.DataFrame) -> np.ndarray:
    """Return strict unit likelihood weights for all unique records."""
    require_exact_training(df, "training_weights")
    if "sample_weight" in df.columns:
        supplied = pd.to_numeric(df["sample_weight"], errors="coerce").to_numpy(float)
        if not np.isfinite(supplied).all() or not np.allclose(supplied, 1.0):
            bad = df.loc[
                ~np.isclose(supplied, 1.0), ["row_id", "sample_weight"]
            ].to_dict("records")
            raise ValueError(f"Every unique likelihood record must have weight 1: {bad[:20]}")
    return np.ones(len(df), dtype=float)


def analysis_weights(df: pd.DataFrame) -> np.ndarray:
    """Campaign-balanced weights for reported metrics and diagnostics only."""
    required = {"campaign_id", "record_equivalence_id"}
    if not required.issubset(df.columns):
        return source_weights(df)
    frame = df[["campaign_id", "record_equivalence_id"]].astype(str)
    pair_size = frame.groupby(
        ["campaign_id", "record_equivalence_id"]
    )["record_equivalence_id"].transform("size").to_numpy(float)
    equivalence_per_campaign = (
        frame.drop_duplicates()
        .groupby("campaign_id")["record_equivalence_id"].nunique()
    )
    n_equivalence = frame["campaign_id"].map(equivalence_per_campaign).to_numpy(float)
    weight = 1.0 / (pair_size * n_equivalence)
    return weight / np.mean(weight)


def grouped_folds(df: pd.DataFrame, n_splits: int, seed: int):
    """Primary campaign-disjoint folds used by every nested model step."""
    require_exact_training(df, "grouped_folds")
    group_column = PRIMARY_GROUP_COLUMN
    summary = df.groupby(group_column).agg(n=("row_id", "size")).reset_index()
    n_splits = min(n_splits, len(summary))
    rng = np.random.default_rng(seed)
    summary["tie"] = rng.random(len(summary))
    summary = summary.sort_values(["n", "tie"], ascending=[False, True])
    target = float(summary["n"].sum()) / n_splits
    load = np.zeros(n_splits)
    assignment: dict[str, int] = {}
    for _, row in summary.iterrows():
        score = load / max(target, 1.0)
        fold = int(np.argmin(score))
        load[fold] += float(row["n"])
        assignment[str(row[group_column])] = fold
    fold_id = df[group_column].astype(str).map(assignment).to_numpy(int)
    splits = []
    for fold in range(n_splits):
        valid = np.flatnonzero(fold_id == fold)
        train = np.flatnonzero(fold_id != fold)
        if not len(train) or not len(valid):
            raise RuntimeError("Empty grouped fold")
        if set(df.iloc[train][group_column]) & set(df.iloc[valid][group_column]):
            raise AssertionError("Campaign leakage")
        if not df.iloc[train]["is_exact"].any() or not df.iloc[valid]["is_exact"].any():
            raise RuntimeError("Every outer fold must contain exact failures")
        splits.append((train, valid))
    return splits, fold_id + 1


def grouped_folds_by(df: pd.DataFrame, group_column: str, n_splits: int, seed: int):
    require_exact_training(df, f"grouped_folds_by[{group_column}]")
    summary = df.groupby(group_column).agg(n=("row_id", "size")).reset_index()
    n_splits = min(n_splits, len(summary))
    rng = np.random.default_rng(seed)
    summary["tie"] = rng.random(len(summary))
    summary = summary.sort_values(["n", "tie"], ascending=[False, True])
    target = float(summary["n"].sum()) / n_splits
    load = np.zeros(n_splits)
    assignment: dict[str, int] = {}
    for _, row in summary.iterrows():
        score = load / max(target, 1.0)
        fold = int(np.argmin(score))
        load[fold] += float(row["n"])
        assignment[str(row[group_column])] = fold
    fold_id = df[group_column].astype(str).map(assignment).to_numpy(int)
    splits = []
    for fold in range(n_splits):
        valid = np.flatnonzero(fold_id == fold)
        train = np.flatnonzero(fold_id != fold)
        if not df.iloc[train]["is_exact"].any() or not df.iloc[valid]["is_exact"].any():
            raise RuntimeError(f"{group_column}: every fold must contain exact failures")
        if set(df.iloc[train][group_column].astype(str)) & set(df.iloc[valid][group_column].astype(str)):
            raise AssertionError(f"{group_column} leakage")
        splits.append((train, valid))
    return splits, fold_id + 1


def record_folds(df: pd.DataFrame, n_splits: int, seed: int):
    require_exact_training(df, "record_folds")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df), dtype=int)
    rng.shuffle(indices)
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for index, row_index in enumerate(indices):
        buckets[index % n_splits].append(int(row_index))
    fold_id = np.zeros(len(df), int)
    splits = []
    for fold, bucket in enumerate(buckets):
        valid = np.asarray(sorted(bucket), int)
        keep = np.ones(len(df), bool)
        keep[valid] = False
        train = np.flatnonzero(keep)
        fold_id[valid] = fold + 1
        splits.append((train, valid))
    return splits, fold_id


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


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = np.asarray(values, float)[order]
    weights = np.asarray(weights, float)[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(quantile, cumulative, values))


def censored_hinge_rmse(frame: pd.DataFrame, prediction_column: str = "mu") -> float:
    """Campaign/equivalence-balanced one-sided error for right-censored rows."""
    runout = frame.loc[frame["is_runout"]].reset_index(drop=True)
    if not len(runout):
        return 0.0
    violation = np.maximum(
        runout["logN"].to_numpy(float) - runout[prediction_column].to_numpy(float),
        0.0,
    )
    weight = analysis_weights(runout)
    return float(np.sqrt(np.sum(weight * violation**2) / np.sum(weight)))


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


def _prepare_sb_groups(groups, prediction_column="pred_logN"):
    """Prepare a local cache of fixed predictions; no reference results are read."""
    prepared = []
    for group in groups:
        y = group["logN"].to_numpy(float)
        pred = group[prediction_column].to_numpy(float)
        if "record_equivalence_id" in group.columns:
            equivalence_size = group.groupby("record_equivalence_id")[
                "record_equivalence_id"
            ].transform("size").to_numpy(float)
            row_weight = 1.0 / (
                max(group["record_equivalence_id"].nunique(), 1) * equivalence_size
            )
        else:
            row_weight = np.full(len(group), 1.0 / len(group))
        prepared.append((y, pred, row_weight, composite_score(pred - y, y)))
    return prepared


def _aggregate_sb_groups(groups):
    y_parts, p_parts, w_parts, c_parts = [], [], [], []
    for y, pred, row_weight, score in groups:
        y_parts.append(y)
        p_parts.append(pred)
        w_parts.append(row_weight)
        c_parts.append(score)
    y = np.concatenate(y_parts)
    pred = np.concatenate(p_parts)
    weight = np.concatenate(w_parts)
    error = pred - y
    mean_y = float(np.sum(weight * y) / np.sum(weight))
    denominator = float(np.sum(weight * (y - mean_y) ** 2))
    mse = float(np.sum(weight * error**2) / np.sum(weight))
    return {
        "SB_RMSE": math.sqrt(mse),
        "SB_MAE": float(np.sum(weight * np.abs(error)) / np.sum(weight)),
        "SB_R2": float(1.0 - np.sum(weight * error**2) / denominator) if denominator > 0 else np.nan,
        "SB_F5": float(np.sum(weight * (np.abs(error) <= LOG5)) / np.sum(weight)),
        "SB_C_star": float(np.mean(c_parts)),
    }


def sb_metrics(groups: list[pd.DataFrame]) -> dict[str, float]:
    """Return campaign-equal point metrics."""
    return _aggregate_sb_groups(_prepare_sb_groups(groups))


def metric_summary(predictions: pd.DataFrame, bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    exact = predictions.loc[predictions["is_exact"]].copy()
    for model, model_df in exact.groupby("model"):
        groups = [g.reset_index(drop=True) for _, g in model_df.groupby(PRIMARY_GROUP_COLUMN)]
        point = sb_metrics(groups)
        prepared = _prepare_sb_groups(groups)
        boots = {key: [] for key in point}
        for _ in range(bootstrap):
            sampled = [prepared[i] for i in rng.integers(0, len(groups), len(groups))]
            values = _aggregate_sb_groups(sampled)
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
    prepared = {model: _prepare_sb_groups(groups, model) for model in ["WB-PIML", comparator]}

    def effect(indices):
        metrics = {
            model: _aggregate_sb_groups([prepared[model][i] for i in indices])
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
        sampled = rng.integers(0, len(groups), len(groups))
        values = effect(sampled)
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
        prepared = [tuple(g[column].mean() for column in ("covered", "width", "interval_score"))
                    for g in groups]

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
                      + list(CENSORED_MODELS))
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
                loss = -log_ndtr(z)
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
    prepared = {model: _prepare_sb_groups(groups, model) for model in models}

    def rmse(indices, model):
        return _aggregate_sb_groups([prepared[model][i] for i in indices])["SB_RMSE"]

    rng = np.random.default_rng(seed)
    rows = []
    for model in models:
        if model == PROPOSED_MODEL:
            continue
        point = rmse(range(len(groups)), model) - rmse(range(len(groups)), PROPOSED_MODEL)
        draws = []
        for _ in range(bootstrap):
            sampled = rng.integers(0, len(groups), len(groups))
            draws.append(rmse(sampled, model) - rmse(sampled, PROPOSED_MODEL))
        rows.append({"comparison": f"{PROPOSED_MODEL} gain over {model}", "RMSE_gain": point,
                     "CI95_lo": float(np.quantile(draws, 0.025)),
                     "CI95_hi": float(np.quantile(draws, 0.975)),
                     "positive_favors_WB_PIML": True})
    return pd.DataFrame(rows)


def component_ablation_summary(sensitivity: pd.DataFrame) -> pd.DataFrame:
    """Extract genuine retrained component removals from the sensitivity audit."""
    component_variants = [
        "selected_pipeline",
        "classic_WB_trunk",
        "no_residual",
        "no_runout_likelihood_training",
        "no_exact_temperature_gate",
        "relaxed_sign_constraints",
        "unsupported_residual_zero",
    ]
    result = sensitivity.loc[
        sensitivity["variant"].isin(component_variants)
    ].copy()
    order = {name: index for index, name in enumerate(component_variants)}
    result["component_order"] = result["variant"].map(order)
    reference = result.loc[
        result["variant"].eq("selected_pipeline"), "SB_RMSE"
    ]
    reference_rmse = float(reference.iloc[0]) if len(reference) else np.nan
    result["RMSE_increase_vs_selected"] = result["SB_RMSE"] - reference_rmse
    result["interpretation"] = np.where(
        result["variant"].eq("selected_pipeline"),
        "reference: fold-specific development-selected WB-PIML settings",
        "positive RMSE increase means the removed/relaxed component helped the selected pipeline",
    )
    return result.sort_values("component_order").drop(columns="component_order")


def selection_rule_sensitivity(nested_candidates: pd.DataFrame) -> pd.DataFrame:
    """Audit RMSE near-tie and censor-margin rules without fitting new models."""
    candidates = nested_candidates.loc[
        nested_candidates["model"].eq("WB-PIML-Anchor")
    ].copy()
    required = {
        "outer_fold", "candidate_index", "inner_SB_RMSE",
        "inner_runout_survival_NLL", "reference_WB_CD_runout_NLL",
        "runout_noninferiority_margin",
    }
    if not required.issubset(candidates.columns):
        missing = sorted(required - set(candidates.columns))
        raise KeyError(f"Selection-rule sensitivity missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for fold, fold_candidates in candidates.groupby("outer_fold", sort=True):
        best_rmse = float(fold_candidates["inner_SB_RMSE"].min())
        for tolerance in [0.0, 0.01, NEAR_TIE_RMSE, 0.04]:
            near = fold_candidates.loc[
                fold_candidates["inner_SB_RMSE"] <= best_rmse + tolerance + 1e-12
            ].copy()
            for margin_multiplier in [0.0, 1.0, 2.0]:
                feasible = near.loc[
                    near["inner_runout_survival_NLL"]
                    <= near["reference_WB_CD_runout_NLL"]
                    + margin_multiplier * near["runout_noninferiority_margin"]
                    + 1e-12
                ].copy()
                relaxed = feasible.empty
                pool = near if relaxed else feasible
                selected = pool.sort_values([
                    "inner_SB_RMSE", "inner_runout_survival_NLL", "candidate_index"
                ]).iloc[0]
                rows.append({
                    "outer_fold": int(fold),
                    "RMSE_near_tie_tolerance": float(tolerance),
                    "censor_margin_multiplier": float(margin_multiplier),
                    "n_near_tie_candidates": len(near),
                    "n_censor_feasible_candidates": len(feasible),
                    "censor_constraint_relaxed": bool(relaxed),
                    "selected_candidate_index_RMSE_then_NLL": int(
                        selected["candidate_index"]
                    ),
                    "selected_inner_SB_RMSE": float(selected["inner_SB_RMSE"]),
                    "selected_inner_runout_survival_NLL": float(
                        selected["inner_runout_survival_NLL"]
                    ),
                    "development_only_audit": True,
                    "used_to_change_reported_model": False,
                })
    return pd.DataFrame(rows)


def validation_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.loc[predictions["is_exact"]].groupby(
        ["strategy", "repeat", "model"]
    ):
        strategy, repeat, model = keys
        source_groups = [g.reset_index(drop=True) for _, g in group.groupby(PRIMARY_GROUP_COLUMN)]
        rows.append({"strategy": strategy, "repeat": repeat, "model": model,
                     "n_exact": len(group), "n_campaigns": group["campaign_id"].nunique(),
                     "n_sources": group["source_id"].nunique(),
                     **sb_metrics(source_groups)})
    return pd.DataFrame(rows)


def external_paired_effects(predictions: pd.DataFrame, bootstrap: int,
                            seed: int) -> pd.DataFrame:
    """Campaign-bootstrap paired effects within each held-out-domain scenario."""
    tables = []
    scenarios = sorted(predictions["strategy"].astype(str).unique())
    for scenario_index, scenario in enumerate(scenarios):
        scenario_predictions = predictions.loc[
            predictions["strategy"].astype(str).eq(scenario)
        ].copy()
        for comparator_index, comparator in enumerate(PRIMARY_COMPARATORS):
            table = paired_effects(
                scenario_predictions, bootstrap,
                seed + 10000 * scenario_index + 101 * comparator_index,
                comparator=comparator,
            )
            table.insert(0, "scenario", scenario)
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def external_uq_summary(predictions: pd.DataFrame, bootstrap: int,
                        seed: int) -> pd.DataFrame:
    """Evaluate development-calibrated intervals under LOSO/LOAO/LOTO/LOEO."""
    tables = []
    for scenario_index, scenario in enumerate(
        sorted(predictions["strategy"].astype(str).unique())
    ):
        table = uq_summary(
            predictions.loc[predictions["strategy"].astype(str).eq(scenario)].copy(),
            bootstrap, seed + 10000 * scenario_index,
        )
        table.insert(0, "scenario", scenario)
        table["calibration"] = (
            "per-holdout nested-selected development OOF absolute residuals"
        )
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def empirical_interval_halfwidths(oof: pd.DataFrame) -> tuple[float, float]:
    """Campaign-balanced development-OOF interval calibration."""
    exact = oof.loc[oof["is_exact"]].reset_index(drop=True)
    score = np.abs(exact["logN"].to_numpy(float) - exact["mu"].to_numpy(float))
    campaign_size = exact.groupby(PRIMARY_GROUP_COLUMN)[PRIMARY_GROUP_COLUMN].transform("size").to_numpy(float)
    weight = 1.0 / campaign_size
    weight /= np.mean(weight)
    return weighted_quantile(score, 0.80, weight), weighted_quantile(score, 0.90, weight)
