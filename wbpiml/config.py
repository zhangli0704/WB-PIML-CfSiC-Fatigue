"""Fixed analysis settings and physical model specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
SEED = 20260824
GAMMA = 0.85
DAMAGE_CAP = 0.40
ARRHENIUS_REFERENCE_C = 800.0
ARRHENIUS_SCALE = 1000.0


LOG5 = math.log10(5.0)
SHEET = "Plot_Data_Verified"
EXPECTED_FILE = "data_v23.xlsx"
EXPECTED_VERIFIED_ROWS = 222
EXPECTED_EXACT_ROWS = 160
EXPECTED_RUNOUT_ROWS = 62
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results_v23_final"
PPTX_OUTPUT_NAME = "PIML_v23_final_figures.pptx"
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


V23_NEAR_TIE_RMSE = 0.020


V23_LOTO_SHORTLIST = 3
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
