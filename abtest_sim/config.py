"""Shared simulation parameters.

All grids live here so that the paper and the code refer to one place.
Environment variables:
    QUICK=1     quick run (fewer replications, reduced grids), a few minutes
    N_SIM=...   Monte Carlo replications per grid cell
    WORKERS=... number of processes (default: all cores)
    SOURCE=...  hillstrom | synthetic | cookie_cats
"""
from __future__ import annotations

import os

QUICK = os.environ.get("QUICK", "0") == "1"
N_SIM = int(os.environ.get("N_SIM", "200" if QUICK else "1000"))
WORKERS = int(os.environ.get("WORKERS", str(os.cpu_count() or 1)))
SOURCE = os.environ.get("SOURCE", "hillstrom")

ALPHA = 0.05          # significance level for all tests
TARGET_POWER = 0.80   # the injected effect is sized for this power
SRM_ALPHA = 0.001     # threshold for the sample ratio mismatch check
SEED = 20261111       # root seed; all cell seeds are spawned from it

# Hillstrom metrics: history (continuous, skewed), visit (binary),
# spend (zero-inflated). For cookie_cats: sum_gamerounds, retention_1.
METRICS_MISSING = ["history", "visit", "spend"]
METRICS_OUTLIER = ["history", "spend"]          # value outliers are undefined for binary metrics
METRICS_SIZE = ["history", "spend"]

N_GRID = [500, 2000] if QUICK else [500, 2000, 10000]
N_GRID_SIZE = [500, 2000, 10000] if QUICK else [500, 1000, 2000, 5000, 10000, 20000, 50000]

MISS_MECHS = ["mcar", "mar", "mnar", "mar_diff", "mnar_diff"]
MISS_RATES = [0.05, 0.20] if QUICK else [0.05, 0.10, 0.20, 0.30]

OUT_TYPES = ["sym_extreme", "one_arm", "unit_x100"]
OUT_EPS = [0.001, 0.01] if QUICK else [0.0005, 0.001, 0.005, 0.01]

# Mechanism parameters (assumptions of the corruption model, not properties of the data)
MNAR_GAMMA = 2.0        # strength of the dependence of P(missing) on the value's rank
MAR_COEF = {"recency": 0.8, "phone": 0.8, "newbie": -0.5}
EXTREME_MULT = 10.0     # outlier = EXTREME_MULT * population Q99.9 * U(0.5, 1.5)
UNIT_MULT = 100.0       # unit error: dollars recorded as cents

HANDLERS_MISSING = ["cc|raw", "zero|raw", "mean|raw", "poststrat|raw"]
HANDLERS_OUTLIER = ["cc|raw", "cc|winsor99", "cc|winsor99pos",
                    "cc|trim99pos", "cc|log", "cc|mwu"]
HANDLERS_SIZE = ["cc|raw", "poststrat|raw", "cc|winsor99pos", "cc|mwu"]

# Scenarios for experiment 3 (growing sample size under fixed corruption)
SIZE_SCENARIOS = [
    {"label": "mcar_20", "mech": "mcar", "rate": 0.20, "out_type": None, "eps": 0.0},
    {"label": "mar_diff_5", "mech": "mar_diff", "rate": 0.05, "out_type": None, "eps": 0.0},
    {"label": "mnar_diff_5", "mech": "mnar_diff", "rate": 0.05, "out_type": None, "eps": 0.0},
    {"label": "one_arm_0.1pct", "mech": None, "rate": 0.0, "out_type": "one_arm", "eps": 0.001},
    {"label": "mnar_diff_5+one_arm", "mech": "mnar_diff", "rate": 0.05,
     "out_type": "one_arm", "eps": 0.001},
]

BOOT_B = 500 if QUICK else 2000
CASE_REPS = 50 if QUICK else 200
