"""Shared helpers for experiment scripts."""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy

from abtest_sim import config

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# results for different sources do not overwrite each other
SUFFIX = "" if config.SOURCE == "hillstrom" else f"_{config.SOURCE}"


def save(df: pd.DataFrame, name: str, started: float) -> Path:
    name = name + SUFFIX
    path = RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    meta = {
        "experiment": name,
        "rows": len(df),
        "seconds": round(time.time() - started, 1),
        "n_sim": config.N_SIM,
        "quick": config.QUICK,
        "source": config.SOURCE,
        "seed": config.SEED,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "platform": platform.platform(),
    }
    (RESULTS / f"{name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"  saved: {path.relative_to(ROOT)} ({len(df)} rows, {meta['seconds']} s)")
    return path
