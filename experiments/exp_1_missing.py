"""Experiment 1. Missing values: mechanism x rate x sample size x metric,
A/A scenario (type I error) and effect scenario (power, bias, coverage).

Artifact: results/exp1_missing.csv
"""
from __future__ import annotations

import time

from abtest_sim import config
from abtest_sim.data import get_pool
from abtest_sim.simulate import Cell, run_grid
from experiments._common import save


def main():
    t0 = time.time()
    src = config.SOURCE
    has_cov = get_pool(src, config.METRICS_MISSING[0] if src != "cookie_cats" else "sum_gamerounds").mar_lp is not None
    metrics = config.METRICS_MISSING if src != "cookie_cats" else ["sum_gamerounds", "retention_1"]
    mechs = [m for m in config.MISS_MECHS if has_cov or not m.startswith("mar")]
    handlers = [h for h in config.HANDLERS_MISSING if has_cov or not h.startswith("poststrat")]
    cells = [Cell("missing", src, m, n, sc, mech, r)
             for m in metrics for n in config.N_GRID for sc in ("aa", "effect")
             for mech in mechs for r in config.MISS_RATES]
    print(f"experiment 1: {len(cells)} cells x {config.N_SIM} replications")
    df = run_grid(cells, handlers)
    save(df, "exp1_missing", t0)


if __name__ == "__main__":
    main()
