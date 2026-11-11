"""Experiment 2. Outliers: type x rate x sample size x metric,
comparing handlers (none, winsorization, trimming, log, Mann-Whitney).

Artifact: results/exp2_outliers.csv
"""
from __future__ import annotations

import time

from abtest_sim import config
from abtest_sim.simulate import Cell, run_grid
from experiments._common import save


def main():
    t0 = time.time()
    src = config.SOURCE
    metrics = config.METRICS_OUTLIER if src != "cookie_cats" else ["sum_gamerounds"]
    cells = [Cell("outlier", src, m, n, sc, None, 0.0, t, e)
             for m in metrics for n in config.N_GRID for sc in ("aa", "effect")
             for t in config.OUT_TYPES for e in config.OUT_EPS]
    print(f"experiment 2: {len(cells)} cells x {config.N_SIM} replications")
    df = run_grid(cells, config.HANDLERS_OUTLIER)
    save(df, "exp2_outliers", t0)


if __name__ == "__main__":
    main()
