"""Experiment 3. Does a larger sample help? Fixed corruption, n from 500
to 50,000 per arm. If corruption introduces bias, the type I error should grow
with n rather than shrink.

Artifact: results/exp3_sample_size.csv
"""
from __future__ import annotations

import time

from abtest_sim import config
from abtest_sim.simulate import Cell, run_grid
from experiments._common import save


def main():
    t0 = time.time()
    src = config.SOURCE
    if src == "cookie_cats":
        raise SystemExit("experiment 3 uses MAR scenarios and needs a source with covariates")
    cells = [Cell("size", src, m, n, sc, s["mech"], s["rate"], s["out_type"], s["eps"], s["label"])
             for m in config.METRICS_SIZE for n in config.N_GRID_SIZE
             for sc in ("aa", "effect") for s in config.SIZE_SCENARIOS]
    n_sim = max(100, config.N_SIM // 2)   # large n is expensive: half the replications
    print(f"experiment 3: {len(cells)} cells x {n_sim} replications")
    df = run_grid(cells, config.HANDLERS_SIZE, n_sim=n_sim)
    save(df, "exp3_sample_size", t0)


if __name__ == "__main__":
    main()
