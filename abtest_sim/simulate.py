"""Monte Carlo engine.

One grid cell = (source, metric, n per arm, scenario, corruption).
In each replication:
  1. two samples of size n (A and B) are drawn with replacement from the control population;
  2. in the "effect" scenario a known relative effect delta is injected into B;
  3. the reference decision is computed on clean data (Welch, alpha);
  4. the data are corrupted (missing values and/or outliers);
  5. every handler returns an estimate, a CI and a p-value.
Cell summary: rejection rate, bias, CI coverage, CI width, share of decisions
"flipped" relative to clean data, type S and type M errors.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as st

from . import config
from .corrupt import apply_missing, apply_outliers
from .data import Pool, get_pool
from .stats import handle, is_mean_diff, srm_pvalue, welch, wilson


@dataclass
class Cell:
    exp: str
    source: str
    metric: str
    n: int
    scenario: str              # "aa" (no effect) or "effect"
    mech: str | None = None    # missingness mechanism
    rate: float = 0.0
    out_type: str | None = None
    eps: float = 0.0
    label: str = ""
    seed: int = 0


def mde_delta(pool: Pool, n: int, alpha: float = config.ALPHA,
              power: float = config.TARGET_POWER) -> float:
    """Relative effect detected with the given power at n per arm under the
    normal approximation: delta = (z_{1-a/2}+z_{pow}) * sd*sqrt(2/n) / mu."""
    z = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    return float(z * pool.sd * np.sqrt(2.0 / n) / pool.mu)


def inject_effect(pool: Pool, y: np.ndarray, delta: float, rng) -> np.ndarray:
    """Continuous metrics: y * (1 + delta). Binary: some zeros become ones so
    that the proportion grows by a factor (1 + delta). For spend the effect is
    only in the order value, not in conversion (assumption)."""
    if delta == 0:
        return y
    if pool.kind == "binary":
        p0 = pool.mu
        q = min(1.0, delta * p0 / (1 - p0))
        y = y.copy()
        flip = (y == 0) & (rng.random(len(y)) < q)
        y[flip] = 1.0
        return y
    return y * (1 + delta)


def run_cell(cell: Cell, handlers: list[str], n_sim: int = config.N_SIM) -> list[dict]:
    pool = get_pool(cell.source, cell.metric)
    rng = np.random.default_rng(cell.seed)
    alpha = config.ALPHA
    delta = mde_delta(pool, cell.n) if cell.scenario == "effect" else 0.0
    true_diff = delta * pool.mu
    N, n = pool.size, cell.n

    H = len(handlers)
    est = np.full((n_sim, H), np.nan)
    lo = np.full((n_sim, H), np.nan)
    hi = np.full((n_sim, H), np.nan)
    pv = np.full((n_sim, H), np.nan)
    clean_rej = np.zeros(n_sim, bool)
    miss_share = np.zeros(n_sim)
    srm_rej = np.zeros(n_sim, bool)

    for s in range(n_sim):
        iA = rng.integers(0, N, n)
        iB = rng.integers(0, N, n)
        yA = pool.y[iA]
        yB = inject_effect(pool, pool.y[iB], delta, rng)
        sA, sB = pool.strata[iA], pool.strata[iB]
        lpA = pool.mar_lp[iA] if pool.mar_lp is not None else None
        lpB = pool.mar_lp[iB] if pool.mar_lp is not None else None

        clean_rej[s] = welch(yA, yB, alpha).p < alpha

        if cell.out_type:
            yA, yB = apply_outliers(pool, cell.out_type, cell.eps, yA, yB, rng)
        if cell.mech:
            yA, yB = apply_missing(pool, cell.mech, cell.rate, yA, yB, lpA, lpB, rng)
            obsA, obsB = int((~np.isnan(yA)).sum()), int((~np.isnan(yB)).sum())
            miss_share[s] = 1 - (obsA + obsB) / (2 * n)
            srm_rej[s] = srm_pvalue(obsA, obsB) < config.SRM_ALPHA

        for h, name in enumerate(handlers):
            r = handle(name, yA, yB, sA, sB, pool.n_strata, alpha)
            est[s, h], lo[s, h], hi[s, h], pv[s, h] = r.est, r.lo, r.hi, r.p

    rows = []
    base = asdict(cell)
    for h, name in enumerate(handlers):
        rej = pv[:, h] < alpha
        k = int(np.nansum(rej))
        rlo, rhi = wilson(k, n_sim)
        row = dict(base, handler=name, delta=delta, true_diff=true_diff, mu=pool.mu,
                   n_sim=n_sim, rej_rate=k / n_sim, rej_lo=rlo, rej_hi=rhi,
                   clean_rej_rate=clean_rej.mean(),
                   flip_rate=float(np.mean(rej != clean_rej)),
                   miss_share=miss_share.mean(), srm_detect=srm_rej.mean())
        if is_mean_diff(name):
            e = est[:, h]
            row["mean_est"] = float(np.nanmean(e))
            row["bias_pct_mu"] = 100 * (row["mean_est"] - true_diff) / pool.mu
            row["coverage"] = float(np.nanmean((lo[:, h] <= true_diff) & (true_diff <= hi[:, h])))
            row["ci_width_pct_mu"] = 100 * float(np.nanmean(hi[:, h] - lo[:, h])) / pool.mu
            if cell.scenario == "effect" and k > 0:
                row["type_s"] = float(np.mean(np.sign(e[rej]) != np.sign(true_diff)))
                row["type_m"] = float(np.mean(np.abs(e[rej])) / true_diff)
        rows.append(row)
    return rows


def _run(args):
    cell, handlers, n_sim = args
    return run_cell(cell, handlers, n_sim)


def run_grid(cells: list[Cell], handlers: list[str], n_sim: int = config.N_SIM,
             workers: int = config.WORKERS) -> pd.DataFrame:
    """Cell seeds are spawned from config.SEED, so results do not depend on the
    number of processes or execution order."""
    seeds = np.random.SeedSequence(config.SEED).spawn(len(cells))
    for c, ss in zip(cells, seeds):
        c.seed = int(ss.generate_state(1)[0])
    jobs = [(c, handlers, n_sim) for c in cells]
    out = []
    if workers <= 1:
        for i, j in enumerate(jobs, 1):
            out.extend(_run(j))
            print(f"\r  cells: {i}/{len(jobs)}", end="", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, rows in enumerate(ex.map(_run, jobs, chunksize=1), 1):
                out.extend(rows)
                print(f"\r  cells: {i}/{len(jobs)}", end="", flush=True)
    print()
    return pd.DataFrame(out)
