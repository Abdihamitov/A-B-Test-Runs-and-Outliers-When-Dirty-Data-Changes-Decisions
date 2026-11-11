"""Data corruption mechanisms: missingness (MCAR / MAR / MNAR, in both arms or
only in arm B) and outliers (symmetric, one arm only, unit error).

A missing value is coded as NaN. Covariates (strata, MAR predictor) remain
known for rows with a missing metric: logs usually lose the metric event,
not the user.
"""
from __future__ import annotations

import numpy as np

from . import config
from .data import Pool


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def mar_intercept(pool: Pool, rate: float) -> float:
    """Find the logit intercept so that the mean missingness probability in the
    population equals rate (bisection)."""
    key = ("mar_a", rate)
    if key in pool._cache:
        return pool._cache[key]
    if pool.mar_lp is None:
        raise ValueError("MAR needs covariates, and this source has none")
    lo, hi = -20.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _expit(mid + pool.mar_lp).mean() < rate:
            lo = mid
        else:
            hi = mid
    pool._cache[key] = (lo + hi) / 2
    return pool._cache[key]


def ecdf_u(pool: Pool, y: np.ndarray) -> np.ndarray:
    """Position of a value in the population distribution (mid-rank ECDF, 0..1)."""
    s = pool.y_sorted
    return (np.searchsorted(s, y, "left") + np.searchsorted(s, y, "right")) / (2.0 * len(s))


def mnar_norm(pool: Pool) -> float:
    key = "mnar_norm"
    if key not in pool._cache:
        pool._cache[key] = float(np.exp(config.MNAR_GAMMA * ecdf_u(pool, pool.y)).mean())
    return pool._cache[key]


def missing_prob(pool: Pool, mech: str, rate: float, y: np.ndarray, lp) -> np.ndarray:
    if mech == "mcar":
        return np.full(len(y), rate)
    if mech == "mar":
        return _expit(mar_intercept(pool, rate) + lp)
    if mech == "mnar":
        w = np.exp(config.MNAR_GAMMA * ecdf_u(pool, y))
        return np.clip(rate * w / mnar_norm(pool), 0.0, 0.95)
    raise ValueError(mech)


def apply_missing(pool: Pool, mech: str, rate: float, yA, yB, lpA, lpB, rng):
    """mech: mcar | mar | mnar (both arms) or *_diff (arm B only)."""
    diff = mech.endswith("_diff")
    base = mech.removesuffix("_diff")
    yA = yA.astype(float, copy=True)
    yB = yB.astype(float, copy=True)
    pB = missing_prob(pool, base, rate, yB, lpB)
    yB[rng.random(len(yB)) < pB] = np.nan
    if not diff:
        pA = missing_prob(pool, base, rate, yA, lpA)
        yA[rng.random(len(yA)) < pA] = np.nan
    return yA, yB


def apply_outliers(pool: Pool, typ: str, eps: float, yA, yB, rng):
    """typ: sym_extreme (bots in both arms), one_arm (arm B only),
    unit_x100 (some values multiplied by 100 in both arms)."""
    if pool.kind == "binary":
        raise ValueError("Value outliers are undefined for a binary metric")
    yA = yA.astype(float, copy=True)
    yB = yB.astype(float, copy=True)
    if typ in ("sym_extreme", "one_arm"):
        ext = config.EXTREME_MULT * pool.q999
        arms = [yB] if typ == "one_arm" else [yA, yB]
        for y in arms:
            m = rng.random(len(y)) < eps
            y[m] = ext * rng.uniform(0.5, 1.5, m.sum())
    elif typ == "unit_x100":
        for y in (yA, yB):
            m = rng.random(len(y)) < eps
            y[m] *= config.UNIT_MULT
    else:
        raise ValueError(typ)
    return yA, yB
