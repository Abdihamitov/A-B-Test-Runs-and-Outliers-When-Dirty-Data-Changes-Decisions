"""Statistical tests and handling strategies for missing values and outliers.

A handler is a string "<missing>|<outliers>":
    missing:  cc (drop rows), zero (fill with 0),
              mean (fill with the arm's observed mean), poststrat (post-stratification)
    outliers: raw (nothing), winsor99 (cap at the pooled Q99),
              winsor99pos (cap at Q99 of positive values),
              trim99pos (drop values above Q99 of positive values),
              log (Welch on log1p), mwu (Mann-Whitney)
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import stats as st


class TestResult(NamedTuple):
    est: float      # effect estimate (for mean-diff handlers: mean B minus mean A)
    lo: float       # lower CI bound (NaN if no CI is defined)
    hi: float
    p: float
    se: float


NAN = float("nan")


def welch(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Welch t-test and CI for the difference in means.
    CI: (mB - mA) +- t_{1-alpha/2, df} * sqrt(sA^2/nA + sB^2/nB),
    df from the Welch-Satterthwaite formula."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return TestResult(NAN, NAN, NAN, NAN, NAN)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    diff = mb - ma
    se2 = va / na + vb / nb
    if se2 <= 0:
        return TestResult(diff, diff, diff, 1.0 if diff == 0 else 0.0, 0.0)
    se = np.sqrt(se2)
    df = se2 ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = 2 * st.t.sf(abs(diff / se), df)
    q = st.t.ppf(1 - alpha / 2, df)
    return TestResult(diff, diff - q * se, diff + q * se, p, se)


def ztest_prop(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Two-proportion z-test: p-value from the pooled proportion (H0: pA = pB),
    Wald CI from the unpooled variance."""
    na, nb = len(a), len(b)
    pa, pb = a.mean(), b.mean()
    pp = (a.sum() + b.sum()) / (na + nb)
    se0 = np.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
    se = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
    diff = pb - pa
    p = 2 * st.norm.sf(abs(diff / se0)) if se0 > 0 else 1.0
    z = st.norm.ppf(1 - alpha / 2)
    return TestResult(diff, diff - z * se, diff + z * se, p, se)


def mwu(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Mann-Whitney U. est = P(B > A) + 0.5 P(B = A) - 0.5 (0 under H0)."""
    if len(a) < 1 or len(b) < 1:
        return TestResult(NAN, NAN, NAN, NAN, NAN)
    res = st.mannwhitneyu(b, a, alternative="two-sided", method="asymptotic")
    return TestResult(res.statistic / (len(a) * len(b)) - 0.5, NAN, NAN, float(res.pvalue), NAN)


def poststrat(yA, sA, yB, sB, k: int, alpha: float = 0.05) -> TestResult:
    """Post-stratification with common stratum weights.
    w_s = share of stratum s among ALL users of both arms (including rows whose
    metric is missing: covariates are known). Arm mean:
        m_g = sum_s w_s * mean(y | observed, stratum s, arm g)
    Difference: m_B - m_A; variance: sum_s w_s^2 (v_As/n_As + v_Bs/n_Bs).
    Common weights matter: with per-arm weights the between-stratum variation
    does not cancel in the difference and the CI becomes too narrow.
    Corrects MAR missingness if the strata capture its cause; useless against MNAR."""
    nA_all = np.bincount(sA, minlength=k).astype(float)
    nB_all = np.bincount(sB, minlength=k).astype(float)

    def arm(y, s):
        obs = ~np.isnan(y)
        n_obs = np.bincount(s[obs], minlength=k).astype(float)
        sum1 = np.bincount(s[obs], weights=y[obs], minlength=k)
        sum2 = np.bincount(s[obs], weights=y[obs] ** 2, minlength=k)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = sum1 / n_obs
            v = (sum2 - n_obs * m ** 2) / (n_obs - 1)
        return n_obs, m, np.maximum(v, 0)

    na, ma, va = arm(yA, sA)
    nb, mb, vb = arm(yB, sB)
    ok = (na >= 2) & (nb >= 2)              # strata estimable in both arms
    if not ok.any():
        return TestResult(NAN, NAN, NAN, NAN, NAN)
    w = (nA_all + nB_all)[ok]
    w = w / w.sum()
    diff = float((w * (mb[ok] - ma[ok])).sum())
    se = float(np.sqrt((w ** 2 * (va[ok] / na[ok] + vb[ok] / nb[ok])).sum()))
    if se == 0:
        return TestResult(diff, diff, diff, 1.0 if diff == 0 else 0.0, 0.0)
    z = st.norm.ppf(1 - alpha / 2)
    return TestResult(diff, diff - z * se, diff + z * se, 2 * st.norm.sf(abs(diff / se)), se)


def bootstrap_ci(a, b, B: int = 2000, alpha: float = 0.05, rng=None, chunk: int = 100):
    """Percentile bootstrap CI for the difference in means. Arms are resampled
    independently, as they were randomized."""
    rng = rng or np.random.default_rng(0)
    diffs = []
    for start in range(0, B, chunk):
        m = min(chunk, B - start)
        ia = rng.integers(0, len(a), (m, len(a)))
        ib = rng.integers(0, len(b), (m, len(b)))
        diffs.append(b[ib].mean(axis=1) - a[ia].mean(axis=1))
    d = np.concatenate(diffs)
    return float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2)), d


def hedges_g(a, b) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return NAN
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float(j * (b.mean() - a.mean()) / sp)


def srm_pvalue(n_a: int, n_b: int) -> float:
    """Chi-square goodness-of-fit to 50/50 on the number of observed rows."""
    return float(st.chisquare([n_a, n_b]).pvalue)


def wilson(k: int, n: int, alpha: float = 0.05):
    """Wilson CI for a proportion (used for rejection rates)."""
    if n == 0:
        return NAN, NAN
    z = st.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


# ---------------- handling pipeline ----------------

MEAN_DIFF_OUT = {"raw", "winsor99", "winsor99pos", "trim99pos"}


def _threshold(yA, yB, q: float, positive: bool) -> float:
    pooled = np.concatenate([yA[~np.isnan(yA)], yB[~np.isnan(yB)]])
    if positive:
        pooled = pooled[pooled > 0]
    return float(np.quantile(pooled, q)) if len(pooled) else np.inf


def handle(name: str, yA, yB, sA=None, sB=None, k: int = 1, alpha: float = 0.05) -> TestResult:
    miss, out = name.split("|")
    yA = np.asarray(yA, dtype=float)
    yB = np.asarray(yB, dtype=float)

    # 1. outliers (threshold computed on the pooled observed sample of both arms)
    if out in ("winsor99", "winsor99pos"):
        thr = _threshold(yA, yB, 0.99, out == "winsor99pos")
        yA, yB = np.minimum(yA, thr), np.minimum(yB, thr)        # NaN stays NaN
    elif out == "trim99pos":
        thr = _threshold(yA, yB, 0.99, True)
        yA = np.where(yA > thr, np.nan, yA)
        yB = np.where(yB > thr, np.nan, yB)
        if miss != "cc":
            raise ValueError("trim is only compatible with cc")

    # 2. missing values
    if miss == "poststrat":
        if out not in ("raw", "winsor99", "winsor99pos") or k <= 1:
            raise ValueError("poststrat needs strata and a mean-diff metric")
        return poststrat(yA, sA, yB, sB, k, alpha)
    if miss == "cc":
        a, b = yA[~np.isnan(yA)], yB[~np.isnan(yB)]
    elif miss == "zero":
        a, b = np.nan_to_num(yA, nan=0.0), np.nan_to_num(yB, nan=0.0)
    elif miss == "mean":
        a = np.where(np.isnan(yA), np.nanmean(yA), yA)
        b = np.where(np.isnan(yB), np.nanmean(yB), yB)
    else:
        raise ValueError(miss)

    # 3. test
    if out == "log":
        return welch(np.log1p(a), np.log1p(b), alpha)
    if out == "mwu":
        return mwu(a, b, alpha)
    return welch(a, b, alpha)


def is_mean_diff(name: str) -> bool:
    return name.split("|")[1] in MEAN_DIFF_OUT
