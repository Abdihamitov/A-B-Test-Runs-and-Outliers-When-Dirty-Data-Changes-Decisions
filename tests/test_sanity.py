"""Correctness checks for the code (not for the study's results).
Run: python -m pytest -q tests/"""
import numpy as np
from scipy import stats as st

from abtest_sim.corrupt import apply_missing, missing_prob
from abtest_sim.data import Pool, covariates, make_synthetic
from abtest_sim.simulate import Cell, mde_delta, run_cell
from abtest_sim.stats import handle, poststrat, welch, wilson


def _pool(metric="history"):
    df = make_synthetic(20000, seed=1)
    s, k, lp = covariates(df)
    from abtest_sim.data import METRIC_KIND
    return Pool("synthetic", metric, METRIC_KIND[metric], df[metric].to_numpy(float), s, k, lp)


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a, b = rng.lognormal(0, 1, 300), rng.lognormal(0.1, 1.3, 500)
    r = welch(a, b)
    ref = st.ttest_ind(b, a, equal_var=False)
    assert np.isclose(r.p, ref.pvalue)
    ci = ref.confidence_interval(0.95)
    assert np.isclose(r.lo, ci.low) and np.isclose(r.hi, ci.high)


def test_wilson_contains_point():
    lo, hi = wilson(50, 1000)
    assert lo < 0.05 < hi


def test_missing_rates_calibrated():
    pool = _pool()
    rng = np.random.default_rng(1)
    for mech in ["mcar", "mar", "mnar"]:
        p = missing_prob(pool, mech, 0.2, pool.y, pool.mar_lp)
        assert abs(p.mean() - 0.2) < 0.01, (mech, p.mean())


def test_diff_missing_only_in_b():
    pool = _pool()
    rng = np.random.default_rng(2)
    yA, yB = apply_missing(pool, "mcar_diff", 0.3, pool.y[:1000], pool.y[:1000], None, None, rng)
    assert not np.isnan(yA).any() and np.isnan(yB).mean() > 0.2


def test_poststrat_single_stratum_equals_mean_diff():
    pool = _pool()
    y = pool.y[:4000]
    s = np.zeros(4000, dtype=np.int64)
    r = poststrat(y[:2000], s[:2000], y[2000:], s[2000:], 1)
    assert np.isclose(r.est, y[2000:].mean() - y[:2000].mean())


def test_poststrat_coverage_without_missing():
    from abtest_sim import data
    pool = _pool()
    data._POOLS[("synthetic", "history")] = pool
    rows = run_cell(Cell("t", "synthetic", "history", 2000, "aa", "mcar", 0.0, seed=9),
                    ["poststrat|raw"], 600)
    assert 0.92 < rows[0]["coverage"] < 0.975


def test_handlers_run():
    rng = np.random.default_rng(3)
    a, b = rng.exponential(1, 200), rng.exponential(1, 200)
    a[:5] = np.nan
    for h in ["cc|raw", "zero|raw", "mean|raw", "cc|winsor99", "cc|winsor99pos",
              "cc|trim99pos", "cc|log", "cc|mwu"]:
        assert np.isfinite(handle(h, a, b).p)


def test_mde_gives_roughly_target_power_on_clean_data():
    from abtest_sim import data
    pool = _pool()
    data._POOLS[("synthetic", "history")] = pool
    rows = run_cell(Cell("t", "synthetic", "history", 2000, "effect", seed=5), ["cc|raw"], 300)
    assert 0.70 < rows[0]["clean_rej_rate"] < 0.90
    assert mde_delta(pool, 2000) > 0
