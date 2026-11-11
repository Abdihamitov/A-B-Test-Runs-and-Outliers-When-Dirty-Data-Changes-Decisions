"""Experiment 4. A real comparison from Hillstrom: e-mail (Mens / Womens E-Mail)
vs control (No E-Mail). The true effect is unknown here; the reference is the
analysis of the full clean data.

Part 1: clean analysis (Welch, two-proportion z-test, Mann-Whitney, bootstrap, Hedges g).
Part 2: the same data are corrupted CASE_REPS times per scenario; we count how
often the decision differs from the clean one and record the p-values.

Artifacts: results/exp4_clean.csv, results/exp4_corrupted.csv,
results/exp4_pvalues.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from abtest_sim import config
from abtest_sim.corrupt import apply_missing, apply_outliers
from abtest_sim.data import covariates, get_pool, load_hillstrom
from abtest_sim.stats import bootstrap_ci, handle, hedges_g, mwu, welch, ztest_prop
from experiments._common import save

ARMS = ["Mens E-Mail", "Womens E-Mail"]
METRICS = ["visit", "conversion", "spend"]
SCENARIOS = [
    ("mcar_10", "mcar", 0.10, None, 0.0),
    ("mnar_diff_5", "mnar_diff", 0.05, None, 0.0),
    ("mar_diff_10", "mar_diff", 0.10, None, 0.0),
    ("one_arm_0.1pct", None, 0.0, "one_arm", 0.001),
    ("sym_extreme_0.1pct", None, 0.0, "sym_extreme", 0.001),
]
HANDLERS = ["cc|raw", "zero|raw", "poststrat|raw", "cc|winsor99pos", "cc|mwu"]


def main():
    if config.SOURCE != "hillstrom":
        raise SystemExit("experiment 4 only runs on Hillstrom (real e-mail arms)")
    t0 = time.time()
    alpha = config.ALPHA
    df = load_hillstrom()
    rng = np.random.default_rng(config.SEED)
    ctrl = df[df.group == "No E-Mail"].reset_index(drop=True)
    sA, k, lpA = covariates(ctrl)

    clean_rows, corr_rows, p_rows = [], [], []
    for arm in ARMS:
        trt = df[df.group == arm].reset_index(drop=True)
        sB, _, _ = covariates(trt)
        # the MAR predictor for B uses recency standardized by the control,
        # so that the mechanism is identical for both arms
        c = config.MAR_COEF
        rec_z = (trt.recency - ctrl.recency.mean()) / ctrl.recency.std()
        lpB = (c["recency"] * rec_z + c["phone"] * (trt.channel == "Phone")
               + c["newbie"] * trt.newbie).to_numpy(float)
        for m in METRICS:
            a = ctrl[m].to_numpy(float)
            b = trt[m].to_numpy(float)
            w = welch(a, b, alpha)
            bl, bh, _ = bootstrap_ci(a, b, config.BOOT_B, alpha, rng)
            row = {"arm": arm, "metric": m, "nA": len(a), "nB": len(b),
                   "meanA": a.mean(), "meanB": b.mean(),
                   "lift_pct": 100 * (b.mean() / a.mean() - 1),
                   "welch_diff": w.est, "welch_lo": w.lo, "welch_hi": w.hi, "welch_p": w.p,
                   "boot_lo": bl, "boot_hi": bh,
                   "mwu_p": mwu(a, b).p, "hedges_g": hedges_g(a, b)}
            if m in ("visit", "conversion"):
                z = ztest_prop(a, b, alpha)
                row.update(z_lo=z.lo, z_hi=z.hi, z_p=z.p)
            clean_rows.append(row)

            pool = get_pool("hillstrom", m)
            for label, mech, rate, ot, eps in SCENARIOS:
                if ot and pool.kind == "binary":
                    continue
                hs = [h for h in HANDLERS if not (pool.kind == "binary" and "winsor" in h)]
                clean_dec = {h: handle(h, a, b, sA, sB, k, alpha).p < alpha for h in hs}
                stats_ = {h: [] for h in hs}
                for r in range(config.CASE_REPS):
                    ya, yb = a, b
                    if ot:
                        ya, yb = apply_outliers(pool, ot, eps, ya, yb, rng)
                    if mech:
                        ya, yb = apply_missing(pool, mech, rate, ya, yb, lpA, lpB, rng)
                    for h in hs:
                        res = handle(h, ya, yb, sA, sB, k, alpha)
                        stats_[h].append(res)
                        p_rows.append({"arm": arm, "metric": m, "scenario": label,
                                       "handler": h, "rep": r, "p": res.p, "est": res.est})
                for h in hs:
                    ps = np.array([x.p for x in stats_[h]])
                    es = np.array([x.est for x in stats_[h]])
                    corr_rows.append({
                        "arm": arm, "metric": m, "scenario": label, "handler": h,
                        "reps": config.CASE_REPS, "clean_reject": clean_dec[h],
                        "reject_rate": float(np.mean(ps < alpha)),
                        "flip_rate": float(np.mean((ps < alpha) != clean_dec[h])),
                        "p_median": float(np.median(ps)),
                        "est_median": float(np.median(es)),
                        "est_clean": w.est if "|raw" in h else np.nan,
                    })
        print(f"  {arm}: done")
    save(pd.DataFrame(clean_rows), "exp4_clean", t0)
    save(pd.DataFrame(corr_rows), "exp4_corrupted", t0)
    save(pd.DataFrame(p_rows), "exp4_pvalues", t0)


if __name__ == "__main__":
    main()
