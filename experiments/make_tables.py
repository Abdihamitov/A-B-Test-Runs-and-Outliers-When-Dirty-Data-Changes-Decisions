"""Summary tables for the paper (Markdown) built from CSVs in results/.
Output: results/tables.md.
"""
from __future__ import annotations

import pandas as pd

from abtest_sim import config
from experiments._common import RESULTS, SUFFIX


def _r(name):
    p = RESULTS / f"{name}{SUFFIX}.csv"
    return pd.read_csv(p) if p.exists() else None


def md(df: pd.DataFrame, digits=3) -> str:
    keep = [c for c in df.columns if c in ("eps", "rate")]   # small grid values must not round to 0
    df = df.round({c: digits for c in df.columns if c not in keep})
    return df.to_markdown(index=False)


def main():
    out = []
    n_pick = 2000 if 2000 in config.N_GRID else config.N_GRID[0]
    eda = _r("eda_metrics")
    if eda is not None:
        out += ["## Table T1. Descriptive statistics of metrics (control group)", "",
                md(eda[["metric", "n", "mean", "sd", "skew", "share_zero", "q99", "q999",
                        "q99_pos", "max"]]), ""]
    e1 = _r("exp1_missing")
    if e1 is not None:
        aa = e1[(e1.scenario == "aa") & (e1.n == n_pick)]
        t = aa.pivot_table(index=["metric", "mech", "rate"], columns="handler", values="rej_rate").reset_index()
        out += [f"## Table T2. Type I error under missingness (A/A, n = {n_pick})", "", md(t), ""]
        ef = e1[(e1.scenario == "effect") & (e1.n == n_pick)]
        t = ef.pivot_table(index=["metric", "mech", "rate"], columns="handler", values="rej_rate").reset_index()
        out += [f"## Table T3. Power under missingness (effect at MDE, n = {n_pick})", "", md(t), ""]
        t = ef.pivot_table(index=["metric", "mech", "rate"], columns="handler", values="coverage").reset_index()
        out += [f"## Table T4. Coverage of the true effect by the 95% CI (n = {n_pick})", "", md(t), ""]
        s = e1[(e1.n == n_pick) & (e1.handler == "cc|raw")]
        t = s.pivot_table(index=["metric", "mech"], columns=["scenario", "rate"], values="srm_detect")
        out += [f"## Table T5. Share of replications where the SRM check (p < {config.SRM_ALPHA}) fired", "",
                md(t.reset_index()), ""]
    e2 = _r("exp2_outliers")
    if e2 is not None:
        for sc, title in [("aa", "Type I error"), ("effect", "Power")]:
            s = e2[(e2.scenario == sc) & (e2.n == n_pick)]
            t = s.pivot_table(index=["metric", "out_type", "eps"], columns="handler",
                              values="rej_rate").reset_index()
            num = 6 if sc == "aa" else 7
            out += [f"## Table T{num}. {title} under outliers (n = {n_pick})", "", md(t), ""]
        s = e2[(e2.scenario == "effect") & (e2.n == n_pick)]
        t = s.pivot_table(index=["metric", "out_type", "eps"], columns="handler",
                          values="bias_pct_mu").reset_index()
        out += ["## Table T8. Bias of the estimate, % of the control mean (effect scenario)", "", md(t, 2), ""]
    e3 = _r("exp3_sample_size")
    if e3 is not None:
        s = e3[e3.scenario == "aa"]
        t = s.pivot_table(index=["metric", "label", "handler"], columns="n", values="rej_rate").reset_index()
        out += ["## Table T9. Type I error as a function of n (A/A)", "", md(t), ""]
    c4 = _r("exp4_clean")
    if c4 is not None:
        out += ["## Table T10. Hillstrom: clean analysis", "", md(c4, 4), ""]
    k4 = _r("exp4_corrupted")
    if k4 is not None:
        t = k4.pivot_table(index=["arm", "metric", "scenario"], columns="handler", values="flip_rate").reset_index()
        out += ["## Table T11. Hillstrom: share of replications where the conclusion changed after corruption", "", md(t), ""]
    (RESULTS / f"tables{SUFFIX}.md").write_text("\n".join(out), encoding="utf-8")
    print(f"written: results/tables{SUFFIX}.md ({len(out)} lines)")


if __name__ == "__main__":
    main()
