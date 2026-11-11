"""Experiment 0. Data description: group sizes, metric distributions,
association of metrics with covariates (tells whether MAR missingness will
shift the means), quantiles used as winsorization thresholds.

Artifacts: results/eda_groups.csv, results/eda_metrics.csv,
results/eda_covariates.csv, results/eda_meta.json, figures/fig1_distributions.png
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as st

from abtest_sim import config
from abtest_sim.data import CONTROL_LABEL, HILLSTROM_FILE, load_source, sha256
from experiments._common import FIGURES, RESULTS, SUFFIX


def main():
    t0 = time.time()
    src = config.SOURCE
    df = load_source(src)
    metrics = ["history", "visit", "conversion", "spend"] if src != "cookie_cats" \
        else ["sum_gamerounds", "retention_1", "retention_7"]

    groups = df.groupby("group").size().rename("n").reset_index()
    groups.to_csv(RESULTS / f"eda_groups{SUFFIX}.csv", index=False)

    ctrl = df[df["group"] == CONTROL_LABEL[src]]
    rows = []
    for m in metrics:
        y = ctrl[m].astype(float)
        pos = y[y > 0]
        rows.append({
            "metric": m, "n": len(y), "mean": y.mean(), "sd": y.std(),
            "cv": y.std() / y.mean() if y.mean() else np.nan,
            "skew": st.skew(y), "share_zero": (y == 0).mean(),
            "q50": y.quantile(.5), "q90": y.quantile(.9), "q99": y.quantile(.99),
            "q999": y.quantile(.999), "max": y.max(),
            "q99_pos": pos.quantile(.99) if len(pos) else np.nan,
            "n_pos": len(pos),
        })
    em = pd.DataFrame(rows)
    em.to_csv(RESULTS / f"eda_metrics{SUFFIX}.csv", index=False)

    cov_rows = []
    if {"recency", "channel", "newbie"} <= set(ctrl.columns):
        for m in metrics:
            y = ctrl[m].astype(float)
            r, p = st.spearmanr(ctrl["recency"], y)
            cov_rows.append({"metric": m, "covariate": "recency", "spearman": r, "p": p})
            for c in ["newbie"]:
                r, p = st.spearmanr(ctrl[c], y)
                cov_rows.append({"metric": m, "covariate": c, "spearman": r, "p": p})
            by = ctrl.groupby("channel")[m].mean()
            for ch, v in by.items():
                cov_rows.append({"metric": m, "covariate": f"mean|channel={ch}", "spearman": v, "p": np.nan})
    pd.DataFrame(cov_rows).to_csv(RESULTS / f"eda_covariates{SUFFIX}.csv", index=False)

    meta = {"source": src, "rows": len(df), "columns": list(df.columns),
            "control_label": CONTROL_LABEL[src]}
    if src == "hillstrom":
        meta["sha256"] = sha256(HILLSTROM_FILE)
    (RESULTS / f"eda_meta{SUFFIX}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    cont = [m for m in metrics if ctrl[m].nunique() > 2]
    fig, axes = plt.subplots(1, len(cont), figsize=(5 * len(cont), 3.6))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, cont):
        y = ctrl[m].astype(float)
        pos = y[y > 0]
        ax.hist(np.log10(pos), bins=60, color="#4C72B0")
        ax.set_xlabel(f"log10({m}), positive values only")
        ax.set_ylabel("users")
        ax.set_title(f"{m}: share of zeros {100 * (y == 0).mean():.1f}%")
        for q, ls in [(.99, "--"), (.999, ":")]:
            ax.axvline(np.log10(pos.quantile(q)), color="k", ls=ls, lw=1, label=f"Q{q * 100:g} (>0)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig1_distributions{SUFFIX}.png", dpi=150)

    print(groups.to_string(index=False))
    print(em.round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
