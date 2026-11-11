"""Figures from the CSVs in results/. Run after experiments 1-4.

fig2_type1_missing.png   type I error (A/A) vs missing rate, by mechanism
fig3_bias_missing.png    bias of the effect estimate vs missing rate
fig4_outliers_power.png  power and type I error vs outlier rate
fig5_sample_size.png     type I error vs n under fixed corruption
fig6_case_forest.png     effect CIs in Hillstrom: clean vs corrupted data
fig7_flip_heatmap.png    share of flipped decisions (experiments 1 and 2)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from abtest_sim import config
from abtest_sim.stats import wilson
from experiments._common import FIGURES, RESULTS, SUFFIX

COLORS = {"cc|raw": "#4C72B0", "zero|raw": "#DD8452", "mean|raw": "#C44E52",
          "poststrat|raw": "#55A868", "cc|winsor99": "#8172B3",
          "cc|winsor99pos": "#937860", "cc|trim99pos": "#DA8BC3",
          "cc|log": "#8C8C8C", "cc|mwu": "#CCB974"}


def mc_band(n_sim, alpha=config.ALPHA):
    """Range that contains the observed rejection rate in 95% of cases when the
    true type I error is nominal (Wilson interval)."""
    return wilson(int(round(alpha * n_sim)), n_sim)


def _read(name):
    p = RESULTS / f"{name}{SUFFIX}.csv"
    return pd.read_csv(p) if p.exists() else None


def fig_missing(df, n_pick):
    aa = df[(df.scenario == "aa") & (df.n == n_pick)]
    mechs = list(dict.fromkeys(aa.mech))
    metrics = list(dict.fromkeys(aa.metric))
    fig, axes = plt.subplots(len(metrics), len(mechs), figsize=(3.2 * len(mechs), 2.8 * len(metrics)),
                             sharex=True, sharey=True, squeeze=False)
    lo, hi = mc_band(int(aa.n_sim.iloc[0]))
    for i, m in enumerate(metrics):
        for j, mech in enumerate(mechs):
            ax = axes[i, j]
            sub = aa[(aa.metric == m) & (aa.mech == mech)]
            ax.axhspan(lo, hi, color="0.9")
            ax.axhline(config.ALPHA, color="k", lw=0.8)
            for h, g in sub.groupby("handler"):
                g = g.sort_values("rate")
                ax.plot(g.rate, g.rej_rate, "o-", ms=3, color=COLORS.get(h), label=h)
            ax.set_yscale("log")
            ax.set_ylim(0.01, 1.05)
            if i == 0:
                ax.set_title(mech)
            if j == 0:
                ax.set_ylabel(f"{m}\nP(reject H0 | A/A)")
            if i == len(metrics) - 1:
                ax.set_xlabel("missing rate")
    axes[0, -1].legend(fontsize=7)
    fig.suptitle(f"Type I error under missingness, n = {n_pick} per arm (grey band: Monte Carlo noise)")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig2_type1_missing{SUFFIX}.png", dpi=150)
    plt.close(fig)

    ef = df[(df.scenario == "effect") & (df.n == n_pick)]
    fig, axes = plt.subplots(len(metrics), len(mechs), figsize=(3.2 * len(mechs), 2.8 * len(metrics)),
                             sharex=True, squeeze=False)
    for i, m in enumerate(metrics):
        for j, mech in enumerate(mechs):
            ax = axes[i, j]
            sub = ef[(ef.metric == m) & (ef.mech == mech)]
            ax.axhline(0, color="k", lw=0.8)
            for h, g in sub.groupby("handler"):
                g = g.sort_values("rate")
                # bias as a fraction of the true effect: -1 means the effect "vanished"
                ax.plot(g.rate, (g.mean_est - g.true_diff) / g.true_diff, "o-", ms=3,
                        color=COLORS.get(h), label=h)
            if i == 0:
                ax.set_title(mech)
            if j == 0:
                ax.set_ylabel(f"{m}\n(estimate - truth) / truth")
            if i == len(metrics) - 1:
                ax.set_xlabel("missing rate")
    axes[0, -1].legend(fontsize=7)
    fig.suptitle(f"Relative bias of the effect estimate, n = {n_pick} per arm")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig3_bias_missing{SUFFIX}.png", dpi=150)
    plt.close(fig)


def fig_outliers(df, n_pick):
    sub = df[df.n == n_pick]
    types = list(dict.fromkeys(sub.out_type))
    metrics = list(dict.fromkeys(sub.metric))
    fig, axes = plt.subplots(2 * len(metrics), len(types),
                             figsize=(3.4 * len(types), 2.6 * 2 * len(metrics)), squeeze=False)
    lo, hi = mc_band(int(sub.n_sim.iloc[0]))
    for i, m in enumerate(metrics):
        for j, t in enumerate(types):
            for k, sc in enumerate(["aa", "effect"]):
                ax = axes[2 * i + k, j]
                g0 = sub[(sub.metric == m) & (sub.out_type == t) & (sub.scenario == sc)]
                if sc == "aa":
                    ax.axhspan(lo, hi, color="0.9")
                    ax.axhline(config.ALPHA, color="k", lw=0.8)
                else:
                    ax.axhline(config.TARGET_POWER, color="k", lw=0.8, ls="--")
                for h, g in g0.groupby("handler"):
                    g = g.sort_values("eps")
                    ax.plot(g.eps, g.rej_rate, "o-", ms=3, color=COLORS.get(h), label=h)
                ax.set_xscale("log")
                ax.set_ylim(-0.02, 1.02)
                if i == 0 and k == 0:
                    ax.set_title(t)
                if j == 0:
                    ax.set_ylabel(f"{m}: " + ("type I error" if sc == "aa" else "power"))
                ax.set_xlabel("outlier rate")
    axes[0, -1].legend(fontsize=7)
    fig.suptitle(f"Outliers: rejection rate of H0, n = {n_pick} per arm")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig4_outliers_power{SUFFIX}.png", dpi=150)
    plt.close(fig)


def fig_size(df):
    aa = df[df.scenario == "aa"]
    labels = list(dict.fromkeys(aa.label))
    metrics = list(dict.fromkeys(aa.metric))
    fig, axes = plt.subplots(len(metrics), len(labels), figsize=(3.2 * len(labels), 2.8 * len(metrics)),
                             sharex=True, sharey=True, squeeze=False)
    for i, m in enumerate(metrics):
        for j, lab in enumerate(labels):
            ax = axes[i, j]
            g0 = aa[(aa.metric == m) & (aa.label == lab)]
            ax.axhline(config.ALPHA, color="k", lw=0.8)
            for h, g in g0.groupby("handler"):
                g = g.sort_values("n")
                ax.errorbar(g.n, g.rej_rate, yerr=[np.maximum(g.rej_rate - g.rej_lo, 0), np.maximum(g.rej_hi - g.rej_rate, 0)],
                            fmt="o-", ms=3, capsize=2, color=COLORS.get(h), label=h)
            ax.set_xscale("log")
            ax.set_ylim(-0.02, 1.02)
            if i == 0:
                ax.set_title(lab, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{m}\ntype I error")
            if i == len(metrics) - 1:
                ax.set_xlabel("n per arm")
    axes[0, -1].legend(fontsize=7)
    fig.suptitle("Growing sample size under fixed corruption (A/A), whiskers: Wilson CI")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig5_sample_size{SUFFIX}.png", dpi=150)
    plt.close(fig)


def fig_case(clean, pv):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, arm in zip(axes, clean.arm.unique()):
        c = clean[(clean.arm == arm) & (clean.metric == "spend")].iloc[0]
        ys, labels = [0], ["clean data (Welch)"]
        ax.errorbar([c.welch_diff], [0], xerr=[[c.welch_diff - c.welch_lo], [c.welch_hi - c.welch_diff]],
                    fmt="ks", capsize=3)
        ax.errorbar([c.welch_diff], [1], xerr=[[c.welch_diff - c.boot_lo], [c.boot_hi - c.welch_diff]],
                    fmt="k^", capsize=3)
        ys.append(1)
        labels.append("clean data (bootstrap)")
        sub = pv[(pv.arm == arm) & (pv.metric == "spend") & (pv.handler == "cc|raw")]
        for k, (sc, g) in enumerate(sub.groupby("scenario"), start=2):
            q = g.est.quantile([0.05, 0.5, 0.95])
            ax.errorbar([q[0.5]], [k], xerr=[[q[0.5] - q[0.05]], [q[0.95] - q[0.5]]], fmt="o",
                        color="#C44E52", capsize=3)
            ys.append(k)
            labels.append(f"{sc}: 5-95% of estimates")
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_yticks(ys, labels, fontsize=8)
        ax.set_title(f"{arm} vs No E-Mail, spend")
        ax.set_xlabel("difference in means, $")
    fig.tight_layout()
    fig.savefig(FIGURES / f"fig6_case_forest{SUFFIX}.png", dpi=150)
    plt.close(fig)


def fig_flip(e1, e2, n_pick):
    parts = []
    if e1 is not None:
        s = e1[(e1.n == n_pick) & (e1.scenario == "effect")]
        s = s.assign(corruption=s.mech + " " + (100 * s.rate).round().astype(int).astype(str) + "%")
        parts.append(s)
    if e2 is not None:
        s = e2[(e2.n == n_pick) & (e2.scenario == "effect")]
        s = s.assign(corruption=s.out_type + " " + (100 * s.eps).map(lambda x: f"{x:g}") + "%")
        parts.append(s)
    if not parts:
        return
    d = pd.concat(parts)
    for m, g in d.groupby("metric"):
        tab = g.pivot_table(index="corruption", columns="handler", values="flip_rate")
        fig, ax = plt.subplots(figsize=(1.2 * tab.shape[1] + 3, 0.32 * tab.shape[0] + 1.5))
        im = ax.imshow(tab.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(tab.shape[1]), tab.columns, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(tab.shape[0]), tab.index, fontsize=7)
        for (r, c), v in np.ndenumerate(tab.values):
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, label="share of decisions differing from clean data")
        ax.set_title(f"{m}, effect scenario, n = {n_pick}")
        fig.tight_layout()
        fig.savefig(FIGURES / f"fig7_flip_heatmap_{m}{SUFFIX}.png", dpi=150)
        plt.close(fig)


def main():
    e1, e2, e3 = _read("exp1_missing"), _read("exp2_outliers"), _read("exp3_sample_size")
    c4, p4 = _read("exp4_clean"), _read("exp4_pvalues")
    n_pick = 2000 if 2000 in config.N_GRID else config.N_GRID[0]
    if e1 is not None:
        fig_missing(e1, n_pick)
    if e2 is not None:
        fig_outliers(e2, n_pick)
    if e3 is not None:
        fig_size(e3)
    if c4 is not None and p4 is not None:
        fig_case(c4, p4)
    fig_flip(e1, e2, n_pick)
    print("figures:", ", ".join(sorted(p.name for p in FIGURES.glob("*.png"))))


if __name__ == "__main__":
    main()
