from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _save(fig, outdir, name):
    fig.savefig(outdir / f"{name}.png", dpi=320, bbox_inches="tight")
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def setup_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 320, "axes.titlesize": 10,
                         "axes.labelsize": 9, "legend.fontsize": 7})


def trajectories(tables, features, outdir):
    for battery, df in tables.items():
        for standardized in [False, True]:
            fig, axes = plt.subplots(3, 3, figsize=(12, 8), constrained_layout=True)
            for ax, feature in zip(axes.flat, features):
                y = df[feature]
                if standardized:
                    y = (y-y.mean())/y.std()
                ax.plot(df.cycle, y, lw=.75)
                ax.set_title(feature); ax.set_xlabel("Cycle")
                ax.set_ylabel("z-score" if standardized else "Raw value")
            _save(fig, outdir, f"figure1_{battery}_{'standardized' if standardized else 'raw'}")


def correlation_heatmaps(future, partial, hfs, outdir):
    for battery in sorted(future.battery.unique()):
        for metric in ["pearson", "spearman"]:
            z = future[(future.battery == battery) & (future.metric == metric)]
            piv = z.pivot(index="feature", columns="horizon", values="value")
            fig, ax = plt.subplots(figsize=(8, 4.8))
            sns.heatmap(piv, vmin=-1, vmax=1, cmap="vlag", center=0, annot=True, fmt=".2f", ax=ax)
            ax.set_title(f"{battery}: {metric.title()} correlation with future SOH")
            _save(fig, outdir, f"figure2_{battery}_{metric}")
        z = partial[partial.battery == battery]
        piv = z.pivot(index="feature", columns="horizon", values="value").reindex(hfs)
        for absolute in [False, True]:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            data = piv.abs() if absolute else piv
            sns.heatmap(data, vmin=0 if absolute else -1, vmax=1,
                        cmap="viridis" if absolute else "vlag", center=None if absolute else 0,
                        annot=True, fmt=".2f", ax=ax)
            ax.set_title(f"{battery}: {'Absolute ' if absolute else ''}partial correlation | SOH(t)")
            _save(fig, outdir, f"figure2_{battery}_{'abs_' if absolute else ''}partial")


def partial_curves(partial, hfs, outdir):
    for battery, z in partial.groupby("battery"):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for f in hfs:
            q = z[z.feature == f].sort_values("horizon")
            ax.plot(q.horizon, q.value.abs(), marker="o", ms=3, label=f)
        ax.set(xlabel="Prediction horizon (cycles)", ylabel="|Partial correlation|", title=f"{battery}: horizon-feature value")
        ax.legend(ncol=4)
        _save(fig, outdir, f"figure3_{battery}_partial_curves")
    agg = partial.assign(abs_value=partial.value.abs()).groupby(["feature", "horizon"]).abs_value.agg(["mean", "std", "median"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for f in hfs:
        q = agg[agg.feature == f].sort_values("horizon")
        x, m, s = q.horizon.to_numpy(), q["mean"].to_numpy(), q["std"].fillna(0).to_numpy()
        ax.plot(x, m, marker="o", ms=3, label=f); ax.fill_between(x, m-s, m+s, alpha=.10)
    ax.set(xlabel="Prediction horizon (cycles)", ylabel="Mean |partial correlation|", title="Cross-battery mean ± SD")
    ax.legend(ncol=4)
    _save(fig, outdir, "figure3_aggregate_partial_curves")


def rank_evolution(rankings, hfs, outdir):
    pr = rankings[rankings.ranking_metric == "abs_partial_correlation"]
    for battery, z in pr.groupby("battery"):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for f in hfs:
            q = z[z.feature == f].sort_values("horizon")
            ax.plot(q.horizon, q["rank"], marker="o", ms=3, label=f)
        ax.invert_yaxis(); ax.set_yticks(range(1, len(hfs)+1))
        ax.set(xlabel="Prediction horizon (cycles)", ylabel="Rank (1 = highest)", title=f"{battery}: partial-correlation rank evolution")
        ax.legend(ncol=4)
        _save(fig, outdir, f"figure4_{battery}_rank_evolution")


def kendall_heatmaps(kendall, outdir):
    for battery, z in kendall.groupby("battery"):
        piv = z.pivot(index="horizon_1", columns="horizon_2", values="value")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(piv, vmin=-1, vmax=1, cmap="vlag", center=0, annot=True, fmt=".2f", square=True, ax=ax)
        ax.set_title(f"{battery}: Kendall tau of HF rankings")
        _save(fig, outdir, f"figure5_{battery}_kendall_similarity")


def degradation_curves(degradation, hfs, outdir):
    z = degradation[degradation.metric == "spearman"].groupby(["feature", "horizon"]).value.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for f in hfs:
        q = z[z.feature == f].sort_values("horizon")
        x, m, s = q.horizon.to_numpy(), q["mean"].to_numpy(), q["std"].fillna(0).to_numpy()
        ax.plot(x, m, marker="o", ms=3, label=f); ax.fill_between(x, m-s, m+s, alpha=.1)
    ax.axhline(0, color="black", lw=.6); ax.set(xlabel="Prediction horizon (cycles)", ylabel="Spearman correlation", title="HF vs future degradation: mean ± SD")
    ax.legend(ncol=4); _save(fig, outdir, "figure6_future_degradation")


def predictive_plots(metrics, lofo, hfs, outdir):
    z = lofo.groupby(["feature", "horizon"]).value.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for f in hfs:
        q = z[z.feature == f].sort_values("horizon")
        x, m, s = q.horizon.to_numpy(), q["mean"].to_numpy(), q["std"].fillna(0).to_numpy()
        ax.plot(x, m, marker="o", ms=3, label=f); ax.fill_between(x, m-s, m+s, alpha=.1)
    q = z[z.feature == "SOH"].sort_values("horizon")
    ax.plot(q.horizon, q["mean"], color="black", lw=2, ls="--", label="SOH (autoregressive)")
    ax.axhline(0, color="black", lw=.6); ax.set(xlabel="Prediction horizon (cycles)", ylabel="MAE(without feature) − MAE(full)", title="Leave-one-feature-out value: cross-battery mean ± SD")
    ax.legend(ncol=3); _save(fig, outdir, "figure7_leave_one_feature_out")

    z = metrics.groupby(["feature", "horizon"]).value.agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for scheme in ["SOH_only", "HF_only", "SOH_plus_HF"]:
        q = z[z.feature == scheme].sort_values("horizon")
        x, m, s = q.horizon.to_numpy(), q["mean"].to_numpy(), q["std"].fillna(0).to_numpy()
        ax.plot(x, m, marker="o", label=scheme); ax.fill_between(x, m-s, m+s, alpha=.12)
    ax.set(xlabel="Prediction horizon (cycles)", ylabel="LOBO test MAE", title="Ridge prediction probe: mean ± SD")
    ax.legend(); _save(fig, outdir, "figure8_ridge_baselines")


def make_all(tables, features, hfs, future, degradation, partial, rankings, kendall, metrics, lofo, outdir):
    setup_style(); trajectories(tables, features, outdir); correlation_heatmaps(future, partial, hfs, outdir)
    partial_curves(partial, hfs, outdir); rank_evolution(rankings, hfs, outdir)
    kendall_heatmaps(kendall, outdir); degradation_curves(degradation, hfs, outdir)
    predictive_plots(metrics, lofo, hfs, outdir)

