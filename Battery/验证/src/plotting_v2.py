from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _save(fig,outdir,name):
    fig.savefig(outdir/f"{name}.png",dpi=320,bbox_inches="tight")
    fig.savefig(outdir/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)


def setup():
    sns.set_theme(style="whitegrid",context="paper")
    plt.rcParams.update({"figure.dpi":120,"axes.titlesize":10,"axes.labelsize":9,"legend.fontsize":7})


def fig1_effective_n(pairwise_partial,outdir):
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(battery,z) in zip(axes.flat,pairwise_partial.groupby("battery")):
        p=z.pivot(index="feature",columns="horizon",values="n_effective")
        sns.heatmap(p,annot=True,fmt=".0f",cmap="viridis",ax=ax,cbar=False); ax.set_title(battery)
    fig.suptitle("V2-1 Effective sample size under pairwise support")
    _save(fig,outdir,"figure_v2_1_effective_sample_size")


def fig2_hf56(delta,outdir):
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    styles={"pairwise":"-","horizon_common":"--","feature_common":"-.","double_common":":"}
    for ax,(battery,z) in zip(axes.flat,delta.groupby("battery")):
        for (rule,feature),q in z.groupby(["support_rule","feature"]):
            q=q.sort_values("horizon"); ax.plot(q.horizon,q.abs_partial,ls=styles[rule],marker="o",ms=2,label=f"{feature} {rule}")
        ax.set_title(battery); ax.set_xlabel("Horizon"); ax.set_ylabel("|Partial correlation|")
    axes.flat[0].legend(ncol=2,fontsize=6)
    fig.suptitle("V2-2 HF5/HF6 under four support rules")
    _save(fig,outdir,"figure_v2_2_hf56_support_rules")


def fig3_ranks(rankings,outdir):
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(battery,z) in zip(axes.flat,rankings.groupby("battery")):
        for f,q in z.groupby("feature"):
            q=q.sort_values("horizon"); ax.plot(q.horizon,q["rank"],marker="o",ms=2,label=f)
        ax.invert_yaxis(); ax.set_yticks(range(1,9)); ax.set_title(battery); ax.set_xlabel("Horizon"); ax.set_ylabel("Rank")
    axes.flat[0].legend(ncol=4)
    fig.suptitle("V2-3 Double-common-support rank evolution")
    _save(fig,outdir,"figure_v2_3_double_common_ranks")


def fig4_kendall(kendall,outdir):
    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for ax,(battery,z) in zip(axes.flat,kendall.groupby("battery")):
        p=z.pivot(index="horizon_1",columns="horizon_2",values="value")
        sns.heatmap(p,vmin=-1,vmax=1,center=0,cmap="vlag",annot=True,fmt=".1f",square=True,cbar=False,ax=ax); ax.set_title(battery)
    fig.suptitle("V2-4 Double-common-support Kendall similarity")
    _save(fig,outdir,"figure_v2_4_double_common_kendall")


def fig5_grouped_lofo(grouped,outdir):
    z=grouped.groupby(["model_context","feature","horizon"]).value.agg(["mean","std"]).reset_index()
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,(ctx,q0) in zip(axes,z.groupby("model_context")):
        for f,q in q0.groupby("feature"):
            q=q.sort_values("horizon"); ax.plot(q.horizon,q["mean"],marker="o",label=f)
        ax.axhline(0,color="black",lw=.6); ax.set_title(ctx); ax.set_xlabel("Horizon"); ax.set_ylabel("MAE(-group) - MAE(full)")
    axes[0].legend(fontsize=6)
    fig.suptitle("V2-5 Grouped LOFO")
    _save(fig,outdir,"figure_v2_5_grouped_lofo")


def fig6_baselines(metrics,alpha,outdir):
    keep=["persistence","current_SOH","history_L8","history_L16","history_L32","history_L32_currentHF"]
    z=metrics[((metrics.alpha==alpha)|metrics.alpha.isna())&metrics.feature.isin(keep)].groupby(["feature","horizon"]).value.agg(["mean","std"]).reset_index()
    fig,ax=plt.subplots(figsize=(9,5))
    for f,q in z.groupby("feature"):
        q=q.sort_values("horizon"); ax.plot(q.horizon,q["mean"],marker="o",label=f)
    ax.set(xlabel="Horizon",ylabel="LOBO test MAE",title="V2-6 Strong SOH baselines and HF augmentation"); ax.legend(ncol=2)
    _save(fig,outdir,"figure_v2_6_history_baselines")


def fig7_relative(rel,outdir):
    z=rel.groupby(["feature","horizon"]).value.agg(["mean","std"]).reset_index()
    fig,ax=plt.subplots(figsize=(9,5))
    for f,q in z.groupby("feature"):
        q=q.sort_values("horizon"); ax.plot(q.horizon,q["mean"],marker="o",label=f)
    ax.axhline(0,color="black",lw=.6); ax.set(xlabel="Horizon",ylabel="Relative improvement (%)",title="V2-7 Normalized prediction improvement"); ax.legend()
    _save(fig,outdir,"figure_v2_7_relative_improvement")


def fig8_missing(missing,alpha,outdir):
    z=missing[missing.alpha==alpha].groupby(["feature","horizon"]).value.agg(["mean","std"]).reset_index()
    fig,ax=plt.subplots(figsize=(9,5))
    for f,q in z.groupby("feature"):
        q=q.sort_values("horizon"); ax.plot(q.horizon,q["mean"],marker="o",label=f)
    ax.set(xlabel="Horizon",ylabel="LOBO test MAE",title="V2-8 HF4-HF6 missingness sensitivity"); ax.legend(fontsize=6)
    _save(fig,outdir,"figure_v2_8_missing_sensitivity")


def fig9_null(null_summary,null_samples,outdir):
    metrics=list(null_samples.columns); fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,m in zip(axes,metrics):
        sns.histplot(null_samples[m],bins=30,ax=ax); obs=float(null_summary.loc[null_summary.metric==m,"observed"].iloc[0])
        ax.axvline(obs,color="red",lw=2,label="observed"); ax.set_title(m.replace("_"," ")); ax.legend()
    fig.suptitle("V2-9 Ranking-structure null test")
    _save(fig,outdir,"figure_v2_9_ranking_null")


def fig10_comparison(partials,cross_counts,trends,outdir):
    rules=list(partials); fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    vals=[partials[r].value.abs().mean() for r in rules]
    axes[0].bar(rules,vals); axes[0].set_title("Mean |partial correlation|"); axes[0].tick_params(axis="x",rotation=30)
    axes[1].bar(rules,[cross_counts[r] for r in rules]); axes[1].set_title("Crossover count"); axes[1].tick_params(axis="x",rotation=30)
    axes[2].bar(rules,[trends[r].value.mean() for r in rules]); axes[2].axhline(0,color="black",lw=.6); axes[2].set_title("Mean distance-vs-Kendall rho"); axes[2].tick_params(axis="x",rotation=30)
    fig.suptitle("V2-10 V1-style pairwise vs controlled-support conclusions")
    _save(fig,outdir,"figure_v2_10_v1_v2_comparison")


def make_all(pairwise,partials,rankings,crossovers,kendalls,trends,delta,metrics,rel,grouped,missing,null_summary,null_samples,alpha,outdir):
    setup(); fig1_effective_n(pairwise,outdir); fig2_hf56(delta,outdir); fig3_ranks(rankings["double_common"],outdir)
    fig4_kendall(kendalls["double_common"],outdir); fig5_grouped_lofo(grouped,outdir); fig6_baselines(metrics,alpha,outdir)
    fig7_relative(rel,outdir); fig8_missing(missing,alpha,outdir); fig9_null(null_summary,null_samples,outdir)
    fig10_comparison(partials,{r:len(crossovers[r]) for r in partials},trends,outdir)

