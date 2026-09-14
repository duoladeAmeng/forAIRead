from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _save(fig,outdir,name):
    fig.savefig(outdir/f"{name}.png",dpi=320,bbox_inches="tight")
    fig.savefig(outdir/f"{name}.pdf",bbox_inches="tight")
    plt.close(fig)


def setup():
    sns.set_theme(style="whitegrid",context="paper")
    plt.rcParams.update({"figure.dpi":120,"axes.titlesize":10,"axes.labelsize":9,"legend.fontsize":7})


def fig1(value,outdir):
    cols={"improvement_currentHF":"Current HF vs SOH history","improvement_HFhistory":"HF history vs SOH history",
          "history_vs_currentHF":"HF history vs current HF"}
    z=value.melt(id_vars=["battery","horizon"],value_vars=list(cols),var_name="comparison",value_name="value")
    z.comparison=z.comparison.map(cols); q=z.groupby(["comparison","horizon"]).value.agg(["mean","std"]).reset_index()
    fig,ax=plt.subplots(figsize=(8,4.8))
    for c,x in q.groupby("comparison"):
        x=x.sort_values("horizon"); ax.plot(x.horizon,100*x["mean"],marker="o",label=c)
    ax.axhline(0,color="black",lw=.6); ax.set(xlabel="Horizon",ylabel="Relative MAE improvement (%)",title="V2.5-1 HF-history versus current-HF")
    ax.legend(); _save(fig,outdir,"figure_v25_1_hf_history_value")


def _utility_heat(group_lofo,group):
    z=group_lofo[(group_lofo.state_definition=="degradation_dynamics")&(group_lofo.feature_group==group)&group_lofo.reliable]
    return z.pivot_table(index="state",columns="horizon",values="lofo_value",aggfunc="mean").reindex(["Slow","Medium","Fast"])


def fig2(group_lofo,outdir):
    groups=sorted(group_lofo.feature_group.unique()); fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,g in zip(axes.flat,groups):
        sns.heatmap(_utility_heat(group_lofo,g),center=0,cmap="vlag",annot=True,fmt=".4f",ax=ax,cbar=False); ax.set_title(g)
    fig.suptitle("V2.5-2 State-conditioned group utility")
    _save(fig,outdir,"figure_v25_2_group_utility_heatmaps")


def _single_surface(group_lofo,group,outdir,name,title):
    fig,ax=plt.subplots(figsize=(8,3.8)); sns.heatmap(_utility_heat(group_lofo,group),center=0,cmap="vlag",annot=True,fmt=".5f",ax=ax)
    ax.set_title(title); _save(fig,outdir,name)


def fig5(rankings,outdir):
    z=rankings.groupby(["state","horizon","feature_group"]).lofo_value.mean().reset_index()
    z["rank"]=z.groupby(["state","horizon"]).lofo_value.rank(ascending=False,method="average")
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,(state,q) in zip(axes,z.groupby("state",sort=False)):
        for g,x in q.groupby("feature_group"):
            x=x.sort_values("horizon"); ax.plot(x.horizon,x["rank"],marker="o",label=g)
        ax.invert_yaxis(); ax.set_yticks(range(1,5)); ax.set_title(state); ax.set_xlabel("Horizon"); ax.set_ylabel("Rank")
    axes[0].legend(fontsize=6); fig.suptitle("V2.5-5 Group ranking within degradation state")
    _save(fig,outdir,"figure_v25_5_state_group_rankings")


def fig6(gain,outdir):
    z=gain[(gain.state_definition=="degradation_dynamics")&gain.reliable].groupby(["state","horizon"]).relative_improvement_percent.agg(["mean","std"]).reset_index()
    fig,ax=plt.subplots(figsize=(8,4.8))
    for state,q in z.groupby("state"):
        q=q.sort_values("horizon"); ax.plot(q.horizon,q["mean"],marker="o",label=state)
    ax.axhline(0,color="black",lw=.6); ax.set(xlabel="Horizon",ylabel="HF relative improvement (%)",title="V2.5-6 SOH-history + HF gain by state"); ax.legend()
    _save(fig,outdir,"figure_v25_6_state_prediction_gain")


def fig7(interaction,outdir):
    fig,ax=plt.subplots(figsize=(8,4.5)); z=interaction.sort_values("interaction_effect_size",ascending=False)
    bars=ax.bar(z.feature_group,z.interaction_effect_size)
    for bar,p in zip(bars,z.empirical_p): ax.text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"p={p:.3f}",ha="center",va="bottom",fontsize=7)
    ax.tick_params(axis="x",rotation=20); ax.set(ylabel="Cross-battery interaction effect size",title="V2.5-7 State × horizon interaction test")
    _save(fig,outdir,"figure_v25_7_interaction_effect")


def fig8(comparison,outdir):
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    sns.barplot(data=comparison,x="state_definition",y="mean_state_range",hue="feature_group",ax=axes[0]); axes[0].tick_params(axis="x",rotation=20); axes[0].set_title("Mean utility range across states")
    sns.barplot(data=comparison,x="state_definition",y="between_state_variance_ratio",hue="feature_group",ax=axes[1]); axes[1].tick_params(axis="x",rotation=20); axes[1].set_title("Between-state variance ratio"); axes[1].legend_.remove()
    fig.suptitle("V2.5-8 State-definition sensitivity"); _save(fig,outdir,"figure_v25_8_state_definition_sensitivity")


def make_all(value,group_lofo,rankings,gain,interaction,comparison,outdir):
    setup(); fig1(value,outdir); fig2(group_lofo,outdir)
    _single_surface(group_lofo,"Group_CC",outdir,"figure_v25_3_group_cc_surface","V2.5-3 Group_CC utility: state × horizon")
    _single_surface(group_lofo,"Group_DischargeIC",outdir,"figure_v25_4_group_dischargeic_surface","V2.5-4 Group_DischargeIC utility: state × horizon")
    fig5(rankings,outdir); fig6(gain,outdir); fig7(interaction,outdir); fig8(comparison,outdir)

