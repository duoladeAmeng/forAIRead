from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,kendalltau
from sklearn.metrics import mean_absolute_error
from src.history_baselines import build_lag_tables,_dataset,_predict


STATE_ORDERS={
    "soh_level":["Early","Middle","Late"],
    "degradation_dynamics":["Slow","Medium","Fast"],
    "cycle_position":["Early-life","Mid-life","Late-life"],
}


def _causal_expanding_tertile(x,min_periods=12):
    labels=[]
    vals=x.to_numpy(float)
    for i,v in enumerate(vals):
        hist=vals[:i+1]; hist=hist[np.isfinite(hist)]
        if not np.isfinite(v) or len(hist)<min_periods: labels.append(np.nan); continue
        q1,q2=np.quantile(hist,[1/3,2/3])
        labels.append("Slow" if v<=q1 else ("Medium" if v<=q2 else "Fast"))
    return pd.Series(labels,index=x.index,dtype="object")


def build_state_tables(tables,cfg):
    out={}
    xidx=np.arange(cfg.HISTORY_LENGTH,dtype=float)
    for battery,df0 in tables.items():
        df=df0.reset_index(drop=True); soh=df.SOH
        dif=soh.shift(1)-soh
        state=pd.DataFrame({"battery":battery,"cycle":df.cycle,"current_soh":soh})
        state["soh_slope"]=soh.rolling(cfg.HISTORY_LENGTH,min_periods=cfg.HISTORY_LENGTH).apply(
            lambda x:np.polyfit(xidx,x,1)[0],raw=True)
        state["mean_degradation_rate"]=dif.rolling(cfg.HISTORY_LENGTH-1,min_periods=cfg.HISTORY_LENGTH-1).mean()
        state["recent_degradation_rate"]=dif.rolling(cfg.RECENT_LENGTH-1,min_periods=cfg.RECENT_LENGTH-1).mean()
        state["volatility"]=soh.diff().rolling(cfg.HISTORY_LENGTH-1,min_periods=cfg.HISTORY_LENGTH-1).std()
        # Fully causal level bins. Dataset-wide tertiles were deliberately not
        # used because their thresholds require future held-out observations.
        state["state_soh_level"]=np.select([soh>=.8,soh>=.5],["Early","Middle"],default="Late")
        state.loc[soh.isna(),"state_soh_level"]=np.nan
        fast_score=(-state.soh_slope+state.recent_degradation_rate)/2
        state["dynamics_score"]=fast_score
        state["state_degradation_dynamics"]=_causal_expanding_tertile(fast_score)
        # Causal lifecycle proxy: fixed 1000-cycle reference declared before
        # analysis, not each battery's eventual total cycle count.
        pos=df.cycle/cfg.CYCLE_POSITION_REFERENCE
        state["normalized_cycle_position"]=pos
        state["state_cycle_position"]=np.select([pos<=1/3,pos<=2/3],["Early-life","Mid-life"],default="Late-life")
        out[battery]=state
    return out


def leakage_audit(cfg):
    return pd.DataFrame([
        {"state_variable":"current_soh","min_source_time_offset":0,"max_source_time_offset":0,"causal":True},
        {"state_variable":"soh_slope","min_source_time_offset":-31,"max_source_time_offset":0,"causal":True},
        {"state_variable":"mean_degradation_rate","min_source_time_offset":-31,"max_source_time_offset":0,"causal":True},
        {"state_variable":"recent_degradation_rate","min_source_time_offset":-7,"max_source_time_offset":0,"causal":True},
        {"state_variable":"volatility","min_source_time_offset":-31,"max_source_time_offset":0,"causal":True},
        {"state_variable":"soh_level_fixed_bins","min_source_time_offset":0,"max_source_time_offset":0,"causal":True},
        {"state_variable":"dynamics_expanding_tertiles","min_source_time_offset":-999,"max_source_time_offset":0,"causal":True},
        {"state_variable":"cycle_position_fixed_1000_reference","min_source_time_offset":0,"max_source_time_offset":0,"causal":True},
    ])


def hf_history_value(v2_metrics,alpha):
    models=["history_L32","history_L32_currentHF","history_L32_HFhistory"]
    z=v2_metrics[(v2_metrics.alpha==alpha)&v2_metrics.feature.isin(models)].copy()
    mae=z.pivot(index=["battery","horizon"],columns="feature",values="value")
    rmse=z.pivot(index=["battery","horizon"],columns="feature",values="rmse")
    out=pd.DataFrame(index=mae.index).reset_index()
    for label,model in zip(["A_history_L32","B_currentHF","C_HFhistory"],models):
        out[f"mae_{label}"]=mae[model].to_numpy(); out[f"rmse_{label}"]=rmse[model].to_numpy()
    out["improvement_currentHF"]=(out.mae_A_history_L32-out.mae_B_currentHF)/out.mae_A_history_L32
    out["improvement_HFhistory"]=(out.mae_A_history_L32-out.mae_C_HFhistory)/out.mae_A_history_L32
    out["history_vs_currentHF"]=(out.mae_B_currentHF-out.mae_C_HFhistory)/out.mae_B_currentHF
    return out


def hf_history_summary(value):
    rows=[]
    for h,z in value.groupby("horizon"):
        row={"horizon":h}
        for m in ["improvement_currentHF","improvement_HFhistory","history_vs_currentHF"]:
            row.update({f"{m}_mean":z[m].mean(),f"{m}_median":z[m].median(),f"{m}_std":z[m].std(),
                        f"{m}_positive_battery_count":int((z[m]>0).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def run_state_prediction(tables,state_tables,cfg):
    lag=build_lag_tables(tables,cfg.HISTORY_LENGTH,cfg.HFS)
    history=[f"SOH_lag{i}" for i in range(cfg.HISTORY_LENGTH)]; full=history+cfg.HFS
    group_rows=[]; gain_rows=[]
    for h in cfg.HORIZONS:
        data=_dataset(lag,h,full)
        for held in cfg.BATTERIES:
            train,test=data[data.battery!=held],data[data.battery==held].copy()
            pred_full=_predict(train,test,full,cfg.PRIMARY_ALPHA)
            pred_hist=_predict(train,test,history,cfg.PRIMARY_ALPHA)
            errors={"full":np.abs(test.target.to_numpy()-pred_full),"history":np.abs(test.target.to_numpy()-pred_hist)}
            group_pred={}
            for group,omit in cfg.FEATURE_GROUPS.items():
                cols=[x for x in full if x not in omit]
                group_pred[group]=np.abs(test.target.to_numpy()-_predict(train,test,cols,cfg.PRIMARY_ALPHA))
            labels=state_tables[held].set_index("cycle").reindex(test.cycle)
            for definition in STATE_ORDERS:
                lab=labels[f"state_{definition}"].to_numpy()
                for state in STATE_ORDERS[definition]:
                    mask=lab==state; n=int(mask.sum())
                    if not n: continue
                    mae_h=float(errors["history"][mask].mean()); mae_f=float(errors["full"][mask].mean())
                    gain_rows.append({"battery":held,"state_definition":definition,"state":state,"horizon":h,
                                      "mae_history":mae_h,"mae_history_plus_HF":mae_f,"absolute_gain":mae_h-mae_f,
                                      "relative_improvement_percent":100*(mae_h-mae_f)/mae_h,"n_state_samples":n,
                                      "reliable":n>=cfg.MIN_STATE_N})
                    for group,e in group_pred.items():
                        mw=float(e[mask].mean())
                        group_rows.append({"battery":held,"state_definition":definition,"state":state,"horizon":h,
                                           "feature_group":group,"mae_full":mae_f,"mae_without":mw,"lofo_value":mw-mae_f,
                                           "n_state_samples":n,"reliable":n>=cfg.MIN_STATE_N})
    return pd.DataFrame(group_rows),pd.DataFrame(gain_rows)


def _partial(z):
    if len(z)<5:return np.nan
    a=np.column_stack([np.ones(len(z)),z.c.to_numpy(float)])
    rx=z.x-a@np.linalg.lstsq(a,z.x,rcond=None)[0]; ry=z.y-a@np.linalg.lstsq(a,z.y,rcond=None)[0]
    return float(pearsonr(rx,ry).statistic) if np.std(rx)>0 and np.std(ry)>0 else np.nan


def state_partials(tables,state_tables,cfg):
    rows=[]; fmap={f:g for g,fs in cfg.FEATURE_GROUPS.items() for f in fs}
    for battery,df0 in tables.items():
        df=df0.reset_index(drop=True); st=state_tables[battery].set_index("cycle").reindex(df.cycle)
        for h in cfg.HORIZONS:
            y=df.SOH.shift(-h)
            common=df[["SOH"]+cfg.HFS].notna().all(axis=1)&y.notna()
            for definition,order in STATE_ORDERS.items():
                labs=st[f"state_{definition}"].to_numpy()
                for state in order:
                    mask=common&(labs==state); n=int(mask.sum())
                    vals=[]
                    for hf in cfg.HFS:
                        z=pd.DataFrame({"x":df.loc[mask,hf],"y":y.loc[mask],"c":df.loc[mask,"SOH"]})
                        val=_partial(z); vals.append((hf,val))
                    group_means={}
                    for g in cfg.FEATURE_GROUPS:
                        gv=[abs(v) for f,v in vals if fmap[f]==g and np.isfinite(v)]
                        group_means[g]=float(np.mean(gv)) if gv else np.nan
                    for hf,val in vals:
                        rows.append({"battery":battery,"state_definition":definition,"state":state,"horizon":h,
                                     "feature":hf,"feature_group":fmap[hf],"value":val,
                                     "group_mean_abs_partial":group_means[fmap[hf]],"n_effective":n,
                                     "reliable":n>=cfg.MIN_STATE_N,"support_rule":"feature_common_within_state"})
    return pd.DataFrame(rows)


def state_rankings_and_crossovers(group_lofo,cfg):
    z=group_lofo[(group_lofo.state_definition=="degradation_dynamics")&group_lofo.reliable].copy()
    z["rank"]=z.groupby(["battery","state","horizon"]).lofo_value.rank(ascending=False,method="average")
    cross=[]
    for (battery,h),q in z.groupby(["battery","horizon"]):
        p=q.pivot(index="feature_group",columns="state",values="rank")
        if not set(STATE_ORDERS["degradation_dynamics"]).issubset(p.columns):continue
        for s1,s2 in itertools.combinations(STATE_ORDERS["degradation_dynamics"],2):
            tau=kendalltau(p[s1],p[s2]).statistic; top1=p[s1].idxmin(); top2=p[s2].idxmin()
            cross.append({"battery":battery,"horizon":h,"state_1":s1,"state_2":s2,"kendall_tau":tau,
                          "top_group_state_1":top1,"top_group_state_2":top2,"top_group_changed":top1!=top2})
    return z,pd.DataFrame(cross)


def state_definition_comparison(group_lofo,cfg):
    rows=[]
    z=group_lofo[group_lofo.reliable]
    for (definition,group),q in z.groupby(["state_definition","feature_group"]):
        cell=q.groupby(["state","horizon"]).lofo_value.mean().unstack()
        state_range=cell.max(axis=0)-cell.min(axis=0)
        total_var=float(q.lofo_value.var()); between=float(q.groupby("state").lofo_value.mean().var())
        rows.append({"state_definition":definition,"feature_group":group,
                     "mean_state_range":state_range.mean(),"max_state_range":state_range.max(),
                     "between_state_variance_ratio":between/total_var if total_var>0 else np.nan,
                     "reliable_rows":len(q)})
    return pd.DataFrame(rows)
