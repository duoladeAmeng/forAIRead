from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_lag_tables(tables, max_lag, hfs):
    out={}
    for battery,df0 in tables.items():
        df=df0.reset_index(drop=True).copy()
        cols={"battery":pd.Series(battery,index=df.index),"cycle":df.cycle,"SOH":df.SOH}
        for lag in range(max_lag):
            cols[f"SOH_lag{lag}"]=df.SOH.shift(lag)
            for hf in hfs: cols[f"{hf}_lag{lag}"]=df[hf].shift(lag)
        for hf in hfs:
            cols[hf]=df[hf]
            cols[f"M{hf[2:]}"]=df[hf].isna().astype(float)
        # Fixed L=32 trend features. All are past/current-only.
        w=df.SOH.rolling(max_lag,min_periods=max_lag)
        cols["SOH_trend_current"]=df.SOH
        cols["SOH_trend_mean"]=w.mean(); cols["SOH_trend_std"]=w.std()
        cols["SOH_trend_lastdiff"]=df.SOH.diff()
        cols["SOH_trend_mean_diff"]=df.SOH.diff().rolling(max_lag-1,min_periods=max_lag-1).mean()
        arr=np.arange(max_lag,dtype=float)
        cols["SOH_trend_slope"]=df.SOH.rolling(max_lag,min_periods=max_lag).apply(
            lambda x: np.polyfit(arr,x,1)[0],raw=True)
        z=pd.DataFrame(cols)
        out[battery]=z
    return out


def _dataset(lag_tables,h,features,complete_case=False):
    chunks=[]
    for battery,z0 in lag_tables.items():
        z=z0[["battery","cycle"]+features].copy()
        z["target"]=z0["SOH"].shift(-h)
        z=z.dropna(subset=["target"])
        lag_ids=[int(f.rsplit("lag",1)[1]) for f in features if "_lag" in f]
        required_history=(max(lag_ids)+1) if lag_ids else (32 if any("SOH_trend_" in f for f in features) else 1)
        z=z[z["cycle"]>=required_history]
        if complete_case: z=z.dropna(subset=features)
        chunks.append(z)
    return pd.concat(chunks,ignore_index=True)


def _predict(train,test,features,alpha):
    pipe=Pipeline([("imputer",SimpleImputer(strategy="median",keep_empty_features=True)),
                   ("scaler",StandardScaler()),("ridge",Ridge(alpha=alpha))])
    pipe.fit(train[features],train.target)
    pred=pipe.predict(test[features])
    return pred


def _metric_row(test,pred,battery,h,model,alpha,L=None):
    return {"battery":battery,"feature":model,"horizon":h,"metric":"MAE",
            "value":mean_absolute_error(test.target,pred),
            "rmse":mean_squared_error(test.target,pred)**.5,"alpha":alpha,
            "history_length":L,"n_test":len(test)}


def model_specs(hfs):
    specs={"current_SOH":["SOH_lag0"],"current_SOH_currentHF":["SOH_lag0"]+hfs}
    for L in [8,16,32]: specs[f"history_L{L}"]=[f"SOH_lag{i}" for i in range(L)]
    specs["history_L32_currentHF"]=[f"SOH_lag{i}" for i in range(32)]+hfs
    trend=["SOH_trend_current","SOH_trend_mean","SOH_trend_std","SOH_trend_slope","SOH_trend_lastdiff","SOH_trend_mean_diff"]
    specs["trend_L32"]=trend
    specs["trend_L32_currentHF"]=trend+hfs
    for L in [16,32]:
        specs[f"history_L{L}_HFhistory"]=[f"SOH_lag{i}" for i in range(L)]+[f"{hf}_lag{i}" for hf in hfs for i in range(L)]
    return specs


def run_history_probes(tables,horizons,hfs,alphas,primary_alpha):
    lag=build_lag_tables(tables,32,hfs); specs=model_specs(hfs); rows=[]
    for h in horizons:
        # Persistence has no fitted parameters and uses only valid current SOH.
        data=_dataset(lag,h,["SOH_lag0"],complete_case=True)
        for held in tables:
            test=data[data.battery==held]
            rows.append(_metric_row(test,test.SOH_lag0.to_numpy(),held,h,"persistence",np.nan,1))
        for model,features in specs.items():
            data=_dataset(lag,h,features)
            L=32 if "32" in model else (16 if "16" in model else (8 if "8" in model else 1))
            for held in tables:
                train,test=data[data.battery!=held],data[data.battery==held]
                for alpha in alphas:
                    pred=_predict(train,test,features,alpha)
                    rows.append(_metric_row(test,pred,held,h,model,alpha,L))
    metrics=pd.DataFrame(rows)
    primary=metrics[(metrics.alpha==primary_alpha)|metrics.alpha.isna()].copy()
    pairs=[("current_SOH","current_SOH_currentHF"),("history_L32","history_L32_currentHF"),("trend_L32","trend_L32_currentHF")]
    rel=[]
    p=primary.pivot_table(index=["battery","horizon"],columns="feature",values="value")
    for base,model in pairs:
        for idx,row in p.iterrows():
            if base not in row or model not in row or pd.isna(row[base]) or pd.isna(row[model]): continue
            rel.append({"battery":idx[0],"horizon":idx[1],"feature":f"{base}_vs_{model}","metric":"relative_improvement_percent",
                        "baseline_mae":row[base],"model_mae":row[model],"absolute_improvement":row[base]-row[model],
                        "value":100*(row[base]-row[model])/row[base]})
    return lag,metrics,pd.DataFrame(rel)


def run_grouped_lofo(lag,tables,horizons,hfs,groups,alpha):
    rows=[]
    contexts={"current_SOH":["SOH_lag0"],"history_L32":[f"SOH_lag{i}" for i in range(32)]}
    for context,base in contexts.items():
        full=base+hfs
        for h in horizons:
            data=_dataset(lag,h,full)
            for held in tables:
                train,test=data[data.battery!=held],data[data.battery==held]
                pred=_predict(train,test,full,alpha); full_mae=mean_absolute_error(test.target,pred)
                for group,omit in groups.items():
                    cols=[x for x in full if x not in omit]
                    q=_predict(train,test,cols,alpha); mae=mean_absolute_error(test.target,q)
                    rows.append({"battery":held,"feature":group,"horizon":h,"metric":"MAE_minus_full",
                                 "value":mae-full_mae,"mae_without":mae,"mae_full":full_mae,
                                 "model_context":context,"alpha":alpha,"n_test":len(test)})
    out=pd.DataFrame(rows)
    out["rank"]=out.groupby(["battery","horizon","model_context"])["value"].rank(ascending=False,method="average")
    return out


def run_missing_sensitivity(lag,tables,horizons,hfs,alphas,primary_alpha):
    no_ic=["SOH_lag0","HF1","HF2","HF3","HF7","HF8"]
    full=["SOH_lag0"]+hfs
    indicators=["M4","M5","M6"]
    specs={"A_drop_HF456":(no_ic,False),"B_HF456_median":(full,False),
           "B_HF456_median_indicators":(full+indicators,False),
           "B_missing_mask_only":(["SOH_lag0"]+indicators,False),
           "C_complete_case_HF456":(full,True)}
    rows=[]
    for h in horizons:
        for model,(features,complete) in specs.items():
            data=_dataset(lag,h,features,complete_case=complete)
            for held in tables:
                train,test=data[data.battery!=held],data[data.battery==held]
                for alpha in alphas:
                    pred=_predict(train,test,features,alpha)
                    rows.append(_metric_row(test,pred,held,h,model,alpha,1))
    return pd.DataFrame(rows)
