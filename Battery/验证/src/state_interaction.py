from __future__ import annotations

import numpy as np
import pandas as pd


def _interaction_matrix(q,states,horizons):
    m=q.pivot_table(index="state",columns="horizon",values="lofo_value",aggfunc="mean").reindex(index=states,columns=horizons)
    a=m.to_numpy(float)
    return a-np.nanmean(a,axis=1,keepdims=True)-np.nanmean(a,axis=0,keepdims=True)+np.nanmean(a)


def interaction_test(group_lofo,states,horizons,B,seed,min_n=30):
    rng=np.random.default_rng(seed); rows=[]; cache=[]
    base=group_lofo[(group_lofo.state_definition=="degradation_dynamics")&(group_lofo.n_state_samples>=min_n)]
    for group,qg in base.groupby("feature_group"):
        mats=[]; batteries=[]
        for battery,q in qg.groupby("battery"):
            mat=_interaction_matrix(q,states,horizons)
            if np.isfinite(mat).all(): mats.append(mat); batteries.append(battery)
        arr=np.stack(mats); mean_pattern=arr.mean(axis=0)
        total=np.mean(arr**2); statistic=float(np.mean(mean_pattern**2)/total) if total>0 else 0
        corrs=[]
        for i in range(len(arr)):
            for j in range(i+1,len(arr)):
                corrs.append(np.corrcoef(arr[i].ravel(),arr[j].ravel())[0,1])
        consistency=float(np.nanmean(corrs))
        null=[]
        for b in range(B):
            pm=[]
            for mat in arr:
                raw=mat.copy()
                for s in range(raw.shape[0]): raw[s]=raw[s,rng.permutation(raw.shape[1])]
                raw=raw-raw.mean(axis=1,keepdims=True)-raw.mean(axis=0,keepdims=True)+raw.mean()
                pm.append(raw)
            pa=np.stack(pm); den=np.mean(pa**2)
            null.append(np.mean(pa.mean(axis=0)**2)/den if den>0 else 0)
        null=np.asarray(null); p=(1+np.sum(null>=statistic))/(B+1)
        rows.append({"feature_group":group,"interaction_effect_size":statistic,"interaction_test_statistic":statistic,
                     "empirical_p":p,"battery_consistency":consistency,"batteries":len(arr),"permutations":B,
                     "null_hypothesis":"no cross-battery-aligned state-by-horizon interaction pattern"})
        cache.extend({"feature_group":group,"permutation":i,"null_statistic":v} for i,v in enumerate(null))
    return pd.DataFrame(rows),pd.DataFrame(cache)

