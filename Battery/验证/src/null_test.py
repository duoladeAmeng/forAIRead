from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy.stats import kendalltau,spearmanr


def _stats(rank_arrays,horizons,features):
    events=[]; rhos=[]; total=0
    for battery,arr in rank_arrays.items():
        for ia,ib in itertools.combinations(range(len(features)),2):
            for j,k in itertools.combinations(range(len(horizons)),2):
                if (arr[ia,j]-arr[ib,j])*(arr[ia,k]-arr[ib,k])<0:
                    total+=1; events.append((features[ia],features[ib],horizons[j],horizons[k],battery))
        pairs=[]
        for j,k in itertools.combinations(range(len(horizons)),2):
            pairs.append((abs(horizons[j]-horizons[k]),kendalltau(arr[:,j],arr[:,k]).statistic))
        rhos.append(spearmanr(*zip(*pairs)).statistic)
    if events:
        e=pd.DataFrame(events,columns=["A","B","h1","h2","battery"])
        replicated=int((e.groupby(["A","B","h1","h2"]).battery.nunique()>=3).sum())
    else: replicated=0
    return total,replicated,float(np.nanmean(rhos))


def ranking_null_test(rankings,horizons,features,B,seed):
    arrays={}
    for battery,z in rankings.groupby("battery"):
        arrays[battery]=z.pivot(index="feature",columns="horizon",values="rank").reindex(index=features,columns=horizons).to_numpy(float)
    observed=_stats(arrays,horizons,features); rng=np.random.default_rng(seed); null=np.empty((B,3))
    # Null: each feature retains its complete horizon profile and marginal rank
    # distribution, but receives an independent circular horizon shift within
    # each battery. This removes shared horizon-specific alignment without
    # shuffling cycle observations or destroying smooth feature profiles.
    for b in range(B):
        perm={}
        for battery,a in arrays.items():
            q=np.empty_like(a)
            for i in range(len(features)): q[i]=np.roll(a[i],rng.integers(0,len(horizons)))
            perm[battery]=q
        null[b]=_stats(perm,horizons,features)
    rows=[]
    names=["crossover_count","replicated_ge3_crossover_count","mean_spearman_distance_vs_kendall_tau"]
    for i,name in enumerate(names):
        obs=observed[i]; vals=null[:,i]
        if i<2: p=(1+np.sum(vals>=obs))/(B+1)
        else: p=(1+np.sum(vals<=obs))/(B+1)
        rows.append({"metric":name,"observed":obs,"null_mean":np.mean(vals),"null_std":np.std(vals,ddof=1),
                     "empirical_p":p,"permutations":B,"null_hypothesis":"no shared horizon-specific feature-rank alignment"})
    return pd.DataFrame(rows),pd.DataFrame(null,columns=names)

