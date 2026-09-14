from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from src.bootstrap import _row_corr, _partial_rows


def _valid_segments(cycles):
    cycles=np.asarray(cycles)
    cuts=np.r_[0,np.where(np.diff(cycles)!=1)[0]+1,len(cycles)]
    return [np.arange(cuts[i],cuts[i+1]) for i in range(len(cuts)-1) if cuts[i+1]>cuts[i]]


def _noncircular_indices(cycles,block,B,rng):
    segs=_valid_segments(cycles); starts=[]
    for seg in segs:
        if len(seg)>=block:
            starts.extend(seg[i:i+block] for i in range(len(seg)-block+1))
    if not starts:
        starts=[s for s in segs]
    n=len(cycles); out=np.empty((B,n),dtype=int)
    for b in range(B):
        chosen=[]
        while sum(map(len,chosen))<n:
            chosen.append(starts[rng.integers(0,len(starts))])
        out[b]=np.concatenate(chosen)[:n]
    return out,len(segs),max((len(s) for s in segs),default=0)


def bootstrap_non_circular(tables,horizons,hfs,blocks,B,seed):
    rng=np.random.default_rng(seed); rows=[]
    for battery,df in tables.items():
        df=df.reset_index(drop=True)
        for h in horizons:
            y=df.SOH.shift(-h); d=df.SOH-y
            for feature in hfs:
                z=pd.DataFrame({"x":df[feature],"y":y,"d":d,"c":df.SOH,"cycle":df.cycle}).dropna()
                if len(z)<10: continue
                vals={k:z[k].to_numpy(float) for k in ["x","y","d","c"]}
                for block in blocks:
                    idx,nseg,maxseg=_noncircular_indices(z.cycle.to_numpy(),min(block,len(z)),B,rng)
                    bx,by,bd,bc=(vals[k][idx] for k in ["x","y","d","c"])
                    stats={"degradation_pearson":_row_corr(bx,bd),
                           "degradation_spearman":_row_corr(rankdata(bx,axis=1),rankdata(bd,axis=1)),
                           "partial_pearson_controlling_SOH":_partial_rows(bx,by,bc)}
                    for metric,samples in stats.items():
                        lo,hi=np.nanpercentile(samples,[2.5,97.5])
                        rows.append({"battery":battery,"feature":feature,"horizon":h,"metric":metric,
                                     "block_length":block,"bootstrap_n":B,"estimate":float(np.nanmedian(samples)),
                                     "ci_low":lo,"ci_high":hi,"n_effective":len(z),
                                     "contiguous_segments":nseg,"max_segment_length":maxseg,"bootstrap_type":"non_circular_segmented"})
    return pd.DataFrame(rows)


def compare_bootstraps(v1_circular,noncirc):
    c=v1_circular.rename(columns={"estimate":"circular_estimate","ci_low":"circular_low","ci_high":"circular_high","n":"circular_n"})
    n=noncirc.rename(columns={"estimate":"noncircular_estimate","ci_low":"noncircular_low","ci_high":"noncircular_high"})
    keys=["battery","feature","horizon","metric","block_length"]
    out=c.merge(n,on=keys,how="inner",suffixes=("_c","_n"))
    out["ci_width_circular"]=out.circular_high-out.circular_low
    out["ci_width_noncircular"]=out.noncircular_high-out.noncircular_low
    return out

