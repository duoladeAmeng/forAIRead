from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau


def _corr(z: pd.DataFrame, x: str, y: str, method: str) -> float:
    if len(z) < 4 or z[x].nunique() < 2 or z[y].nunique() < 2:
        return np.nan
    fn = pearsonr if method == "pearson" else spearmanr
    return float(fn(z[x], z[y]).statistic)


def _partial(z: pd.DataFrame) -> float:
    if len(z) < 5 or min(z[c].nunique() for c in ["x", "y", "c"]) < 2:
        return np.nan
    a = np.column_stack([np.ones(len(z)), z["c"].to_numpy(float)])
    rx = z["x"].to_numpy(float) - a @ np.linalg.lstsq(a, z["x"].to_numpy(float), rcond=None)[0]
    ry = z["y"].to_numpy(float) - a @ np.linalg.lstsq(a, z["y"].to_numpy(float), rcond=None)[0]
    return float(pearsonr(rx, ry).statistic) if np.std(rx) > 0 and np.std(ry) > 0 else np.nan


def _base_candidate(df, h, horizons, hfs, rule):
    n = len(df)
    mask = pd.Series(True, index=df.index)
    if rule in {"horizon_common", "double_common"}:
        mask &= np.arange(n) <= n - 1 - max(horizons)
    if rule == "double_common":
        # Strictest implementation: identical current-cycle rows for every HF
        # and every horizon, including all future SOH labels.
        mask &= df[["SOH"] + hfs].notna().all(axis=1)
        for hh in horizons:
            mask &= df["SOH"].shift(-hh).notna()
    return mask


def compute_support_rule(tables, horizons, features, hfs, rule):
    future, degradation, partial, support = [], [], [], []
    for battery, df0 in tables.items():
        df = df0.reset_index(drop=True)
        for h in horizons:
            y = df["SOH"].shift(-h)
            d = df["SOH"] - y
            candidate = _base_candidate(df, h, horizons, hfs, rule)
            common = candidate.copy()
            if rule == "feature_common":
                common &= df[["SOH"] + hfs].notna().all(axis=1) & y.notna()
            for feature in features:
                cols = pd.DataFrame({"x": df[feature], "y": y, "d": d, "c": df["SOH"], "cycle": df["cycle"]})
                if rule in {"feature_common", "double_common"} and feature in hfs:
                    valid = common
                else:
                    valid = candidate & cols[["x", "y", "c"]].notna().all(axis=1)
                z = cols.loc[valid]
                for metric in ["pearson", "spearman"]:
                    future.append({"battery": battery, "feature": feature, "horizon": h, "metric": metric,
                                   "value": _corr(z, "x", "y", metric), "n_effective": len(z), "support_rule": rule})
                if feature in hfs:
                    zd = cols.loc[valid & cols["d"].notna()]
                    for metric in ["pearson", "spearman"]:
                        degradation.append({"battery": battery, "feature": feature, "horizon": h, "metric": metric,
                                            "value": _corr(zd, "x", "d", metric), "n_effective": len(zd), "support_rule": rule})
                    partial.append({"battery": battery, "feature": feature, "horizon": h,
                                    "metric": "pearson_partial_controlling_SOH", "value": _partial(z),
                                    "n_effective": len(z), "support_rule": rule})
            hf_ns = [r["n_effective"] for r in partial[-len(hfs):]]
            support.append({"battery": battery, "horizon": h, "support_rule": rule,
                            "n_effective_min": min(hf_ns), "n_effective_max": max(hf_ns),
                            "identical_hf_support": len(set(hf_ns)) == 1})
    return pd.DataFrame(future), pd.DataFrame(degradation), pd.DataFrame(partial), pd.DataFrame(support)


def rank_outputs(partial, horizons, hfs, min_n=5):
    rankings = partial.copy()
    rankings.loc[rankings.n_effective < min_n, "value"] = np.nan
    rankings["rank"] = rankings.groupby(["battery", "horizon"])["value"].transform(
        lambda s: s.abs().rank(ascending=False, method="average"))
    cross = []
    for battery, b in rankings.groupby("battery"):
        piv = b.pivot(index="feature", columns="horizon", values="rank").reindex(hfs)
        for fa, fb in itertools.combinations(hfs, 2):
            for h1, h2 in itertools.combinations(horizons, 2):
                vals = piv.loc[[fa, fb], [h1, h2]]
                if vals.isna().any().any(): continue
                if (vals.loc[fa,h1]-vals.loc[fb,h1]) * (vals.loc[fa,h2]-vals.loc[fb,h2]) < 0:
                    cross.append({"battery":battery,"feature_A":fa,"feature_B":fb,"horizon_1":h1,"horizon_2":h2,
                                  "rank_A_h1":vals.loc[fa,h1],"rank_B_h1":vals.loc[fb,h1],
                                  "rank_A_h2":vals.loc[fa,h2],"rank_B_h2":vals.loc[fb,h2]})
    cross = pd.DataFrame(cross)
    if len(cross):
        keys=["feature_A","feature_B","horizon_1","horizon_2"]
        cnt=cross.groupby(keys).battery.nunique().rename("battery_count").reset_index()
        cross=cross.merge(cnt,on=keys)
    sims=[]; trends=[]
    for battery,b in rankings.groupby("battery"):
        piv=b.pivot(index="feature",columns="horizon",values="rank").reindex(hfs)
        pairs=[]
        for h1 in horizons:
            for h2 in horizons:
                tau=float(kendalltau(piv[h1],piv[h2],nan_policy="omit").statistic)
                sims.append({"battery":battery,"horizon_1":h1,"horizon_2":h2,"metric":"kendall_tau","value":tau})
                if h2>h1: pairs.append((abs(h2-h1),tau))
        a,bv=zip(*pairs)
        trends.append({"battery":battery,"metric":"spearman_distance_vs_tau",
                       "value":float(spearmanr(a,bv,nan_policy="omit").statistic)})
    return rankings, cross, pd.DataFrame(sims), pd.DataFrame(trends)


def hf56_deltas(partials_by_rule):
    rows=[]
    for rule,p in partials_by_rule.items():
        z=p[p.feature.isin(["HF5","HF6"])].copy(); base=z[z.horizon==1].set_index(["battery","feature"]).value.abs()
        for _,r in z.iterrows():
            rows.append({**r.to_dict(),"abs_partial":abs(r.value),
                         "delta_abs_from_h1":abs(r.value)-base.get((r.battery,r.feature),np.nan)})
    return pd.DataFrame(rows)

