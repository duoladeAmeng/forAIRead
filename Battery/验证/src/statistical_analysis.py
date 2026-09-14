from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata


def safe_corr(x, y, method="pearson"):
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(z) < 4 or z["x"].nunique() < 2 or z["y"].nunique() < 2:
        return np.nan
    return float((pearsonr if method == "pearson" else spearmanr)(z["x"], z["y"]).statistic)


def partial_corr(x, y, control):
    z = pd.DataFrame({"x": x, "y": y, "c": control}).dropna()
    if len(z) < 5:
        return np.nan
    a = np.column_stack([np.ones(len(z)), z["c"].to_numpy()])
    rx = z["x"].to_numpy() - a @ np.linalg.lstsq(a, z["x"].to_numpy(), rcond=None)[0]
    ry = z["y"].to_numpy() - a @ np.linalg.lstsq(a, z["y"].to_numpy(), rcond=None)[0]
    return safe_corr(rx, ry)


def analyze_correlations(tables: dict[str, pd.DataFrame], horizons: list[int], features: list[str], hfs: list[str]):
    future, degradation, partial = [], [], []
    for battery, df in tables.items():
        for h in horizons:
            y = df["SOH"].shift(-h)
            d = df["SOH"] - y
            n = int(y.notna().sum())
            for feature in features:
                for metric in ["pearson", "spearman"]:
                    future.append({"battery": battery, "feature": feature, "horizon": h,
                                   "metric": metric, "value": safe_corr(df[feature], y, metric), "n": n})
            for feature in hfs:
                for metric in ["pearson", "spearman"]:
                    degradation.append({"battery": battery, "feature": feature, "horizon": h,
                                        "metric": metric, "value": safe_corr(df[feature], d, metric), "n": n})
                partial.append({"battery": battery, "feature": feature, "horizon": h,
                                "metric": "pearson_partial_controlling_SOH", 
                                "value": partial_corr(df[feature], y, df["SOH"]), "n": n})
    return pd.DataFrame(future), pd.DataFrame(degradation), pd.DataFrame(partial)


def rankings_and_similarity(future, degradation, partial, horizons, hfs):
    records = []
    sources = [
        (future[(future.metric == "spearman") & future.feature.isin(hfs)], "abs_future_soh_spearman"),
        (degradation[degradation.metric == "spearman"], "abs_degradation_spearman"),
        (partial, "abs_partial_correlation"),
    ]
    for source, name in sources:
        x = source.copy()
        x["rank"] = x.groupby(["battery", "horizon"])["value"].transform(lambda s: s.abs().rank(ascending=False, method="average"))
        x["ranking_metric"] = name
        records.append(x[["battery", "feature", "horizon", "ranking_metric", "value", "rank"]])
    rankings = pd.concat(records, ignore_index=True)

    pr = rankings[rankings.ranking_metric == "abs_partial_correlation"]
    cross = []
    for battery, b in pr.groupby("battery"):
        piv = b.pivot(index="feature", columns="horizon", values="rank").reindex(hfs)
        for fa, fb in itertools.combinations(hfs, 2):
            for h1, h2 in itertools.combinations(horizons, 2):
                vals = piv.loc[[fa, fb], [h1, h2]]
                if vals.isna().any().any():
                    continue
                d1, d2 = vals.loc[fa, h1] - vals.loc[fb, h1], vals.loc[fa, h2] - vals.loc[fb, h2]
                if d1 * d2 < 0:
                    cross.append({"battery": battery, "feature_A": fa, "feature_B": fb,
                                  "horizon_1": h1, "horizon_2": h2,
                                  "rank_A_h1": vals.loc[fa, h1], "rank_B_h1": vals.loc[fb, h1],
                                  "rank_A_h2": vals.loc[fa, h2], "rank_B_h2": vals.loc[fb, h2]})
    crossovers = pd.DataFrame(cross)
    if len(crossovers):
        keys = ["feature_A", "feature_B", "horizon_1", "horizon_2"]
        counts = crossovers.groupby(keys).battery.nunique().rename("battery_count").reset_index()
        crossovers = crossovers.merge(counts, on=keys, how="left")

    sims, trends = [], []
    for battery, b in pr.groupby("battery"):
        piv = b.pivot(index="feature", columns="horizon", values="rank").reindex(hfs)
        pairs = []
        for h1 in horizons:
            for h2 in horizons:
                tau = kendalltau(piv[h1], piv[h2], nan_policy="omit").statistic
                sims.append({"battery": battery, "horizon_1": h1, "horizon_2": h2,
                             "metric": "kendall_tau_partial_ranking", "value": tau})
                if h2 > h1:
                    pairs.append((abs(h2-h1), tau))
        trends.append({"battery": battery, "feature": "all_HFs", "horizon": -1,
                       "metric": "spearman_horizon_distance_vs_kendall_tau",
                       "value": safe_corr(*zip(*pairs), method="spearman")})
    return rankings, crossovers, pd.DataFrame(sims), pd.DataFrame(trends)


def summarize_across_batteries(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.groupby(group_cols, dropna=False)["value"].agg(["mean", "median", "std", "count"]).reset_index()
    return out

