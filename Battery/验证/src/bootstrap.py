from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _indices(n, block, B, rng):
    blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, blocks))
    offsets = np.arange(block)
    return ((starts[:, :, None] + offsets) % n).reshape(B, -1)[:, :n]


def _row_corr(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt(np.sum(a*a, axis=1) * np.sum(b*b, axis=1))
    return np.divide(np.sum(a*b, axis=1), den, out=np.full(len(a), np.nan), where=den > 0)


def _partial_rows(x, y, c):
    rxy, rxc, ryc = _row_corr(x, y), _row_corr(x, c), _row_corr(y, c)
    den = np.sqrt((1-rxc*rxc) * (1-ryc*ryc))
    return np.divide(rxy-rxc*ryc, den, out=np.full(len(x), np.nan), where=den > 1e-12)


def bootstrap_all(tables, horizons, hfs, block_lengths, B, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for battery, df in tables.items():
        for h in horizons:
            y = df["SOH"].shift(-h)
            d = df["SOH"] - y
            for feature in hfs:
                base = pd.DataFrame({"x": df[feature], "y": y, "d": d, "c": df["SOH"]}).dropna()
                n = len(base)
                if n < 10:
                    continue
                vals = {k: base[k].to_numpy(float) for k in ["x", "y", "d", "c"]}
                for block in block_lengths:
                    idx = _indices(n, min(block, n), B, rng)
                    bx, by, bd, bc = (vals[k][idx] for k in ["x", "y", "d", "c"])
                    stats = {
                        "degradation_pearson": _row_corr(bx, bd),
                        "degradation_spearman": _row_corr(rankdata(bx, axis=1), rankdata(bd, axis=1)),
                        "partial_pearson_controlling_SOH": _partial_rows(bx, by, bc),
                    }
                    for metric, samples in stats.items():
                        lo, hi = np.nanpercentile(samples, [2.5, 97.5])
                        rows.append({"battery": battery, "feature": feature, "horizon": h,
                                     "metric": metric, "block_length": block, "bootstrap_n": B,
                                     "estimate": float(np.nanmedian(samples)), "ci_low": lo, "ci_high": hi, "n": n})
    return pd.DataFrame(rows)

