from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _dataset(tables, h, features):
    chunks = []
    for battery, df in tables.items():
        z = df[features].copy()
        z["target"] = df["SOH"].shift(-h)
        z["battery"] = battery
        z = z.dropna(subset=["target"])
        chunks.append(z)
    return pd.concat(chunks, ignore_index=True)


def _fit_predict(train, test, features, alphas):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas)),
    ])
    pipe.fit(train[features], train["target"])
    pred = pipe.predict(test[features])
    return pred, float(pipe.named_steps["ridge"].alpha_)


def run_probe(tables, horizons, features, hfs, alphas):
    metrics, lofo = [], []
    schemes = {"SOH_only": ["SOH"], "HF_only": hfs, "SOH_plus_HF": features}
    batteries = list(tables)
    for h in horizons:
        all_data = _dataset(tables, h, features)
        for held_out in batteries:
            train, test = all_data[all_data.battery != held_out], all_data[all_data.battery == held_out]
            full_pred = None
            full_mae = None
            for scheme, cols in schemes.items():
                pred, alpha = _fit_predict(train, test, cols, alphas)
                mae = mean_absolute_error(test.target, pred)
                rmse = mean_squared_error(test.target, pred) ** 0.5
                metrics.append({"battery": held_out, "feature": scheme, "horizon": h,
                                "metric": "MAE", "value": mae, "rmse": rmse,
                                "alpha": alpha, "n_test": len(test)})
                if scheme == "SOH_plus_HF":
                    full_pred, full_mae = pred, mae
            for omitted in features:
                cols = [f for f in features if f != omitted]
                pred, alpha = _fit_predict(train, test, cols, alphas)
                mae = mean_absolute_error(test.target, pred)
                rmse = mean_squared_error(test.target, pred) ** 0.5
                full_rmse = mean_squared_error(test.target, full_pred) ** 0.5
                lofo.append({"battery": held_out, "feature": omitted, "horizon": h,
                             "metric": "MAE_minus_full", "value": mae-full_mae,
                             "rmse_value": rmse-full_rmse, "mae_without": mae,
                             "mae_full": full_mae, "alpha": alpha, "n_test": len(test)})
    return pd.DataFrame(metrics), pd.DataFrame(lofo)

