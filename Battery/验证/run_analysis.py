from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd

import config as cfg
from src.load_data import inspect_workbooks, read_battery_raw
from src.extract_features import extract_battery_features
from src.statistical_analysis import analyze_correlations, rankings_and_similarity, summarize_across_batteries
from src.bootstrap import bootstrap_all
from src.prediction_probe import run_probe
from src.plotting import make_all


def quality_report(tables):
    rows = []
    for battery, df in tables.items():
        for col in cfg.FEATURES + ["capacity_Ah"]:
            x = pd.to_numeric(df[col], errors="coerce")
            med, mad = x.median(), (x-x.median()).abs().median()
            extreme = ((x-med).abs() > 8 * 1.4826 * mad) if mad > 0 else pd.Series(False, index=x.index)
            rows.append({"battery": battery, "feature": col, "n": len(x), "missing": int(x.isna().sum()),
                         "infinite": int(np.isinf(x).sum()), "robust_8mad_extreme": int(extreme.sum()),
                         "extreme_cycles": ";".join(map(str, df.loc[extreme, "cycle"].tolist()))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extraction", action="store_true")
    ap.add_argument("--skip-bootstrap", action="store_true")
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()
    for d in [cfg.PROCESSED_DIR, cfg.RESULTS_DIR, cfg.FIGURES_DIR, cfg.CACHE_DIR]: d.mkdir(exist_ok=True)

    inventory = inspect_workbooks(cfg.DATA_ROOT, cfg.BATTERIES)
    inventory.to_csv(cfg.RESULTS_DIR / "raw_file_inventory.csv", index=False)
    tables, audits, file_audits = {}, [], []
    for battery in cfg.BATTERIES:
        feature_path = cfg.PROCESSED_DIR / f"{battery}_features.csv"
        if args.skip_extraction and feature_path.exists():
            tables[battery] = pd.read_csv(feature_path, parse_dates=["start_time"])
        else:
            print(f"Reading and extracting {battery}...", flush=True)
            raw, fa = read_battery_raw(cfg.DATA_ROOT, battery)
            features, audit = extract_battery_features(raw, battery, cfg)
            features.to_csv(feature_path, index=False)
            tables[battery] = features; audits.append(audit); fa.insert(0, "battery", battery); file_audits.append(fa)
    if audits: pd.concat(audits).to_csv(cfg.RESULTS_DIR / "cycle_extraction_audit.csv", index=False)
    if file_audits: pd.concat(file_audits).to_csv(cfg.RESULTS_DIR / "workbook_cycle_inventory.csv", index=False)

    quality_report(tables).to_csv(cfg.RESULTS_DIR / "feature_quality_report.csv", index=False)
    horizons = cfg.MAIN_HORIZONS + [h for h in cfg.EXTENDED_HORIZONS if all(len(x) > h+10 for x in tables.values())]
    future, degradation, partial = analyze_correlations(tables, horizons, cfg.FEATURES, cfg.HFS)
    future[future.metric == "pearson"].to_csv(cfg.RESULTS_DIR / "pearson_future_soh.csv", index=False)
    future[future.metric == "spearman"].to_csv(cfg.RESULTS_DIR / "spearman_future_soh.csv", index=False)
    degradation.to_csv(cfg.RESULTS_DIR / "future_degradation_correlation.csv", index=False)
    partial.to_csv(cfg.RESULTS_DIR / "partial_correlation.csv", index=False)
    rankings, crossovers, kendall, trend = rankings_and_similarity(future, degradation, partial, horizons, cfg.HFS)
    rankings.to_csv(cfg.RESULTS_DIR / "feature_rankings.csv", index=False)
    crossovers.to_csv(cfg.RESULTS_DIR / "ranking_crossovers.csv", index=False)
    kendall.to_csv(cfg.RESULTS_DIR / "kendall_horizon_similarity.csv", index=False)
    trend.to_csv(cfg.RESULTS_DIR / "kendall_distance_trend.csv", index=False)

    for name, frame, groups in [
        ("future_soh_summary", future, ["feature", "horizon", "metric"]),
        ("degradation_summary", degradation, ["feature", "horizon", "metric"]),
        ("partial_summary", partial, ["feature", "horizon", "metric"]),
    ]:
        summarize_across_batteries(frame, groups).to_csv(cfg.RESULTS_DIR / f"{name}.csv", index=False)

    if not args.skip_bootstrap:
        print("Running 1000-replicate circular moving-block bootstrap...", flush=True)
        boot = bootstrap_all(tables, horizons, cfg.HFS, cfg.BLOCK_LENGTHS, cfg.BOOTSTRAP_N, cfg.RANDOM_SEED)
        boot.to_csv(cfg.RESULTS_DIR / "bootstrap_confidence_intervals.csv", index=False)
    print("Running leave-one-battery-out Ridge probes...", flush=True)
    metrics, lofo = run_probe(tables, horizons, cfg.FEATURES, cfg.HFS, cfg.RIDGE_ALPHAS)
    metrics.to_csv(cfg.RESULTS_DIR / "ridge_metrics.csv", index=False)
    lofo.to_csv(cfg.RESULTS_DIR / "leave_one_feature_out.csv", index=False)
    if not args.skip_plots:
        print("Rendering figures...", flush=True)
        make_all(tables, cfg.FEATURES, cfg.HFS, future, degradation, partial, rankings, kendall, metrics, lofo, cfg.FIGURES_DIR)

    manifest = {"batteries": {k: len(v) for k,v in tables.items()}, "horizons": horizons,
                "soh_definition": "discharge_capacity_Ah / 1.1_Ah_rated_capacity",
                "bootstrap_n": cfg.BOOTSTRAP_N, "block_lengths": cfg.BLOCK_LENGTHS,
                "random_seed": cfg.RANDOM_SEED}
    (cfg.RESULTS_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__": main()

