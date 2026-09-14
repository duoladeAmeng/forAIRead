from __future__ import annotations

import argparse
import json
import pandas as pd

import config_v2 as cfg
from src.support_analysis import compute_support_rule,rank_outputs,hf56_deltas
from src.history_baselines import run_history_probes,run_grouped_lofo,run_missing_sensitivity
from src.bootstrap_v2 import bootstrap_non_circular,compare_bootstraps
from src.null_test import ranking_null_test
from src.plotting_v2 import make_all


def load_tables():
    out={}
    for b in cfg.BATTERIES:
        p=cfg.PROCESSED_DIR/f"{b}_features.csv"
        if not p.exists(): raise FileNotFoundError(f"Run v1 extraction first: {p}")
        out[b]=pd.read_csv(p,parse_dates=["start_time"])
    return out


def audit(tables):
    rows=[]
    for b,df in tables.items():
        for f in cfg.HFS:
            rows.append({"battery":b,"feature":f,"n_total":len(df),"missing":df[f].isna().sum(),
                         "missing_rate":df[f].isna().mean()})
    a=pd.DataFrame(rows); a.to_csv(cfg.RESULTS_DIR/"validation_v2_audit.csv",index=False)
    print("Validation v2 audit:")
    print("- current correlation support rule: feature-wise pairwise dropna; v1 n is y-only, not actual n")
    print("- current horizon support rule: t <= T-h (horizon-dependent current-cycle range)")
    print("- actual missing rate per HF:")
    print(a.pivot(index="battery",columns="feature",values="missing_rate").round(4).to_string())
    print("- current prediction baseline: current SOH_t Ridge; RidgeCV uses non-grouped default CV")
    print("- current bootstrap type: circular blocks after dropna compression")
    return a


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--skip-bootstrap",action="store_true"); ap.add_argument("--skip-plots",action="store_true")
    args=ap.parse_args()
    for d in [cfg.RESULTS_DIR,cfg.FIGURES_DIR,cfg.CACHE_DIR]: d.mkdir(exist_ok=True)
    tables=load_tables(); audit(tables)

    partials={}; rankings={}; crossovers={}; kendalls={}; trends={}
    naming={"pairwise":"pairwise","horizon_common":"common_horizon_support","feature_common":"feature_common_support","double_common":"double_common"}
    futures={}; degradations={}
    print("Computing four support-rule analyses...",flush=True)
    for rule,prefix in naming.items():
        future,degradation,partial,support=compute_support_rule(tables,cfg.HORIZONS,cfg.FEATURES,cfg.HFS,rule)
        futures[rule]=future; degradations[rule]=degradation; partials[rule]=partial
        if rule=="pairwise":
            future.to_csv(cfg.RESULTS_DIR/"future_soh_correlation_pairwise.csv",index=False)
            degradation.to_csv(cfg.RESULTS_DIR/"future_degradation_correlation_pairwise.csv",index=False)
            partial.to_csv(cfg.RESULTS_DIR/"partial_correlation_pairwise.csv",index=False)
        else:
            future.to_csv(cfg.RESULTS_DIR/f"{prefix}_future_soh.csv",index=False)
            degradation.to_csv(cfg.RESULTS_DIR/f"{prefix}_degradation.csv",index=False)
            partial.to_csv(cfg.RESULTS_DIR/f"{prefix}_partial.csv",index=False)
        r,c,k,t=rank_outputs(partial,cfg.HORIZONS,cfg.HFS,cfg.MIN_INTERPRETABLE_N)
        rankings[rule],crossovers[rule],kendalls[rule],trends[rule]=r,c,k,t
        r.to_csv(cfg.RESULTS_DIR/f"{prefix}_rankings.csv",index=False)
        c.to_csv(cfg.RESULTS_DIR/f"{prefix}_crossovers.csv",index=False)
        k.to_csv(cfg.RESULTS_DIR/f"{prefix}_kendall.csv",index=False)
        t.to_csv(cfg.RESULTS_DIR/f"{prefix}_kendall_trend.csv",index=False)
        support.to_csv(cfg.RESULTS_DIR/f"{prefix}_support_sizes.csv",index=False)
    delta=hf56_deltas(partials); delta.to_csv(cfg.RESULTS_DIR/"hf5_hf6_support_rule_deltas.csv",index=False)

    print("Running fixed-alpha LOBO history/trend probes and sensitivity grid...",flush=True)
    lag,metrics,relative=run_history_probes(tables,cfg.HORIZONS,cfg.HFS,cfg.RIDGE_ALPHAS,cfg.PRIMARY_ALPHA)
    metrics.to_csv(cfg.RESULTS_DIR/"history_baseline_metrics.csv",index=False)
    relative.to_csv(cfg.RESULTS_DIR/"normalized_prediction_improvement.csv",index=False)
    grouped=run_grouped_lofo(lag,tables,cfg.HORIZONS,cfg.HFS,cfg.FEATURE_GROUPS,cfg.PRIMARY_ALPHA)
    grouped.to_csv(cfg.RESULTS_DIR/"grouped_lofo.csv",index=False)
    missing=run_missing_sensitivity(lag,tables,cfg.HORIZONS,cfg.HFS,cfg.RIDGE_ALPHAS,cfg.PRIMARY_ALPHA)
    missing.to_csv(cfg.RESULTS_DIR/"hf456_missing_sensitivity.csv",index=False)

    if not args.skip_bootstrap:
        print("Running 1000-replicate non-circular segmented moving-block bootstrap...",flush=True)
        noncirc=bootstrap_non_circular(tables,cfg.HORIZONS,cfg.HFS,cfg.BLOCK_LENGTHS,cfg.BOOTSTRAP_N,cfg.RANDOM_SEED)
        noncirc.to_csv(cfg.RESULTS_DIR/"bootstrap_non_circular.csv",index=False)
        old=pd.read_csv(cfg.PROJECT_ROOT/"results"/"bootstrap_confidence_intervals.csv")
        compare_bootstraps(old,noncirc).to_csv(cfg.RESULTS_DIR/"bootstrap_circular_vs_non_circular.csv",index=False)

    print("Running 1000-permutation ranking null test...",flush=True)
    null_summary,null_samples=ranking_null_test(rankings["double_common"],cfg.HORIZONS,cfg.HFS,cfg.NULL_N,cfg.RANDOM_SEED)
    null_summary.to_csv(cfg.RESULTS_DIR/"ranking_null_test.csv",index=False)
    null_samples.to_csv(cfg.CACHE_DIR/"ranking_null_samples.csv",index=False)
    if not args.skip_plots:
        print("Rendering Validation v2 figures...",flush=True)
        make_all(partials["pairwise"],partials,rankings,crossovers,kendalls,trends,delta,metrics,relative,grouped,missing,
                 null_summary,null_samples,cfg.PRIMARY_ALPHA,cfg.FIGURES_DIR)

    manifest={"batteries":cfg.BATTERIES,"horizons":cfg.HORIZONS,"support_rules":list(naming),
              "history_lengths":cfg.HISTORY_LENGTHS,"primary_alpha":cfg.PRIMARY_ALPHA,"alpha_sensitivity":cfg.RIDGE_ALPHAS,
              "bootstrap_n":cfg.BOOTSTRAP_N,"block_lengths":cfg.BLOCK_LENGTHS,"null_permutations":cfg.NULL_N,
              "v1_results_preserved":True,"processed_csv_modified":False}
    (cfg.RESULTS_DIR/"run_manifest_v2.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__": main()

