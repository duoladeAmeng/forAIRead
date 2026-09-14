from __future__ import annotations

import argparse,json
import pandas as pd
import config_v25 as cfg
from src.state_analysis import (build_state_tables,leakage_audit,hf_history_value,hf_history_summary,
    run_state_prediction,state_partials,state_rankings_and_crossovers,state_definition_comparison)
from src.state_interaction import interaction_test
from src.plotting_v25 import make_all


def load_tables():
    return {b:pd.read_csv(cfg.PROCESSED_DIR/f"{b}_features.csv",parse_dates=["start_time"]) for b in cfg.BATTERIES}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--skip-plots",action="store_true"); args=ap.parse_args()
    for d in [cfg.RESULTS_DIR,cfg.FIGURES_DIR,cfg.CACHE_DIR]:d.mkdir(exist_ok=True)
    print("Auditing existing v2 HF-history outputs...",flush=True)
    v2=pd.read_csv(cfg.RESULTS_V2_DIR/"history_baseline_metrics.csv")
    required={"history_L32","history_L32_currentHF","history_L32_HFhistory"}
    assert required.issubset(v2.feature.unique()),"Required v2 HF-history models are missing"
    value=hf_history_value(v2,cfg.PRIMARY_ALPHA); summary=hf_history_summary(value)
    value.to_csv(cfg.RESULTS_DIR/"hf_history_value.csv",index=False)
    summary.to_csv(cfg.RESULTS_DIR/"hf_history_value_summary.csv",index=False)

    tables=load_tables(); states=build_state_tables(tables,cfg)
    leak=leakage_audit(cfg); assert (leak.max_source_time_offset<=0).all() and leak.causal.all()
    leak.to_csv(cfg.RESULTS_DIR/"state_leakage_audit.csv",index=False)
    for b,z in states.items(): z.to_csv(cfg.CACHE_DIR/f"{b}_causal_states.csv",index=False)
    counts=pd.concat([z.melt(id_vars=["battery","cycle"],value_vars=[c for c in z if c.startswith("state_")],var_name="state_definition",value_name="state") for z in states.values()])
    counts.groupby(["battery","state_definition","state"],dropna=False).size().rename("n").reset_index().to_csv(cfg.RESULTS_DIR/"state_sample_counts.csv",index=False)

    print("Training shared LOBO models and evaluating held-out states...",flush=True)
    group_lofo,gain=run_state_prediction(tables,states,cfg)
    group_lofo.to_csv(cfg.RESULTS_DIR/"state_conditioned_group_lofo.csv",index=False)
    gain.to_csv(cfg.RESULTS_DIR/"state_conditioned_prediction_gain.csv",index=False)
    partial=state_partials(tables,states,cfg); partial.to_csv(cfg.RESULTS_DIR/"state_conditioned_partial.csv",index=False)
    rankings,cross=state_rankings_and_crossovers(group_lofo,cfg)
    rankings.to_csv(cfg.RESULTS_DIR/"state_rankings.csv",index=False); cross.to_csv(cfg.RESULTS_DIR/"state_rank_crossovers.csv",index=False)
    comparison=state_definition_comparison(group_lofo,cfg); comparison.to_csv(cfg.RESULTS_DIR/"state_definition_comparison.csv",index=False)

    print("Running 1000-permutation State x Horizon interaction tests...",flush=True)
    interaction,null=interaction_test(group_lofo,["Slow","Medium","Fast"],cfg.HORIZONS,cfg.PERMUTATIONS,cfg.RANDOM_SEED,cfg.MIN_STATE_N)
    interaction.to_csv(cfg.RESULTS_DIR/"state_horizon_interaction_test.csv",index=False)
    null.to_csv(cfg.CACHE_DIR/"state_horizon_interaction_null.csv",index=False)
    if not args.skip_plots:
        print("Rendering Validation v2.5 figures...",flush=True)
        make_all(value,group_lofo,rankings,gain,interaction,comparison,cfg.FIGURES_DIR)
    manifest={"questions":["HF-history versus current HF","State x Horizon feature utility"],"batteries":cfg.BATTERIES,
              "horizons":cfg.HORIZONS,"history_length":cfg.HISTORY_LENGTH,"alpha":cfg.PRIMARY_ALPHA,
              "state_definitions":["causal fixed SOH level","causal expanding degradation dynamics tertiles","causal fixed-reference cycle position"],
              "permutations":cfg.PERMUTATIONS,"min_reliable_state_n":cfg.MIN_STATE_N,"v1_v2_preserved":True}
    (cfg.RESULTS_DIR/"run_manifest_v25.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=="__main__":main()

