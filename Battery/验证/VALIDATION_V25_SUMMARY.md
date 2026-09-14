# CALCE Validation v2.5 科研总结

## 最终判断

**Conclusion A：Strong State + Horizon Support（集中在 Group_DischargeIC，且状态效应非单调）。**

论文级结论：

> 在强 SOH-history 基线和 leave-one-battery-out 评估下，完整 HF-history 没有提供稳定增益；相反，当前 feature-group 的边际预测价值表现出可跨电池复现的 degradation-state × forecast-horizon interaction，主要由 Group_DischargeIC 驱动。因此数据支持对当前特征组采用受约束的 State+Horizon routing，但不支持无差别引入高维 HF 历史序列。

## Q1：HF-history 是否比 current HF 更有价值？

否。v2 已经包含以下三个模型，本轮直接复用结果，没有重新实现：

- A：`history_L32`
- B：`history_L32_currentHF`
- C：`history_L32_HFhistory`

固定 alpha=1 的跨电池结果：

| h | Current-HF 相对 A | HF-history 相对 A | HF-history 相对 Current-HF | HF-history 优于 Current-HF 的电池数 |
|---:|---:|---:|---:|---:|
| 1 | +7.36% | −7.12% | −15.64% | 0/4 |
| 4 | +3.82% | −5.32% | −9.52% | 0/4 |
| 8 | +5.27% | −4.65% | −10.52% | 0/4 |
| 12 | +4.27% | −6.16% | −10.89% | 0/4 |
| 16 | +3.15% | −5.97% | −9.43% | 0/4 |
| 24 | +3.06% | −5.17% | −8.53% | 0/4 |
| 32 | +2.65% | −2.05% | −4.90% | 1/4 |
| 48 | +1.35% | −3.12% | −4.72% | 2/4 |
| 64 | +2.35% | −0.06% | −2.55% | 2/4 |

HF-history 在短中 horizon 明显劣于 current HF；到 h=64 只是接近 SOH-history baseline，仍没有跨电池稳定优势。结论是：高维 HF-history 更像增加维度、共线和插补负担，而不是提供稳定的时序信息。下一阶段不应默认把 32×8 个 HF lag 全部送入深度模型。

## 因果 State 定义

全部 state 特征只使用 `≤t` 的信息：

- `current_soh = SOH_t`
- `soh_slope`：SOH(t−31:t) 线性斜率
- `mean_degradation_rate`：过去 31 个一阶下降量的均值
- `recent_degradation_rate`：最近 7 个一阶下降量的均值
- `volatility`：过去 31 个 SOH 一阶差分的标准差

主定义 `degradation_dynamics` 使用 `(-slope + recent degradation rate)/2`，并用截至当前时刻的 expanding tertiles 标记 Slow/Medium/Fast。四块电池分别得到约 216–355 个样本/状态，状态分布足以进行分层测试。

SOH-level sensitivity 使用预先固定的物理区间（SOH≥0.8、0.5–0.8、<0.5），没有使用整块 held-out battery 的未来分布来估计 tertile。Cycle-position 对照使用预先声明的 1000-cycle reference，而不是最终 `total_cycles`，避免从实验终点获得未来信息。泄漏审计中所有 `max_source_time_offset=0`。

## Q2：Feature utility 是否依赖 State 和 Horizon？

是，但主要集中在 `Group_DischargeIC`，并且不是简单的 Slow→Medium→Fast 单调规律。

同一个模型在三块训练电池上拟合，然后在 held-out battery 内按 causal state 切片；没有为每个 state 单独训练模型。完整模型为 `SOH-history L32 + all current HF`。

### Group_DischargeIC

跨电池平均 LOFO value：

- Slow：h=12–64 多数为负，h=48 为 −0.00148，表示在 Slow state 中删除 IC 组反而可能改善预测。
- Medium：从 h=12 开始成为最重要组，h=24/32/64 分别约 0.00082、0.00070、0.00065。
- Fast：短 horizon 接近 0 或略负，中长 horizon 转为小幅正值，但通常低于 Medium。

Medium > Slow 的方向在 h=12、16、24、32 为 4/4 电池，在 h=48、64 为 3/4。Fast > Slow 在 h=48/64 为 3/4。因此 state effect 具有跨电池复现性，但最佳状态是 Medium，而不是预设的 Fast。

### 其他 groups

- Group_CC 在多数 state/horizon 中仍为正，Fast 和 Slow 状态通常由 CC 排名第一；其 State×Horizon interaction 不显著。
- Group_CV 的平均边际价值很小或为负，虽检测到交互，但实际效应应谨慎解释。
- Group_EnergyEfficiency 较稳定，没有显著 interaction。

## State × Horizon permutation test

检验统计量是各电池 state×horizon utility matrix 去除 state/horizon 可加主效应后的 interaction pattern，其 effect size 衡量跨电池平均 interaction pattern 占总 interaction energy 的比例。Null hypothesis：不存在跨电池对齐的 state-by-horizon interaction。置换在每块电池、每个 state 内独立重排 horizon，保留 feature-group utility 的边际分布，不打乱 cycle 样本。

| Feature group | Effect size | Battery consistency | Empirical p |
|---|---:|---:|---:|
| Group_DischargeIC | 0.439 | 0.231 | **0.006** |
| Group_CV | 0.378 | 0.128 | **0.047** |
| Group_EnergyEfficiency | 0.266 | 0.048 | 0.372 |
| Group_CC | 0.230 | 0.052 | 0.598 |

主要可信交互来自 Group_DischargeIC。Group_CV 虽刚好达到 0.05，但 consistency 和绝对 LOFO 很小，不作为建模主依据。

## 同一 horizon、不同 state

Dynamics-state 的 group ranking 在短 horizon 较一致：h=1/4 时三种 state 的 top group 均为 Group_CC。到 h=12–64：

- Medium 的 top group 变为 Group_DischargeIC；
- Slow 与 Fast 的 top group 仍通常为 Group_CC；
- h=12 时 Slow↔Medium、Medium↔Fast 的 top group 在 4/4 电池改变；
- h=32–64 的 state-pair Kendall tau 常接近 0 或为负，显示相同 horizon 下 state 会显著改变 feature-group 排名。

全部 battery×horizon×state-pair 中 top group 改变率为 44.4%，变化集中于中长 horizon。

## 同一 state、不同 horizon

- Slow：CC 始终主导，DischargeIC 在中长 horizon 常为负。
- Medium：h≤8 时 CC 主导，从 h=12 起 DischargeIC 超过 CC。
- Fast：CC 始终主导，DischargeIC 只在中长 horizon 获得小幅正贡献。

因此 v2 的“DischargeIC 中长 horizon 增强”不是所有 state 中相同的全局规律，而主要由 Medium dynamics state 驱动。这正是 State×Horizon routing 相比 horizon-only weighting 的实质增量。

## State-conditioned prediction gain

`SOH-history+HF` 相对 `SOH-history` 的平均相对改善：

- h=1：Slow 10.87%、Medium 10.03%、Fast 7.45%，均为 4/4 正改善。
- h=24：Slow 2.18%、Medium 4.02%、Fast 0.94%；Medium 为 4/4 正，Slow/Fast 为 3/4。
- h=48：Slow −2.61%、Medium 1.87%、Fast 0.62%；Medium/Fast > Slow 均为 3/4。
- h=64：Slow −1.00%、Medium 2.22%、Fast 0.85%；Medium/Fast > Slow 均为 3/4。

长 horizon 下 HF 对 Slow state 没有稳定收益，而对 Medium/Fast 仍常有正收益，说明显式 state-aware 使用比统一加权更合理。不过跨电池的增益幅度差异较大，应使用软路由或正则化 gating，不宜硬编码某一状态必选某一组。

## Partial-correlation 辅助证据

Feature-common within-state partial correlation 使用同一 battery/state/horizon 下 HF1–HF8 全部有效的共同样本；`n<30` 标为 unreliable。共 2592 行中 392 行不可靠，主要来自 SOH-level 极端区间和部分 HF4–HF6 有效区间，因此偏相关只作为描述性证据。

在可靠 dynamics-state 样本中，Medium state 的 Group_DischargeIC 平均绝对偏相关从 h=1 的 0.063 增至 h=64 的 0.231；Fast 从 0.080 增至 0.215；Slow 从 0.052 增至 0.176。方向上支持长 horizon 增强，但 prediction LOFO 表明 Slow state 的 IC 数值可能相关却没有正的不可替代预测贡献，再次说明主结论应以 LOFO 为准。

## Lifecycle confounding

Degradation-dynamics 与固定 cycle-position 的区分不是绝对的：

- DischargeIC 的 mean state range：dynamics 0.00109，cycle-position 0.00100；between-state variance ratio 分别为 0.276 与 0.256。
- EnergyEfficiency：dynamics ratio 0.404，cycle-position 0.126，dynamics 明显更强。
- CC：dynamics state range 0.00038，小于 cycle-position 的 0.00071，但 variance ratio 接近。
- SOH-level 对各组的区分整体较弱。

因此 dynamics state 并非 cycle position 的简单复制，但两者仍明显相关。State+Horizon 结论主要由 DischargeIC 的交互置换结果和跨电池方向一致性支撑，不应声称 lifecycle confounding 已被完全消除。

## 最终五项决策

1. **是否有必要使用 State？** 有。Group_DischargeIC 在 Medium 与 Slow/Fast state 的中长 horizon utility 明显不同，且 interaction `p=0.006`。
2. **是否有必要使用 Horizon？** 有。短 horizon 三种 state 均由 CC 主导，而 Medium state 在 h≥12 转为 IC 主导。
3. **二者是否存在 interaction？** 存在，主要是 Group_DischargeIC；不是所有 feature group 都存在。
4. **推荐模型：** **State+Horizon**，即受约束的 `w_g(X,h)` group routing，而不是单 HF、静态或纯 horizon-only weighting。
5. **是否值得进入深度模型实验？** 值得进入小规模、强基线约束的原型实验。建议保留 L=32 SOH-history encoder，只路由 current feature groups；先比较 Static、Horizon-only、State-only、State+Horizon 四个嵌套模型。HF-history 不应默认加入，除非新的正则化/降维消融证明其价值。

## 交付物

- `results_v25/hf_history_value.csv`
- `results_v25/hf_history_value_summary.csv`
- `results_v25/state_conditioned_group_lofo.csv`
- `results_v25/state_conditioned_partial.csv`
- `results_v25/state_horizon_interaction_test.csv`
- `results_v25/state_rankings.csv`
- `results_v25/state_rank_crossovers.csv`
- `results_v25/state_leakage_audit.csv`
- `results_v25/state_definition_comparison.csv`
- `results_v25/state_conditioned_prediction_gain.csv`
- `figures_v25/`：8 张 PNG + 8 张 PDF，PNG 320 dpi
- `cache_v25/`：causal state 表和 interaction null samples

复现命令：

```powershell
& 'E:\CodeDir\Battery\.venv\Scripts\python.exe' run_analysis_v25.py
```

