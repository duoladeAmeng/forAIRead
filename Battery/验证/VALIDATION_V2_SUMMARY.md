# CALCE Validation v2 科研总结

## 最终判断

**Conclusion B — Moderate Support：Horizon dependency exists but is state-, redundancy-, and dataset-dependent.**

一句不夸大的论文级结论：

> 在 CALCE CS2-35/36/37/38 上，严格控制 horizon/feature 支持集后，HF5/HF6 的偏相关强度与特征排名仍呈显著的步长相关重排；但排名相似度随 horizon 距离下降的证据消失，且相对 SOH-history 的预测增益较小且不随 horizon 单调增加，因此当前证据支持“条件性的步长依赖”，而不是普遍、稳定的 horizon-only 特征权重规律。

## 1. v1 审计结论

v1 的 correlation/partial correlation 对每个特征独立 `dropna`，输出的 `n` 却统一使用 `y.notna().sum()`，并非真实样本数；不同 horizon 使用 `t≤T-h`，所以短 horizon 多使用尾部退化 cycle；HF4–HF6 缺失率为 34.9%–46.5%；SOH baseline 仅使用 `SOH_t`；RidgeCV 没有 battery-grouped 内层验证；circular bootstrap 在 `dropna` 后的压缩序列上采块。

v2 已修复真实 `n_effective`，并分别实现 pairwise、horizon-common、feature-common 和严格 double-common support。v1 的 `results/`、`figures/` 和四份 processed CSV 未被覆盖或修改。

## 2. 支持集控制

Double-common support 要求：当前 `SOH_t`、HF1–HF8 以及全部九个 horizon 的未来 SOH 同时有效，并统一限制 `t≤T-64`。因此同一电池的所有 HF、所有 horizon 使用完全相同的 current-cycle 样本：

| Battery | Double-common n |
|---|---:|
| CS2-35 | 551 |
| CS2-36 | 587 |
| CS2-37 | 616 |
| CS2-38 | 528 |

相比之下，v1-style pairwise 支持中 HF4–HF6 只有约 549–642 个样本，而 HF1/HF7 可使用约 814–1028 个样本。v1 的跨特征排名确实混合了不同生命周期支持区间。

## 3. HF5/HF6 长 horizon 增强

该现象在 double-common support 下仍存在，而且方向为 4/4 一致：

| Battery | HF5 h=1 | HF5 h=64 | HF6 h=1 | HF6 h=64 |
|---|---:|---:|---:|---:|
| CS2-35 | 0.027 | 0.270 | 0.091 | 0.341 |
| CS2-36 | 0.030 | 0.151 | 0.040 | 0.193 |
| CS2-37 | 0.019 | 0.209 | 0.029 | 0.277 |
| CS2-38 | 0.016 | 0.180 | 0.021 | 0.197 |

跨电池平均 `|P(HF6)|` 从 0.045 增至 0.252；HF5 从 0.023 增至 0.202。`Δ|P|(64)` 在四种支持规则下数值接近：HF5 每块电池约 +0.12 至 +0.24，HF6 约 +0.15 至 +0.27。因此 HF5/HF6 的增强并不是由 h=1/h=64 生命周期范围不同或跨特征支持集差异单独造成。

需要保留的限制是：偏相关强度并不等于不可替代的模型贡献；HF4–HF6 只在较早、可覆盖指定电压区间的生命周期区域定义。

## 4. Ranking crossover 与 null test

Double-common support 中检测到 826 个 battery-level crossover，56 个 feature-pair/horizon-pair 事件在至少 3/4 电池重复。典型 4/4 事件包括 HF2↔HF5 的 h=1→24、h=1→48，以及 HF5↔HF8 的 h=1→24。

Null hypothesis 为“各特征不存在共享的 horizon-specific 排名对齐”。置换保留每个特征完整的 horizon rank profile 和边际 rank 分布，但在每块电池内对每个特征独立作循环 horizon 位移；不打乱 cycle-level 时间序列。

| Statistic | Observed | Null mean ± SD | Empirical p |
|---|---:|---:|---:|
| Crossover count | 826 | 545.0 ± 30.1 | 0.000999 |
| Replicated ≥3/4 count | 56 | 12.3 ± 4.6 | 0.000999 |
| Mean distance-vs-Kendall rho | +0.145 | ≈0.000 ± 0.112 | 0.914 |

因此 crossover 的数量与跨电池复现程度超过 null expectation；但“相隔越远的 horizon 排名越不相似”没有超过 null expectation。

## 5. Kendall horizon similarity

这是 v1 中明显减弱、并在最严格控制后消失的结论：

- Horizon-common：四块电池仍为负（−0.093、−0.444、−0.772、−0.366）。
- Feature-common：+0.170、+0.150、−0.019、+0.139。
- Double-common：+0.266、+0.213、−0.057、+0.157。

可见负趋势主要依赖 feature support/lifecycle coverage，而不只是短 horizon 多出的尾部 63 个 cycle。严格支持集下只有 CS2-37 保持微弱负值，故 Evidence 3 不成立。

## 6. 强 SOH baseline

主预测结果使用固定 `alpha=1`；同时完整报告 `{0.01,0.1,1,10,100}`，不做 cycle-level 随机 CV，也不基于 held-out battery 选 alpha。各 alpha 下主要结论一致。

| h | Persistence | Current SOH | SOH history L=32 | History L=32 + current HF |
|---:|---:|---:|---:|---:|
| 1 | 0.01188 | 0.01547 | 0.01479 | 0.01370 |
| 8 | 0.01871 | 0.01897 | 0.01740 | 0.01649 |
| 24 | 0.02999 | 0.02532 | 0.02169 | 0.02103 |
| 32 | 0.03529 | 0.02786 | 0.02356 | 0.02294 |
| 48 | 0.04634 | 0.03035 | 0.02541 | 0.02506 |
| 64 | 0.05740 | 0.03296 | 0.02764 | 0.02701 |

L=32 history 相比 current-SOH Ridge 的平均 MAE 改善为 h=1 的 4.4% 和 h=24–64 的约 14.4%–16.2%；除 h=1 的一块电池外，其余比较均为 4/4 改善。Persistence 在 h=1 最强，但随 horizon 快速恶化。

SOH level/trend baseline 没有优于完整 L=32 history，说明压缩的均值、标准差、斜率和差分不足以取代原始历史窗口。

## 7. HF 相对强 baseline 的增量

SOH-history+HF 相比 SOH-history 的平均相对改善为：

| h | Relative improvement |
|---:|---:|
| 1 | 7.36% |
| 4 | 3.82% |
| 8 | 5.27% |
| 12 | 4.27% |
| 16 | 3.15% |
| 24 | 3.06% |
| 32 | 2.65% |
| 48 | 1.35% |
| 64 | 2.35% |

改善在 h=1–24 和 h=64 为 4/4 电池；h=32/48 为 3/4，例外均为 CS2-37（−0.47%、−1.37%）。五个固定 alpha 下正/负模式完全一致。

因此 HF 在强 SOH history 后仍有小而稳定的增量，但 v1 根据绝对 MAE 差得到的“增益随 horizon 增大”不能成立。归一化后增益总体下降，并非 horizon 越长 HF 的整体预测价值越大。

## 8. Grouped LOFO

在 `history_L32 + all current HF` 的完整模型中：

- Group_CC（HF1+HF7）在 h=1–48 为 4/4 正贡献，平均 LOFO 从 0.000776 降至 0.000425；h=64 为 3/4。
- Group_EnergyEfficiency（HF3）在 h=1–24 为 4/4，h=32–64 为 3/4；贡献较稳定但逐渐变小。
- Group_DischargeIC（HF4+HF5+HF6）在 h=1 为 4/4，h=8–32/64 为 3/4，平均贡献从 0.000135 增至 h=24/32 的约 0.00035，h=64 为 0.000315；存在中长步长相对上升，但跨电池并不完全稳定。
- Group_CV（HF2+HF8）大多接近 0 或为负，显示其信息可由其他输入替代。

这构成 feature-group importance 随 h 重分配的预测证据，但强度较小，不满足 Strong Support 所要求的稳定跨电池重排。

## 9. HF4–HF6 missingness sensitivity

缺失版本 B（median imputation + M4/M5/M6）相对删除 HF4–HF6 的版本 A，在 29/36 个 battery×horizon 组合中改善；平均优势从 h=8 的约 0.00015 MAE 增至 h=64 的约 0.00132。单独增加 indicator 相比保留 HF4–HF6 但无 indicator，只带来约 −0.00004 至 −0.00028 MAE 的小幅改善。

`[SOH,M4,M5,M6]` 的 missing-mask-only 模型在 0/36 个组合中优于不含 HF4–HF6、但包含 HF1/HF2/HF3/HF7/HF8 的版本 A，平均差约 +0.0028 至 +0.0034 MAE。因此 missing mask 确有少量状态信息，但不是主要预测 shortcut；HF4–HF6 的数值本身提供了额外信息。

Complete-case 模型的 MAE 更低，但它只测试较早、较容易的 lifecycle region，不能与全生命周期版本直接比较，也不能作为“插值策略更好”的证据。

## 10. Bootstrap 改进

非循环 moving-block bootstrap 使用原始 cycle index 划分连续有效段，块不跨缺失 gap，也不首尾 wrap。B=1000，块长 10/20/30。

与 v1 circular CI 相比，non-circular CI 并未系统性变宽：全部比较中 48.1% 更宽；平均 partial-correlation CI 宽度在块长 20 时为 0.298，v1 circular 为 0.364。差异说明首尾环接和压缩序列会改变不确定性，但没有证据表明 v1 一律低估 CI。v2 的 segmented non-circular 结果应作为主要敏感性结果。

## 11. 对 Validation v2 十二个问题的回答

1. **v1 哪些仍成立？** HF5/HF6 长 horizon 增强、明显 ranking crossover、HF 对 current-SOH 的增量仍成立。
2. **哪些减弱/消失？** horizon distance 与 Kendall similarity 的负相关消失；HF 整体预测增益随 horizon 增大的说法在归一化后消失。
3. **HF5/HF6 是否仍增强？** 是，double-common 下 4/4 一致。
4. **Crossover 是否超过 null？** 是，总数和 ≥3/4 复现数均 `p=0.000999`。
5. **Kendall 是否仍随距离下降？** 否，double-common 为 3 正 1 微负，null `p=0.914`。
6. **SOH-history 强多少？** 长 horizon 相比 current-SOH Ridge 约改善 14%–16%。
7. **加入 HF 后还有增量吗？** 有，多数 horizon 为 4/4，h=32/48 为 3/4。
8. **绝对还是相对？** 两者均计算；应以相对改善为主，其范围约 1.35%–7.36%，且不随 h 单调增加。
9. **Grouped LOFO 是否随 h 改变？** 有一定重分配，DischargeIC 相对增强、CC/Efficiency 相对减弱，但跨电池不完全稳定。
10. **HF4–HF6 missingness 是否造成 shortcut？** 不是主要来源；mask-only 很弱，但 complete-case 的生命周期选择效应必须警惕。
11. **是否足以进入 horizon-aware model？** 足以做受约束的下一阶段原型与外部验证，不足以宣称 horizon-only weighting 已获强支持。
12. **更支持 `w_j(h)` 还是 `w_j(X,h)`？** 更支持 **`w_j(X,h)`**。严格 feature-common support 消除了 Kendall 距离趋势，说明相关性不仅依赖 h，也依赖可观测状态、生命周期区域、冗余和缺失机制。下一阶段宜让权重同时依赖状态与 horizon，并对缺失 mask、feature group 和 SOH-history 做显式消融。

## 12. 交付与复现

- 入口：`run_analysis_v2.py`
- 配置：`config_v2.py`
- 新模块：`support_analysis.py`、`history_baselines.py`、`bootstrap_v2.py`、`null_test.py`、`plotting_v2.py`
- 结果：`results_v2/`（41 个 CSV + 1 个运行 manifest JSON）
- 图：`figures_v2/`（10 PNG + 10 PDF，PNG 320 dpi）
- Null 样本缓存：`cache_v2/ranking_null_samples.csv`

一次命令复现：

```powershell
& 'E:\CodeDir\Battery\.venv\Scripts\python.exe' run_analysis_v2.py
```
