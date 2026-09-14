你现在要在我已有的 CALCE CS2 步长感知特征价值验证项目上进行 Validation v2。

仓库位置：

Battery/验证/

现有主要文件包括：

config.py
run_analysis.py

src/
    load_data.py
    extract_features.py
    statistical_analysis.py
    bootstrap.py
    prediction_probe.py
    plotting.py

processed/
results/
figures/

现有代码已经完成：
- CS2-35 / 36 / 37 / 38 原始 Excel 解析
- SOH 与 HF1-HF8 提取
- future SOH correlation
- future degradation correlation
- partial correlation controlling SOH(t)
- feature ranking / crossover
- Kendall horizon similarity
- circular moving-block bootstrap
- leave-one-battery-out Ridge prediction probe
- leave-one-feature-out 分析

请不要重新实现整个项目，也不要修改已有 HF1-HF8 的主定义。

本轮 Validation v2 的目标是：

\[
\boxed{
排除样本支持集变化、特征缺失、弱 SOH baseline 和共线性
对“feature relevance 随 forecasting horizon 改变”这一结论造成的混杂
}
\]

最终需要回答：

\[
\boxed{
在更加严格、公平的实验设置下，
horizon-dependent feature relevance 是否仍然存在？
}
\]

==================================================
一、先审查当前代码，不要立即修改
==================================================

首先阅读：

- config.py
- run_analysis.py
- src/statistical_analysis.py
- src/bootstrap.py
- src/prediction_probe.py
- src/extract_features.py

并确认当前实现。

特别注意以下已知问题：

1. 当前 partial correlation / correlation 每个 feature 独立 dropna，
   不同 HF 实际使用的 cycle 数可能不同。

2. 当前输出 CSV 中的 n 很可能只是：
   y.notna().sum()
   而不是某个 feature 真正参与 correlation 的有效样本数。

3. h=1 与 h=64 当前使用的 current-cycle 范围不同。

4. HF4-HF6 存在大量与 degradation state 相关的结构性缺失。

5. 当前 SOH-only Ridge baseline 只使用当前 SOH_t，
   没有使用 SOH 历史窗口。

6. HF1 与 HF7 高度冗余；
   HF2 与 HF8 也可能存在明显冗余。

7. 当前 bootstrap 是 circular block bootstrap，
   并且在 dropna 后的压缩序列上运行。

完成审查后，在日志中先输出：

Validation v2 audit:
- current correlation support rule
- current horizon support rule
- actual missing rate per HF
- current prediction baseline
- current bootstrap type

然后再开始修改。

==================================================
二、保留 v1，不覆盖已有结果
==================================================

不要覆盖：

results/
figures/

中的现有 v1 结果。

新增：

results_v2/
figures_v2/

如果需要中间缓存，可增加：

cache_v2/

所有 Validation v2 的结果必须与旧实验并存。

==================================================
三、修复真实有效样本数 n
==================================================

修改 correlation / partial correlation 逻辑。

对于每一个：

battery
feature
horizon
metric

必须记录真正参与计算的：

n_effective

例如 partial correlation：

\[
Corr(HF_j(t), SOH_{t+h} \mid SOH_t)
\]

只有 HF_j(t)、SOH_t、SOH_{t+h} 三者全部非缺失的 cycle 才计入。

不要再将统一的 y.notna().sum() 当作所有 feature 的 n。

结果 CSV 至少包含：

battery
feature
horizon
metric
value
n_effective

生成：

results_v2/partial_correlation_pairwise.csv

以及相应 future SOH / degradation correlation 文件。

==================================================
四、Horizon Common Support Analysis
==================================================

这是 Validation v2 的核心。

设：

\[
H_{max}=64
\]

对于同一块电池，所有 horizon：

\[
h\in\{1,4,8,12,16,24,32,48,64\}
\]

必须使用相同的 current-cycle 候选范围。

如果电池长度为 T，则统一：

\[
t \leq T-H_{max}
\]

也就是说：

h=1 不允许额外使用最后 63 个 current cycles。

这样：

\[
P_j(1)
\]

和：

\[
P_j(64)
\]

才能在相同生命周期支持集上比较。

重新计算：

- Pearson future SOH correlation
- Spearman future SOH correlation
- future degradation correlation
- partial correlation controlling SOH_t
- feature ranking
- ranking crossover
- Kendall horizon similarity

输出文件加前缀：

common_horizon_support_

例如：

results_v2/common_horizon_support_partial.csv
results_v2/common_horizon_support_rankings.csv
results_v2/common_horizon_support_crossovers.csv
results_v2/common_horizon_support_kendall.csv

==================================================
五、Feature Common Support Analysis
==================================================

当前 HF4-HF6 缺失远多于 HF1/HF7。

因此直接对不同 feature 使用各自 pairwise complete samples 后再进行排名，并不完全公平。

增加严格的：

\[
\boxed{
feature-common-support
}
\]

对于固定：

battery + horizon

如果要比较 HF1-HF8，则只使用同时满足：

SOH_t 有效
SOH_{t+h} 有效
HF1-HF8 全部有效

的 cycle。

然后在这一完全相同的样本集合中，计算：

\[
P_j(h)
\]

以及：

\[
Rank_j(h)
\]

确保同一个 battery/horizon 下：

HF1-HF8 的 n_effective 完全相同。

输出：

results_v2/feature_common_support_partial.csv
results_v2/feature_common_support_rankings.csv
results_v2/feature_common_support_crossovers.csv
results_v2/feature_common_support_kendall.csv

同时保存每个：

battery
horizon

最终剩余的样本数。

如果某组样本量过小，明确标记，不强行解释。

==================================================
六、Double Common Support
==================================================

再做最严格版本：

同时满足：

1. horizon-common-support
2. feature-common-support

即所有 horizon 使用相同 current-cycle 范围，
所有 HF 在同一 horizon 使用相同有效 cycle。

称为：

DOUBLE COMMON SUPPORT

这是 Validation v2 中最重要的统计结果。

输出：

results_v2/double_common_partial.csv
results_v2/double_common_rankings.csv
results_v2/double_common_crossovers.csv
results_v2/double_common_kendall.csv

重点比较：

v1
vs
horizon-common-support
vs
feature-common-support
vs
double-common-support

看 horizon-dependent ranking 是否仍存在。

==================================================
七、重新评价 HF5 / HF6 长 horizon 增强
==================================================

重点分析：

HF5
HF6

对于每块电池画：

\[
|P_j(h)|
\]

比较四种 support rule：

- v1 pairwise
- horizon common
- feature common
- double common

重点回答：

HF6 从短 horizon 到长 horizon 增强的现象是否仍存在？

不能只报告 h=1 与 h=64 数值。

还需要计算：

\[
\Delta P_j(h)=|P_j(h)|-|P_j(1)|
\]

并输出跨电池结果。

==================================================
八、SOH History Strong Baseline
==================================================

当前 SOH-only 只用 SOH_t，不够强。

增加历史窗口：

\[
L\in\{8,16,32\}
\]

主结果优先使用：

\[
L=32
\]

构造：

\[
X_t^{SOH}
=
[SOH_{t-L+1},...,SOH_t]
\]

预测：

\[
SOH_{t+h}
\]

至少比较：

A. Persistence baseline

\[
\hat SOH_{t+h}=SOH_t
\]

B. Current-SOH Ridge

\[
SOH_t\rightarrow SOH_{t+h}
\]

C. SOH-history Ridge

\[
SOH_{t-L+1:t}\rightarrow SOH_{t+h}
\]

D. SOH-history + current HF

\[
[SOH_{t-L+1:t},HF_t]\rightarrow SOH_{t+h}
\]

E. SOH-history + HF-history

\[
[SOH_{t-L+1:t},HF_{t-L+1:t}]
\rightarrow SOH_{t+h}
\]

如果 E 特征维度过大，可以首先只做 L=16/32。

仍然严格：

leave-one-battery-out

所有：

- imputer
- scaler
- model fitting
- hyperparameter selection

都不能看到 held-out battery。

==================================================
九、Ridge 超参数避免 cycle-level 内部 CV 偏乐观
==================================================

当前 RidgeCV 默认 CV 不考虑 battery grouping。

Validation v2 中不要使用普通 cycle-level random CV 选择 alpha。

优先实现：

INNER LEAVE-ONE-BATTERY-OUT

例如外层：

train = CS2-35/36/37
test = CS2-38

则 alpha 的选择必须在：

35 / 36 / 37

内部再做 battery-level validation。

不能随机拆 cycle。

如果实现复杂，也允许：

固定一组 alpha

\[
\{0.01,0.1,1,10,100\}
\]

分别运行，并报告 sensitivity。

但不能使用普通 random cycle CV。

==================================================
十、加入 SOH Trend Baseline
==================================================

为了检验 HF 是否只是替代 SOH trend，额外构造简单趋势特征。

使用过去 L=32 个 SOH：

- current SOH
- mean SOH
- std SOH
- linear slope
- last difference
- mean first difference

例如：

\[
Slope_{SOH}(t)
\]

通过窗口内 cycle index 对 SOH 做线性拟合。

构造：

SOH_level_trend baseline

然后比较：

SOH-history
SOH-level-trend
SOH-history + HF

重点回答：

\[
HF
\]

在已经知道当前 SOH 和历史退化趋势后是否仍有预测增量。

==================================================
十一、Grouped LOFO
==================================================

由于 HF1/HF7 高度冗余，不要只做单特征 LOFO。

增加 feature groups：

Group_CC:
HF1 + HF7

Group_CV:
HF2 + HF8

Group_EnergyEfficiency:
HF3

Group_DischargeIC:
HF4 + HF5 + HF6

完整输入：

SOH + all HF

分别删除整个 group：

\[
V_g(h)
=
MAE_{-g}(h)-MAE_{full}(h)
\]

输出：

results_v2/grouped_lofo.csv

重点观察：

\[
Rank_g(h)
\]

是否随 horizon 改变。

相较 single-feature LOFO，
grouped LOFO 应作为主要预测价值证据。

==================================================
十二、HF4-HF6 结构性缺失敏感性分析
==================================================

必须至少做三个版本。

Version A：

完全删除 HF4/HF5/HF6。

输入：

SOH + HF1 + HF2 + HF3 + HF7 + HF8

Version B：

保留 HF4-HF6，
使用训练集 median imputation，
并为每个 HF 增加 missing indicator：

M4
M5
M6

例如：

M6=1 表示 HF6 缺失。

Version C：

complete-case restricted experiment

仅使用 HF4-HF6 均可定义的 lifecycle region。

比较三者。

重点判断：

horizon-aware evidence 是否高度依赖：

HF4-HF6 missingness pattern。

如果 Model B 明显优于 A，
还需要检查是否主要依赖 missing indicator 本身。

单独做：

missing-mask-only

预测 probe：

\[
[M4,M5,M6,SOH]
\rightarrow SOH_{t+h}
\]

如果 mask 本身预测力很强，必须在总结中明确报告。

==================================================
十三、Bootstrap 改进
==================================================

保留当前 circular block bootstrap 结果。

另外增加普通：

NON-CIRCULAR MOVING BLOCK BOOTSTRAP

不要首尾 wrap-around。

使用：

block length = 10,20,30
B = 1000

并且：

不要简单 dropna 后把原本相隔很远的 cycle 当作连续邻居。

需要保留原始 cycle index。

如果有效样本存在 gap：

- block 不应跨越大的 missing gap；
或者
- 明确使用 contiguous valid segments 分段 bootstrap。

输出：

results_v2/bootstrap_non_circular.csv

比较 circular vs non-circular CI。

==================================================
十四、Ranking Crossover Null Test
==================================================

当前检测到很多 ranking crossover，
但 crossover 数量本身受 feature pair × horizon pair 组合数量影响。

增加 null test。

目标：

检验真实数据中的：

- crossover count
- replicated >= 3/4 crossover count
- horizon-distance vs Kendall-tau trend

是否超过“无 horizon-specific ranking structure”下的随机水平。

设计一个合理的 permutation/null procedure。

重要：

必须尽量保留时间相关结构和每个 HF 的边际分布。

不要简单逐 sample 完全随机 shuffle 时间序列。

优先考虑：

- block permutation
或
- 在固定 feature trajectory 下置换 horizon labels / ranking profiles

并说明 null hypothesis。

至少进行：

1000 次 permutation

输出：

observed
null_mean
null_std
empirical_p

结果：

results_v2/ranking_null_test.csv

==================================================
十五、Normalized Prediction Improvement
==================================================

当前绝对增益：

\[
MAE_{baseline}-MAE_{model}
\]

会随着 baseline MAE 增大而自然增大。

因此增加：

\[
RI(h)
=
\frac{
MAE_{baseline}(h)-MAE_{model}(h)
}{
MAE_{baseline}(h)
}
\]

输出百分比：

relative_improvement_percent

对以下比较均计算：

current SOH vs SOH+HF

SOH-history vs SOH-history+HF

SOH-trend vs SOH-trend+HF

不要再仅根据 absolute MAE difference 得出：

“HF 价值随 horizon 增大”

这种结论。

==================================================
十六、推荐新增图
==================================================

生成 figures_v2：

Figure V2-1
effective sample size heatmap

feature × horizon

每块电池一张。

Figure V2-2
HF5/HF6 partial correlation under four support rules

Figure V2-3
double-common-support feature rank evolution

Figure V2-4
double-common-support Kendall horizon similarity

Figure V2-5
grouped LOFO vs horizon

Figure V2-6
SOH-current / SOH-history / SOH-history+HF MAE

Figure V2-7
relative improvement vs horizon

Figure V2-8
HF4-HF6 missing sensitivity

Figure V2-9
observed crossover statistic vs null distribution

Figure V2-10
v1 vs Validation-v2 conclusion comparison

图片保存：

PNG + PDF
dpi >= 300

==================================================
十七、Validation v2 的核心判据
==================================================

不要为了支持原假设选择性报告结果。

将结果分为：

A：Strong Support

如果同时满足：

1. double-common-support 下 feature ranking 仍明显随 horizon 改变；

2. horizon-distance 与 Kendall similarity 在多数电池中仍为负；

3. grouped LOFO 显示不同 feature group 的预测贡献随 horizon 重分配；

4. SOH-history + HF 相比 SOH-history 在多个 horizon、多数 held-out battery 上仍有稳定提升；

5. horizon-dependent 结果不是仅由 HF4-HF6 missingness 驱动。

则结论：

Strong evidence for horizon-dependent feature utility.

--------------------------------------------------

B：Moderate Support

如果：

统计 ranking 仍有 horizon dependency，

但 predictive contribution / cross-battery consistency 较弱，

则结论：

Horizon dependency exists but is state-, redundancy-, or dataset-dependent.

--------------------------------------------------

C：Weak / Unsupported

如果：

double-common-support 后 ranking 变化明显消失，

或者 SOH-history 已解释绝大部分 HF 增益，

或者 horizon effect 主要来自 HF4-HF6 missingness，

则明确结论：

Current evidence is insufficient to support horizon-aware feature modeling.

不能为了继续课题强行解释。

==================================================
十八、最终科研总结
==================================================

生成：

VALIDATION_V2_SUMMARY.md

必须回答：

1. v1 中哪些结论在严格 support control 后仍成立？

2. 哪些结论减弱或消失？

3. HF5/HF6 长 horizon 增强是否仍存在？

4. feature ranking crossover 是否超过 null expectation？

5. Kendall similarity 随 horizon distance 下降是否仍存在？

6. SOH-history 相比 SOH_t 强多少？

7. 加入 HF 后，相比 SOH-history 是否仍有增量？

8. 这个增量是 absolute 还是 relative improvement？

9. grouped LOFO 是否显示 feature-group importance 随 h 改变？

10. HF4-HF6 missingness 是否造成 shortcut？

11. 是否仍有充分依据进入 horizon-aware model？

12. 如果进入模型阶段，更支持：

\[
w_j(h)
\]

还是：

\[
w_j(X,h)
\]

给出明确理由。

最后必须给出：

Conclusion A / B / C

以及一句不夸大的论文级结论。

==================================================
十九、开发要求
==================================================

不要只写代码。

必须：

- 实际运行
- 修复错误
- 检查结果
- 生成 CSV
- 生成图片
- 写科研总结

不要修改原始 processed CSV。

不要改变 HF 定义来让结果更好看。

新增函数应尽量模块化。

建议新增：

src/support_analysis.py
src/history_baselines.py
src/grouped_ablation.py
src/null_test.py

也可以在现有模块中合理扩展。

运行入口仍建议：

run_analysis_v2.py

所有新增参数加入 config.py 或独立 config_v2.py。

最终确保一次命令可复现 Validation v2。