你是一名熟悉锂离子电池 SOH 预测、时间序列分析、统计检验和 Python 科研编程的研究助手。

我正在研究一个现象：

\[
\boxed{
\text{不同输入健康特征对未来 } SOH_{t+h} \text{ 的预测价值会随预测步长 } h \text{ 改变}
}
\]

也就是说，我希望验证：

\[
V_j(h)
=
\text{第 }j\text{ 个特征对 }SOH_{t+h}\text{ 的预测价值}
\]

是否是 prediction horizon \(h\) 的函数，而不是一个固定常数。

当前阶段的目标是“验证现象”，不是设计复杂深度学习模型。请优先使用统计分析和简单 prediction probe，暂时不要使用 Transformer、LSTM 等复杂模型。

---

# 一、数据

我已经下载好马里兰大学 CALCE Battery Research Group 的 CS2 数据集。

重点研究四块电池：

- CS2-35
- CS2-36
- CS2-37
- CS2-38

请首先自动检查我提供的数据目录，理解文件组织方式、MAT/CSV/XLSX 等文件格式、字段名称和各充放电 cycle 的数据结构。

不要预设文件格式或字段名称。

需要首先输出：

1. 找到哪些 CS2-35/36/37/38 数据文件；
2. 每个文件包含哪些字段；
3. 如何识别 cycle；
4. 如何识别恒流充电 CCCS、恒压充电 CVCS、恒流放电 CCDS；
5. 是否存在 Capacity、Voltage、Current、Time 等字段；
6. 原始数据中是否已经提供 cycle-level discharge capacity / SOH；
7. 四块电池分别有多少有效 cycle。

如果实际文件结构与预期不同，请根据数据实际结构自适应处理，而不是直接报错退出。

---

# 二、SOH 定义

当前阶段直接使用数据中的真实容量计算 SOH，不训练 SOH estimation 模型。

定义：

\[
SOH_t=\frac{C_t}{C_{\mathrm{nom}}}
\]

其中 \(C_t\) 为第 \(t\) 个 cycle 的实际放电容量。

请首先检查数据和官方实验设定，确定：

\[
C_{\mathrm{nom}}
\]

采用额定容量还是初始实际容量。

如果存在多种合理定义，请分别说明，并在主实验中固定一种定义，确保 CS2-35、36、37、38 使用完全一致的定义。

保存每块电池：

```text
cycle, SOH
```

的序列。

---

# 三、输入特征

主实验使用：

\[
X_t=
[SOH_t,HF1_t,HF2_t,\ldots,HF8_t]
\]

注意：

- 历史 SOH 本身保留为一个 autoregressive input feature；
- HF1-HF8 是 auxiliary health features；
- 不要因为某个 HF 和当前 SOH 相关性低就提前删除；
- 本实验的重要目的恰恰是观察“低当前相关性的 HF 是否可能在较长预测步长下变得更有价值”。

HF1-HF8 按以下定义提取。

---

## HF1：恒流充电阶段能量

恒流充电 CCCS：

\[
HF1=
\int_{CCCS} V(t)I(t)\,dt
\]

使用实际采样时间进行数值积分，例如 trapezoidal integration。

---

## HF2：恒压充电阶段能量

恒压充电 CVCS：

\[
HF2=
\int_{CVCS}V(t)I(t)\,dt
\]

---

## HF3：能量转换效率

首先计算恒流放电阶段释放能量：

\[
EF3=
\int_{CCDS}|V(t)I(t)|\,dt
\]

然后定义：

\[
HF3=
\frac{EF3}{HF1+HF2}
\]

注意统一电流正负号 convention，保证能量为正值。

---

## HF4：4.0 V 到 3.4 V 放电区间容量变化

在恒流放电 CCDS 阶段，计算电压从：

\[
4.0V\rightarrow3.4V
\]

过程中释放的容量：

\[
HF4=
Q(V=3.4)-Q(V=4.0)
\]

如果恰好没有 4.0 V 或 3.4 V 的采样点，请使用线性插值获得对应容量。

不要简单寻找最近采样点代替，除非插值不可行。

---

## HF5、HF6：增量容量特征

在 CCDS 阶段，根据：

\[
\frac{dQ}{dV}
\]

构建增量容量曲线。

由于原始数据为离散采样，可使用：

\[
\frac{dQ}{dV}\bigg|_k
\approx
\frac{Q_k-Q_{k-1}}{V_k-V_{k-1}}
\]

但原始差分会对噪声非常敏感，因此：

1. 先检查数据采样质量；
2. 必要时对 \(Q-V\) 曲线进行适度平滑；
3. 平滑参数不能根据最终结果人为调节；
4. 对所有四块电池使用一致规则；
5. 记录所使用的方法与参数。

定义：

\[
HF5=
\text{主要 }dQ/dV\text{ 峰值}
\]

\[
HF6=
\text{该峰值对应的电压}
\]

必须制定统一的 peak selection rule，避免每个 cycle 人工选择不同峰。

如果由于放电方向导致 \(dQ/dV\) 为负，需要统一符号或选择绝对峰值，并清楚记录规则。

---

## HF7：恒流充电时间

\[
HF7=T_{\mathrm{CCCS}}
\]

---

## HF8：恒压充电时间

\[
HF8=T_{\mathrm{CVCS}}
\]

---

# 四、首先检查特征质量

生成每块电池的：

```text
cycle
SOH
HF1
HF2
HF3
HF4
HF5
HF6
HF7
HF8
```

表格。

保存为独立 CSV。

例如：

```text
processed/
    CS2_35_features.csv
    CS2_36_features.csv
    CS2_37_features.csv
    CS2_38_features.csv
```

然后检查：

- 缺失值；
- 极端异常值；
- 无穷值；
- dQ/dV 数值爆炸；
- 重复 cycle；
- cycle 顺序错误。

现阶段不要为了得到漂亮结果随意删除异常 cycle。

如果需要删除或插值，必须：

1. 给出明确规则；
2. 报告删除了哪些 cycle；
3. 同一规则应用到全部电池；
4. 保留未经处理的原始特征结果。

---

# 五、预测步长设置

主实验使用：

\[
h\in\{1,4,8,12,16,24,32\}
\]

单位为 cycle。

如果某块电池数据长度允许，再额外计算：

\[
h=48,\quad64
\]

作为扩展实验，但不要替代主实验。

对于每个 horizon：

\[
Y_t^{(h)}=SOH_{t+h}
\]

只保留具有真实未来标签的样本：

\[
t=1,\ldots,T-h
\]

---

# 六、实验 A：当前特征与未来 SOH 的相关性

对于：

\[
X_j\in
\{SOH,HF1,\ldots,HF8\}
\]

分别计算：

\[
R_j^{P}(h)
=
Pearson(X_{j,t},SOH_{t+h})
\]

以及：

\[
R_j^{S}(h)
=
Spearman(X_{j,t},SOH_{t+h})
\]

注意：

SOH 与多数 HF 都会随 cycle 呈趋势，因此普通相关性只能作为第一层描述性证据，不能直接作为“预测价值”的最终证明。

输出每块电池：

```text
feature × horizon
```

的 Pearson 和 Spearman 矩阵。

---

# 七、实验 B：HF 与未来退化量之间的关系

定义未来 \(h\) 个 cycle 内的退化：

\[
D_{t,h}=SOH_t-SOH_{t+h}
\]

对 HF1-HF8 计算：

\[
G_j(h)
=
Corr(HF_{j,t},D_{t,h})
\]

同时计算：

- Pearson；
- Spearman。

本分析主要回答：

> 当前 HF 能否提供未来一段时间内“会退化多少”的信息？

重点观察：

\[
G_j(h)
\]

是否随 \(h\) 出现明显变化。

---

# 八、实验 C：控制当前 SOH 后的 partial correlation

这是本实验的核心分析之一。

对于：

\[
HF_j,\quad j=1,\ldots,8
\]

计算：

\[
P_j(h)
=
Corr(HF_{j,t},SOH_{t+h}\mid SOH_t)
\]

即控制当前：

\[
SOH_t
\]

之后，HF 与未来 SOH 的 partial correlation。

可以通过 residualization 实现：

第一步：

\[
HF_j=a_0+a_1SOH_t+\epsilon_x
\]

第二步：

\[
SOH_{t+h}=b_0+b_1SOH_t+\epsilon_y
\]

第三步：

\[
P_j(h)=Corr(\epsilon_x,\epsilon_y)
\]

也可以使用经过验证的 partial correlation 实现。

需要明确报告计算方法。

本实验回答：

> 已经知道当前 SOH 后，该 HF 是否仍然包含关于未来 SOH 的额外信息？

重点寻找类似现象：

\[
P_j(1)\approx0
\]

但是：

\[
|P_j(24)|,\ |P_j(32)|
\]

明显增加。

---

# 九、实验 D：Feature Ranking 随 horizon 的变化

对于每个 horizon：

\[
h
\]

分别根据以下指标排序：

1. \(|R_j^S(h)|\)
2. \(|G_j^S(h)|\)
3. \(|P_j(h)|\)

主要关注第 3 项 partial correlation ranking。

定义：

\[
Rank_j(h)
\]

观察是否存在：

\[
Rank_i(h_1)<Rank_j(h_1)
\]

但：

\[
Rank_i(h_2)>Rank_j(h_2)
\]

即：

\[
\boxed{\text{ranking crossover}}
\]

请自动检测所有 feature pair 的 ranking crossover，并输出：

```text
feature_A
feature_B
horizon_1
horizon_2
rank_A_h1
rank_B_h1
rank_A_h2
rank_B_h2
```

同时统计某一 crossover 是否在 CS2-35、36、37、38 中重复出现。

---

# 十、实验 E：不同 horizon 的特征排名相似度

对于每个：

\[
h_i,h_j
\]

使用 feature ranking 计算：

\[
Kendall\ \tau(h_i,h_j)
\]

重点使用：

\[
|P_j(h)|
\]

得到的 HF1-HF8 ranking。

构造：

\[
Horizon\times Horizon
\]

的 Kendall-\(\tau\) 矩阵。

分析是否存在：

\[
|h_i-h_j|\uparrow
\]

时：

\[
\tau(h_i,h_j)\downarrow
\]

的趋势。

如果存在，计算：

\[
|h_i-h_j|
\]

与：

\[
\tau(h_i,h_j)
\]

之间的 Spearman correlation，作为一个辅助量化指标。

---

# 十一、不要把四块电池直接拼接

CS2-35、36、37、38 必须首先独立分析。

分别得到：

\[
V_{35,j}(h)
\]

\[
V_{36,j}(h)
\]

\[
V_{37,j}(h)
\]

\[
V_{38,j}(h)
\]

然后再做跨电池汇总。

对于每个指标报告：

- 四块电池独立结果；
- mean；
- median；
- standard deviation；
- 跨电池一致性。

不要直接把四块电池的 cycle 拼接后计算一个 correlation。

---

# 十二、统计不确定性：Moving Block Bootstrap

电池 cycle 数据具有明显时间自相关，因此不要默认每个 cycle 是 iid 样本。

不要只使用普通 correlation 的 iid p-value。

对主要指标：

\[
G_j(h)
\]

和：

\[
P_j(h)
\]

采用 Moving Block Bootstrap 或 Circular Block Bootstrap 计算 95% confidence interval。

block length 可以首先尝试：

\[
L_b=20
\]

同时做：

\[
L_b=10,\ 20,\ 30
\]

的敏感性分析。

Bootstrap 次数建议：

\[
B=1000
\]

如计算量过大，开发调试阶段可先使用 200，最终结果必须使用至少 1000。

输出：

\[
estimate,\ CI_{2.5\%},\ CI_{97.5\%}
\]

不要因为某些结果不显著就修改 block size。

---

# 十三、简单 Prediction Probe

在完成统计现象分析之后，再加入一个非常简单的 prediction probe。

不要使用深度学习。

优先使用：

\[
Ridge Regression
\]

预测：

\[
SOH_{t+h}
\]

输入：

\[
[SOH_t,HF1_t,\ldots,HF8_t]
\]

为了避免时间泄漏和跨电池泄漏，使用 leave-one-battery-out：

例如：

```text
train: CS2-35, CS2-36, CS2-37
test:  CS2-38
```

轮流：

```text
35+36+37 -> 38
35+36+38 -> 37
35+37+38 -> 36
36+37+38 -> 35
```

所有标准化参数必须只在训练电池上拟合，然后应用到测试电池。

对于每个：

\[
h
\]

训练独立 Ridge predictor。

---

# 十四、Leave-One-Feature-Out Predictive Value

完整模型：

\[
M_{\mathrm{all}}
\]

使用：

\[
[SOH,HF1,\ldots,HF8]
\]

然后分别删除某个 feature：

\[
M_{-j}
\]

定义：

\[
V_j(h)
=
MAE_{-j}(h)-MAE_{\mathrm{all}}(h)
\]

也可以同时计算：

\[
RMSE_{-j}(h)-RMSE_{\mathrm{all}}(h)
\]

解释：

如果：

\[
V_j(h)>0
\]

说明删除该特征后预测性能下降，即该 feature 对该 horizon 有正向增量预测价值。

重点研究：

\[
V_j(h)
\]

是否随 horizon 改变，以及不同 feature 的 ranking 是否发生 crossover。

同时加入以下 baseline：

### Baseline 1

SOH-only：

\[
SOH_t\rightarrow SOH_{t+h}
\]

### Baseline 2

HF-only：

\[
HF1_t,\ldots,HF8_t
\rightarrow SOH_{t+h}
\]

### Baseline 3

SOH + HF：

\[
SOH_t,HF1_t,\ldots,HF8_t
\rightarrow SOH_{t+h}
\]

重点分析：

\[
\Delta E(h)
=
MAE_{\mathrm{SOH-only}}(h)
-
MAE_{\mathrm{SOH+HF}}(h)
\]

观察 auxiliary HF 相对于历史 SOH 的增量价值是否随 horizon 增大。

---

# 十五、需要生成的核心图

请使用 Python + matplotlib/seaborn 绘制适合论文使用的高分辨率图片。

每张图片同时保存：

```text
PNG
PDF
```

建议 dpi ≥ 300。

至少生成以下图。

---

## Figure 1：SOH 与 HF trajectory

对于四块电池分别画：

\[
SOH,HF1,\ldots,HF8
\]

随 cycle 的变化。

可以分别标准化后比较趋势，但原始尺度图也必须保存。

---

## Figure 2：Feature × Horizon correlation heatmap

分别画：

- Pearson；
- Spearman；
- partial correlation。

重点图为：

\[
|P_j(h)|
\]

heatmap。

---

## Figure 3：Horizon–Feature Value Curve

横轴：

\[
h
\]

纵轴：

\[
|P_j(h)|
\]

HF1-HF8 每个一条曲线。

需要：

1. 每块电池单独图；
2. 四块电池聚合图。

聚合图使用 mean/median line，并表示跨电池 variability。

不要把 SOH 与 partial correlation HF 图混淆，因为 SOH 是控制变量。

---

## Figure 4：Feature Rank Evolution

横轴：

\[
h
\]

纵轴：

\[
Rank
\]

绘制：

\[
HF1-HF8
\]

排名随 horizon 的变化。

排名 1 放在图顶部。

突出 ranking crossover。

---

## Figure 5：Horizon × Horizon Kendall-\(\tau\) Heatmap

利用：

\[
Rank(h)
\]

构建 horizon similarity heatmap。

---

## Figure 6：Future degradation correlation

绘制：

\[
Corr(HF_j,SOH_t-SOH_{t+h})
\]

随 \(h\) 的变化。

---

## Figure 7：Leave-One-Feature-Out Predictive Value

横轴：

\[
h
\]

纵轴：

\[
V_j(h)
=
MAE_{-j}-MAE_{\mathrm{all}}
\]

每个 HF 一条线。

如果 SOH 也作为被删除变量进行实验，则将其单独强调为：

```text
autoregressive SOH feature
```

而不是与 auxiliary HF 在概念上完全等同。

---

## Figure 8：SOH-only vs HF-only vs SOH+HF

横轴：

\[
h
\]

纵轴：

\[
MAE
\]

比较三种输入方案。

主要观察：

\[
SOH+HF
\]

相对于：

\[
SOH-only
\]

的增益是否随着 \(h\) 发生变化。

---

# 十六、结果表

至少生成以下 CSV：

```text
results/
    pearson_future_soh.csv
    spearman_future_soh.csv
    future_degradation_correlation.csv
    partial_correlation.csv
    feature_rankings.csv
    ranking_crossovers.csv
    kendall_horizon_similarity.csv
    bootstrap_confidence_intervals.csv
    ridge_metrics.csv
    leave_one_feature_out.csv
```

每张表必须包含：

```text
battery
feature
horizon
metric
value
```

等必要字段。

---

# 十七、成功判据

不要因为希望验证假设，就强行解释结果。

只有在数据支持的情况下才能认为存在明显的 horizon-aware / horizon-dependent feature relevance。

重点检查以下三类证据：

### Evidence 1

多个 HF 的：

\[
P_j(h)
\]

随 horizon 存在明显变化，并且 bootstrap CI 支持这种变化不是单纯噪声。

### Evidence 2

存在稳定 ranking crossover：

\[
Rank_i(h_1)<Rank_j(h_1)
\]

但是：

\[
Rank_i(h_2)>Rank_j(h_2)
\]

并且最好能够在至少 3/4 块电池中重复。

### Evidence 3

不同 horizon 之间的 feature ranking similarity：

\[
Kendall\ \tau(h_i,h_j)
\]

随着：

\[
|h_i-h_j|
\]

增大而整体下降。

### Evidence 4

Leave-one-feature-out 的：

\[
V_j(h)
\]

同样存在明显 horizon dependency，并与 correlation / partial correlation 结果至少部分一致。

如果这些现象没有出现，也必须如实报告：

```text
当前 CS2-35/36/37/38 数据没有提供足够证据支持 horizon-dependent feature relevance。
```

不要为了得到预期结论调整数据或筛选结果。

---

# 十八、科研上必须避免的问题

请主动检查以下问题。

### 1. 数据泄漏

任何预测模型的 scaler、normalizer、特征处理参数只能从训练电池得到。

### 2. 时间泄漏

构造：

\[
SOH_{t+h}
\]

时，feature 只能来自：

\[
t
\]

及其之前。

不能使用 \(t+h\) 前的未来信息。

### 3. SOH 自相关造成伪相关

因此必须同时报告：

- ordinary correlation；
- future degradation correlation；
- partial correlation controlling \(SOH_t\)。

### 4. 共同时间趋势造成伪相关

如果发现所有 feature 与 future SOH 都接近 ±1，需要进一步进行敏感性分析。

可增加：

\[
\Delta HF_t=HF_t-HF_{t-k}
\]

或局部 slope：

\[
Slope(HF_{t-L+1:t})
\]

与 future degradation 的关系。

但这些属于扩展分析，不要在主分析结果出来之前替换原始 HF。

### 5. 不要只通过 attention 或 SHAP 证明预测价值

当前实验不使用深度模型，也不把 attention weight 当作 feature importance。

---

# 十九、代码工程要求

请生成结构清晰、可重复运行的 Python 项目。

建议：

```text
project/
│
├── data/
│
├── processed/
│
├── results/
│
├── figures/
│
├── src/
│   ├── load_data.py
│   ├── extract_soh.py
│   ├── extract_features.py
│   ├── correlation_analysis.py
│   ├── partial_correlation.py
│   ├── bootstrap.py
│   ├── ranking_analysis.py
│   ├── prediction_probe.py
│   └── plotting.py
│
├── run_analysis.py
├── requirements.txt
└── README.md
```

所有核心参数集中放在配置区，例如：

```python
BATTERIES = ["CS2-35", "CS2-36", "CS2-37", "CS2-38"]

HORIZONS = [1, 4, 8, 12, 16, 24, 32]

BOOTSTRAP_N = 1000
BLOCK_LENGTH = 20
RANDOM_SEED = 42
```

不要把参数散落写死在多个脚本里。

---

# 二十、运行顺序

请按照以下顺序实际执行，而不是只给我代码：

### Step 1
检查原始文件和数据结构。

### Step 2
正确解析四块电池的 cycle。

### Step 3
计算 SOH。

### Step 4
提取 HF1-HF8。

### Step 5
进行数据质量检查。

### Step 6
执行 ordinary correlation。

### Step 7
执行 future degradation correlation。

### Step 8
执行 partial correlation。

### Step 9
分析 ranking crossover 和 Kendall-\(\tau\)。

### Step 10
运行 block bootstrap。

### Step 11
运行 Ridge prediction probe 和 leave-one-feature-out。

### Step 12
生成所有图表和 CSV。

### Step 13
根据实际结果总结是否支持 horizon-dependent feature relevance。

---

# 二十一、最终需要给我的结论

完成实验后，请不要只告诉我“代码运行成功”。

请输出一份科研式总结，至少回答以下问题：

1. 不同 HF 的 predictive relevance 是否确实随 \(h\) 改变？
2. 哪些 HF 在 short horizon 更重要？
3. 哪些 HF 在 medium/long horizon 相对更重要？
4. 是否出现明确的 feature ranking crossover？
5. 哪些 crossover 可以在 3/4 或 4/4 电池重复？
6. SOH_t 的预测价值随 horizon 如何变化？
7. HF 在控制 SOH_t 后是否仍具有额外信息？
8. SOH+HF 相对于 SOH-only 的增益是否随 horizon 增大？
9. correlation evidence 与 prediction-probe evidence 是否一致？
10. 当前证据是否足以支持下一阶段设计 horizon-aware feature weighting 模型？

最后给出明确判断：

```text
结论 A：数据明显支持 horizon-dependent feature relevance；
```

或：

```text
结论 B：存在部分迹象，但跨电池一致性不足；
```

或：

```text
结论 C：当前数据不能支持该假设。
```

不能根据我的预期人为选择结论。

---

# 二十二、工作方式要求

请直接读取数据、编写代码、运行代码、检查报错、修复代码并继续实验。

不要停留在“建议我怎么做”。

如果遇到字段名、文件结构、cycle 标记与预期不同，请优先根据实际文件自动分析并解决。

如果某个 HF 无法严格按照上述定义提取：

1. 明确说明原因；
2. 展示数据证据；
3. 提出最接近原定义的可重复替代方案；
4. 不要未经说明自行改变特征定义。

所有分析必须可复现。

请从“检查数据目录和识别 CS2-35、36、37、38 文件结构”开始执行。