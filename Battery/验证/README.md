# CALCE CS2 步长感知特征价值验证

本项目从 CALCE CS2-35/36/37/38 原始 Arbin Excel 文件提取 SOH 与 HF1–HF8，并执行普通相关、未来退化相关、控制当前 SOH 的偏相关、排名交叉、Kendall 相似度、循环移动块 Bootstrap，以及留一电池 Ridge prediction probe。

## 运行

```powershell
& 'E:\CodeDir\Battery\.venv\Scripts\python.exe' run_analysis.py
```

环境由父项目的 `uv` 管理；重复分析时可用 `--skip-extraction` 复用已提取 CSV。所有参数集中在 `config.py`。

## Validation v2

严格支持集、SOH-history/trend、固定 alpha 敏感性、grouped LOFO、HF4–HF6 缺失机制、分段非循环 Bootstrap 和 ranking null test：

```powershell
& 'E:\CodeDir\Battery\.venv\Scripts\python.exe' run_analysis_v2.py
```

v2 只读取 `processed/`，结果写入 `results_v2/`、`figures_v2/` 和 `cache_v2/`，不会覆盖 v1。科研结论见 `VALIDATION_V2_SUMMARY.md`。

## 固定方法

- SOH 主定义：cycle 放电容量 / 1.1 Ah 额定容量；同时保存以首个有效实测容量归一化的 `SOH_initial` 供敏感性分析。
- 工步不依赖固定 `Step_Index`：CCCS 由正电流且明显电压跨度识别；CVCS 由正电流、4.15 V 以上且电压跨度不超过 35 mV 识别；CCDS 由中位电流小于 -0.2 A 识别。
- HF1/HF2/放电能量：按真实 `Date_Time` 梯形积分，单位 Wh。
- HF4：放电 Q–V 曲线在线性插值得到 4.0 V 与 3.4 V 容量差。
- HF5/HF6：3.0–4.0 V、5 mV 等距网格；Q 插值后统一使用 21 点三阶 Savitzky–Golay；取正的 `-dQ/dV` 全局主峰并排除半窗口边界。
- 缺失与极端值不从原始特征表删除；统计分析使用成对完整样本，预测器的中位数填补器只在训练电池拟合。
- Bootstrap：循环移动块，1000 次，块长 10/20/30。
- Prediction probe：外层严格 leave-one-battery-out，填补、标准化和 Ridge 均只在训练电池拟合。
