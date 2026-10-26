# HAR-Residual-LSTM

论文复现，用于个人研究。

使用 HAR 基线与 LSTM 残差修正进行波动率预测。

## 目录

| 路径 | 内容 |
| --- | --- |
| `scripts/` | 模型、实验与绘图代码 |
| `data/` | 输入价格与收益率 CSV |
| `requirements.txt` | 原实验环境的依赖版本 |
| `MANIFEST.sha256` | 输入数据的 SHA-256 校验值 |

## 环境

Python 3.11。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 运行

在仓库根目录执行：

```bash
# 60 日窗口实验
python scripts/run_experiments.py
python scripts/make_supporting_outputs.py

# 96 日窗口扩展实验（五个随机种子，运行时间较长）
python scripts/run_outline_complete.py
python scripts/make_complete_figures.py

# 扩展实验完成后绘制预测对比曲线
python scripts/plot_three_model_trends.py

# 可选：计算年度特征重要性
python scripts/plot_feature_importance_by_year.py
python scripts/plot_engineered_importance_by_year.py
```

路径由脚本位置自动确定。运行生成的结果、预测值与图件写入 `outputs/`，该目录已加入 `.gitignore`。

## 数据说明

- `prices.csv` 与 `returns.csv` 按日期记录市场序列；预测目标为 IXIC 未来五日方差的对数。
- 数据保留 VIX 列，实验明确排除 VIX。
- 两套实验的窗口长度和有效样本不同，应分别运行和比较。
- 年度 TreeSHAP 使用全样本拟合的辅助 LightGBM 模型，属于回顾性解释。
