# HAR-Residual-LSTM

本项目基于已有研究开展模型实现与扩展，用于个人研究与方法展示。代码将持续迭代，部分设计与实验设置可能与原研究有所不同。

使用 HAR 基线与 LSTM 残差修正进行波动率预测。

## 目录

| 路径 | 内容 |
| --- | --- |
| `scripts/run_experiments.py` | 60 日窗口的模型训练与评估 |
| `scripts/run_outline_complete.py` | 96 日窗口的扩展实验、基线比较与消融 |
| `scripts/temporal_split.py` | 两套实验共用的时间划分 |
| `tests/` | 四日间隔与目标窗口不重叠的检查 |
| `data/` | 输入价格与收益率 CSV |
| `requirements.txt` | 原实验环境的依赖版本 |
| `MANIFEST.sha256` | 输入数据的 SHA-256 校验值 |

## 环境

原实验环境为 Python 3.11.14、PyTorch 2.13.0+cpu；完整依赖版本保留在 `requirements.txt`。其中 Matplotlib 和 Seaborn 保留为历史环境记录，当前代码不再绘图。

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

# 96 日窗口扩展实验（五个随机种子，运行时间较长）
python scripts/run_outline_complete.py
```

路径由脚本位置自动确定，无需设置个人电脑目录，也可以从其他工作目录运行脚本。训练和评估生成的结果、预测值写入 `outputs/`，该目录已加入 `.gitignore`。

## 样本划分

按日期排序，以有效样本数的 70% 和 85% 为分界。训练集结束后排除 4 个样本，再开始验证集；验证集结束后排除 4 个样本，再开始测试集。这些样本不作为训练、验证或测试的预测起点。

间隔按输入数据中连续的交易日观测计数，周末和休市日不补齐。预测目标仍为未来 5 日；历史输入窗口可以重叠，分界两侧的未来目标窗口不重叠。例如 100 个样本（索引从 0 开始）：

```text
训练 0–69 | 排除 70–73 | 验证 74–84 | 排除 85–88 | 测试 89–99
```

两套实验都使用此规则，运行时各自保存 `split_summary.csv` 和间隔元数据。

## 检查

```bash
python -m unittest discover -s tests -v
```

## 数据说明

- `prices.csv` 与 `returns.csv` 按日期记录市场序列；预测目标为 IXIC 未来五日方差的对数。
- 数据保留 VIX 列，实验明确排除 VIX。
- 两套实验的窗口长度和有效样本不同，应分别运行和比较。
