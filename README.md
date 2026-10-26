# HAR-Residual-LSTM

本项目基于已有研究开展模型实现与扩展，用于个人研究与方法展示。代码将持续迭代，部分设计与实验设置可能与原研究有所不同。

使用 HAR 基线与 LSTM 残差修正进行波动率预测。

## 模型公式

以下公式对应当前代码。记 $t$ 为交易日观测索引，$r_t$ 为 IXIC 收益率，预测跨度 $H=5$，数值下限 $\varepsilon=10^{-12}$。

### 1. 预测目标与 HAR 特征

未来五日方差代理及其对数为：

$$
V_t^{+}=\sum_{k=1}^{H}r_{t+k}^{2},
\qquad y_t=\log\!\left(\max(V_t^{+},\varepsilon)\right).
$$

该目标由日收益率平方和构造。预测时可用的历史状态为：

$$
v_t=\log\!\left(\max\!\left(\sum_{k=0}^{H-1}r_{t-k}^{2},\varepsilon\right)\right),
\qquad
\mathbf z_t=\begin{bmatrix}
v_t \\
\frac{1}{5}\sum_{j=0}^{4}v_{t-j} \\
\frac{1}{22}\sum_{j=0}^{21}v_{t-j}
\end{bmatrix}.
$$

代码中的 `har_d`、`har_w`、`har_m` 分别对应这三个分量：当前五日对数方差，以及它的 5 日、22 日移动平均。这里先取对数，再计算移动平均。

### 2. HAR 基线与残差

在训练集 $\mathcal T$ 上以普通最小二乘拟合 HAR：

$$
(\hat\beta_0,\hat{\boldsymbol\beta})
=\underset{\beta_0,\boldsymbol\beta}{\arg\min}
\sum_{t\in\mathcal T}\left(y_t-\beta_0-\boldsymbol\beta^{\top}\mathbf z_t\right)^2,
\qquad
\hat y_t^{\mathrm{HAR}}=\hat\beta_0+\hat{\boldsymbol\beta}^{\top}\mathbf z_t.
$$

固定 HAR 参数后，LSTM 的学习目标为残差：

$$
e_t=y_t-\hat y_t^{\mathrm{HAR}}.
$$

训练残差来自训练集上拟合的 HAR，验证和测试残差使用同一组 HAR 参数；代码没有对训练残差做交叉拟合。

### 3. LSTM 残差修正

令 $\mathbf x_t$ 为多资产输入，$L$ 为序列长度。用波浪号表示按训练集统计量标准化后的变量，则网络可概括为：

$$
\mathbf h_t=\operatorname{LSTM}_{\theta}
\!\left(\tilde{\mathbf x}_{t-L+1},\ldots,\tilde{\mathbf x}_{t}\right)_{\mathrm{last}},
\qquad
\hat{\tilde e}_t=g_{\theta}\!\left(\operatorname{LayerNorm}(\mathbf h_t),\mathbf u_t\right),
\qquad
\hat e_t=\mu_e+s_e\hat{\tilde e}_t.
$$

$\mu_e$、$s_e$ 是训练残差的均值和用于标准化的尺度。$\mathbf u_t$ 表示可选的标准化 HAR 输入。网络在训练集上学习标准化残差，验证集用于早停。两套脚本的默认残差模型有以下区别：

| 设置 | `run_experiments.py` | `run_outline_complete.py` |
| --- | --- | --- |
| 序列长度 $L$ | 60 | 96 |
| 每日输入 $\mathbf x_t$ | 多资产收益率及其绝对值 | 多资产收益率 |
| 输出层额外输入 $\mathbf u_t$ | 标准化 HAR 特征 | 无 |
| 输出头 $g_\theta$ | 含 GELU、Dropout 的多层感知机 | 线性层 |
| 训练损失 | Smooth L1，作用于标准化残差 | 均方误差，作用于标准化残差 |

输入和目标的标准化参数均只从训练集估计；VIX 不进入输入。

### 4. 验证集调权与最终预测

在验证集 $\mathcal V$ 上选择残差修正权重：

$$
\alpha^{\ast}=\underset{\alpha\in\mathcal A}{\arg\min}
\frac{1}{|\mathcal V|}\sum_{t\in\mathcal V}
\left(y_t-\hat y_t^{\mathrm{HAR}}-\alpha\hat e_t\right)^2,
\qquad
\mathcal A=\{0,0.01,\ldots,1.25\}.
$$

最终对数方差预测为：

$$
\hat y_t=\hat y_t^{\mathrm{HAR}}+\alpha^{\ast}\hat e_t.
$$

权重选定后固定用于测试集。扩展实验还对五个随机种子各自调权后的对数预测取平均，用于集成预测与配对比较。

### 5. 评估尺度

MSE、MAE 和 $R^2$ 在对数方差尺度计算。代码在计算 QLIKE 时将预测转换回方差尺度，并做数值截断：

$$
\hat V_t=\exp\!\left(\operatorname{clip}(\hat y_t,-30,10)\right),
\qquad
\operatorname{QLIKE}=\frac{1}{|\mathcal S|}\sum_{t\in\mathcal S}
\left(\frac{\exp(y_t)}{\hat V_t}+\log\hat V_t\right),
$$

其中 $\mathcal S$ 为测试集。训练、验证、测试之间的四日间隔规则见下文“样本划分”。

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
