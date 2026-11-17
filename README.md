# HAR-Residual-LSTM

本项目基于已有研究开展模型实现与扩展，用于个人研究与方法展示。代码将持续迭代，部分设计与实验设置可能与原研究有所不同。

使用 HAR 基线与 LSTM 残差修正进行波动率预测。
关联论文Cross-Asset Volatility Forecasting with HAR-Guided LSTM Residual Learning

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
| `Dockerfile` | CPU 实验镜像的构建入口 |
| `.github/workflows/docker.yml` | 构建、验证并发布 Docker 镜像 |

## Docker 部署

容器以**批处理**方式运行：启动后读取 CSV、训练模型并写出评估结果，任务完成后自动退出。镜像内含 Python 3.11、CPU 版 PyTorch 和仓库自带数据，不需要在宿主机安装 Python，也不需要 GPU。

### 1. 准备 Docker

Windows 和 macOS 启动 Docker Desktop，Windows 需使用 Linux 容器。Linux 启动 Docker Engine。运行 `docker version`，确认同时看到 Client 和 Server 信息。下面使用命名数据卷的单行命令可以在 Bash 和 PowerShell 中执行。

### 2. 获取镜像

任选一种方式，两种方式最后都使用本地镜像名 `har-residual-lstm:latest`。

**从源码构建。** 克隆仓库并切到含有 `Dockerfile` 的版本，在仓库根目录执行：

```bash
git clone https://github.com/Lichao-Lin/HAR-Residual-LSTM.git
cd HAR-Residual-LSTM
docker build -t har-residual-lstm:latest .
```

默认分支包含 Docker 配置时，可直接按上面执行。如果要试用尚未合并的部署 PR，请先切到它的分支再执行 `docker build`。如果已经克隆，可以在现有仓库中切换分支并构建。构建时会下载依赖，并按 `MANIFEST.sha256` 校验镜像内的 `prices.csv` 和 `returns.csv`。

**从 GitHub Container Registry 拉取。** `main` 上的 Docker 工作流构建并验证通过后，将镜像发布到 GHCR，并添加 `latest` 和完整提交 SHA 两个标签。可用的预构建镜像由 Linux x86-64 CI 生成。

```bash
docker pull ghcr.io/lichao-lin/har-residual-lstm:latest
docker tag ghcr.io/lichao-lin/har-residual-lstm:latest har-residual-lstm:latest
```

复现实验时可把两条命令中的远程 `latest` 换成对应的完整提交 SHA 标签。若尚无发布记录，使用源码构建；若提示无权拉取，请确认自己有软件包的读取权限，并登录 GHCR。

### 3. 短训练和完整实验

先跑一次短训练，检查镜像、输入和输出是否可用。短训练只用于验证流程：

```bash
docker run --rm -v har-lstm-quick:/app/outputs har-residual-lstm:latest python scripts/har_lstm_cli.py --device cpu --quick
```

确认正常后，选择完整实验：

```bash
# 60 日窗口实验：使用镜像默认命令
docker run --rm -v har-lstm-60d:/app/outputs har-residual-lstm:latest

# 96 日窗口的扩展实验：使用五个随机种子，运行时间较长
docker run --rm -v har-lstm-96d:/app/outputs har-residual-lstm:latest python scripts/run_outline_complete.py
```

命令默认在前台显示日志。`--rm` 会在退出时移除容器，命名数据卷会保留结果。两套完整实验会写固定文件名；重复运行时可更换卷名，为每次运行单独留档。

### 4. 查看和导出结果

下表路径相对于容器的 `/app/outputs`，也是对应数据卷里的目录结构。

| 运行方式 | 主要输出 |
| --- | --- |
| 短训练 | `cli/quick_*/` 下的 `metrics.csv`、`predictions.csv`、`split_summary.csv`、`metadata.json` |
| 60 日窗口 | `artifacts/main_performance.csv`、`artifacts/test_predictions.csv`、`artifacts/split_summary.csv` |
| 96 日窗口 | `artifacts_complete/all_model_summary.csv`、`artifacts_complete/ensemble_predictions.csv`、`artifacts_complete/split_summary.csv` |

例如，将 60 日实验的数据卷复制到当前目录下的 `har-results-60d`。建议选择一个尚不存在的目标目录：

```bash
docker create --name har-lstm-export -v har-lstm-60d:/app/outputs har-residual-lstm:latest
docker cp har-lstm-export:/app/outputs ./har-results-60d
docker rm har-lstm-export
```

导出其他实验时，把 `har-lstm-60d` 和目标目录换成对应的名字。临时容器无需启动。

在 Linux 上，若希望结果直接出现在宿主机目录，可用绑定挂载，并以当前用户的 UID 和 GID 运行以保证可写：

```bash
mkdir -p outputs
docker run --rm --user "$(id -u):$(id -g)" --mount "type=bind,source=$PWD/outputs,target=/app/outputs" har-residual-lstm:latest
```

### 5. 使用自己的 CSV

准备包含 `prices.csv` 和 `returns.csv` 的宿主机目录，保留与仓库示例相同的列和日期格式。将下例的 `/absolute/path/to/input` 换成实际绝对路径（Windows 示例为 `C:/data/har-input`）：

```bash
docker run --rm --mount "type=bind,source=/absolute/path/to/input,target=/app/data,readonly" -v har-lstm-custom:/app/outputs har-residual-lstm:latest python scripts/run_outline_complete.py
```

挂载会替换本次运行看到的 `/app/data`。镜像构建时的哈希校验覆盖仓库自带的 CSV，外部挂载的数据需要自行核对。

### 常见问题

| 现象 | 处理方法 |
| --- | --- |
| `docker version` 只有 Client，或提示无法连接守护进程 | 启动 Docker Desktop 或 Docker Engine 后重试。 |
| 找不到 `Dockerfile` | 确认当前目录是仓库根目录，且已切到包含 Docker 配置的版本。 |
| 拉取镜像时提示 `manifest unknown` 或 `denied` | 在 GitHub Actions 确认对应标签已发布，并检查 GHCR 登录状态与软件包权限；也可从源码构建。 |
| 训练结束后容器从 `docker ps` 消失 | 批处理完成后容器退出，`--rm` 会移除它。请从命名数据卷导出结果。 |
| 写入 `/app/outputs` 时提示权限不足 | 使用示例的命名数据卷；在 Linux 上绑定宿主目录时，先创建该目录，并使用上面的 `--user` 参数。 |

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

其中 $\mathcal S$ 为测试集。训练、验证、测试之间的四日间隔规则见上文“样本划分”。
