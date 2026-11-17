# 用 Docker 运行 HAR-Residual-LSTM

镜像内含仓库的两份 CSV，构建时会按 `MANIFEST.sha256` 校验。镜像使用 Python 3.11 和 CPU 版 PyTorch，不需要 GPU。默认执行 60 日窗口的训练与评估，任务结束后容器退出。

## 获取镜像

在仓库根目录构建：

```bash
docker build -t har-residual-lstm .
```

GitHub Actions 在 `main` 上通过验证后发布镜像，并为每次发布添加提交 SHA 标签和 `latest` 标签。若软件包需要访问权限，请先登录 GHCR，再拉取镜像：

```bash
docker pull ghcr.io/lichao-lin/har-residual-lstm:latest
docker tag ghcr.io/lichao-lin/har-residual-lstm:latest har-residual-lstm:latest
```

## 运行并保存结果

使用 Docker 命名数据卷保留结果。先跑一次短训练，检查模型和数据能否正常工作：

```bash
docker run --rm -v har-lstm-results:/app/outputs har-residual-lstm python scripts/har_lstm_cli.py --device cpu --quick
```

运行完整的论文实验：

```bash
# 60 日窗口实验
docker run --rm -v har-lstm-results:/app/outputs har-residual-lstm

# 96 日窗口扩展实验，五个随机种子
docker run --rm -v har-lstm-results:/app/outputs har-residual-lstm python scripts/run_outline_complete.py
```

将结果从数据卷复制到当前目录的 `outputs/`：

```bash
docker create --name har-lstm-export -v har-lstm-results:/app/outputs har-residual-lstm
docker cp har-lstm-export:/app/outputs ./
docker rm har-lstm-export
```

在 Linux 上也可以直接挂载宿主机目录。让容器使用当前用户的 UID 和 GID，就能写入该目录：

```bash
mkdir -p outputs
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/outputs:/app/outputs" har-residual-lstm
```

如需使用其他数据，把含 `returns.csv` 和 `prices.csv` 的目录只读挂载到 `/app/data`。两个完整实验使用固定文件名，重复运行前可以为每次实验指定不同的数据卷来保留结果。
