#!/usr/bin/env bash

# 同步当前默认依赖、检查当前 PyTorch 环境的 CUDA 可用性，然后启动训练。

# 如果某一行命令失败，脚本立刻退出。
set -euo pipefail

# 消除 CuBLAS 确定性算法的警告
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 这里没有显式选择 PyTorch extra；若当前环境不含 CUDA 版 PyTorch，下面的检查会终止脚本。
uv sync

# 不允许静默退回 CPU 训练。
uv run python -c 'import sys, torch; print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else "GPU PyTorch is required for training")'

# "$@" 表示把传给脚本的所有参数原样转发给训练入口。
uv run python -m snake_ai.train "$@"
