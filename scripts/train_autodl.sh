#!/usr/bin/env bash

# 这个脚本 = 同步依赖 + 启动训练 + 把命令行参数传给 train.py

# 如果某一行命令失败，脚本立刻退出。
set -euo pipefail

# 消除 CuBLAS 确定性算法的警告
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 游戏使用 cpu extra；训练必须显式选择 CUDA 12.4 的 cu124 extra。
uv sync

# 不允许静默退回 CPU 训练。
uv run python -c 'import sys, torch; print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else "GPU PyTorch is required for training")'

# "$@"表示把你传给脚本的所有参数原样转发给训练程序。
uv run python -m snake_ai.train "$@"
