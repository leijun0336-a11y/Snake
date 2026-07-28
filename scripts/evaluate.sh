#!/usr/bin/env bash

# 如果某一行命令失败，脚本立刻退出。
set -euo pipefail

# 消除 CuBLAS 确定性算法的警告
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# "$@" 表示把传给脚本的所有参数原样转发给评估入口。
uv run python -m snake_ai.evaluate "$@"
