#!/usr/bin/env bash

# 如果某一行命令失败，脚本立刻退出。
set -euo pipefail

# "$@"表示把你传给脚本的所有参数原样转发给训练程序。
uv run --extra cpu python -m snake_ai.evaluate "$@"
