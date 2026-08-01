#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
SYSTEM_PYTHON="${SNAKE_SYSTEM_PYTHON:-}"

if [[ -z "$SYSTEM_PYTHON" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        SYSTEM_PYTHON="$(command -v python3)"
    else
        SYSTEM_PYTHON="$(command -v python)"
    fi
fi

# 新环境优先继承系统包，以便直接复用服务器已经安装的 CUDA PyTorch。
if [[ ! -x "$VENV_PYTHON" ]]; then
    if "$SYSTEM_PYTHON" -c 'import torch' >/dev/null 2>&1; then
        echo "Detected system PyTorch; creating .venv with system site packages."
        uv venv --python "$SYSTEM_PYTHON" --system-site-packages .venv
    fi
fi

uv sync --no-dev --extra train --inexact

# 兼容已经存在、但尚未启用 system-site-packages 的 .venv。
if ! "$VENV_PYTHON" -c 'import torch' >/dev/null 2>&1 \
    && "$SYSTEM_PYTHON" -c 'import torch' >/dev/null 2>&1; then
    echo "Detected system PyTorch; enabling system site packages for the existing .venv."
    uv venv --python "$SYSTEM_PYTHON" --allow-existing --system-site-packages .venv
fi

# 训练必须使用 CUDA；系统中只有 CPU PyTorch 时仍需安装 cu124。
if ! "$VENV_PYTHON" -c \
    'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' \
    >/dev/null 2>&1; then
    echo "CUDA PyTorch was not found; installing the cu124 build."
    uv sync --no-dev --extra cu124 --extra train --inexact
fi

"$VENV_PYTHON" -c \
    'import sys, torch; print(f"Using torch={torch.__version__}, cuda={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else "GPU PyTorch is required for training")'

"$VENV_PYTHON" -m snake_ai.train "$@"
