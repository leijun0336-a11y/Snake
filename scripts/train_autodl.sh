#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
SYSTEM_PYTHON="${SNAKE_SYSTEM_PYTHON:-}"

has_cuda_torch() {
    local python_bin="$1"
    [[ -x "$python_bin" ]] || return 1
    "$python_bin" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1
}

has_local_torch() {
    local python_bin="$1"
    [[ -x "$python_bin" ]] || return 1
    "$python_bin" -c 'import pathlib, sys, torch; prefix = pathlib.Path(sys.prefix).resolve(); torch_path = pathlib.Path(torch.__file__).resolve(); sys.exit(0 if torch_path.is_relative_to(prefix) else 1)' >/dev/null 2>&1
}

if [[ -z "$SYSTEM_PYTHON" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        SYSTEM_PYTHON="$(command -v python3)"
    else
        SYSTEM_PYTHON="$(command -v python)"
    fi
fi

SYSTEM_HAS_CUDA_TORCH=false
if has_cuda_torch "$SYSTEM_PYTHON"; then
    SYSTEM_HAS_CUDA_TORCH=true
fi

# 新环境只复用系统中可以实际访问 GPU 的 PyTorch。
if [[ ! -x "$VENV_PYTHON" ]]; then
    if [[ "$SYSTEM_HAS_CUDA_TORCH" == true ]]; then
        echo "Detected system CUDA PyTorch; creating .venv with system site packages."
        uv venv --python "$SYSTEM_PYTHON" --system-site-packages .venv
    fi
fi

uv sync --no-dev --extra train --inexact

# 让已有环境复用系统 GPU PyTorch。若本地 CPU torch 遮蔽了它，只移除该本地包。
if ! has_cuda_torch "$VENV_PYTHON" && [[ "$SYSTEM_HAS_CUDA_TORCH" == true ]]; then
    echo "Detected system CUDA PyTorch; enabling system site packages for the existing .venv."
    uv venv --python "$SYSTEM_PYTHON" --allow-existing --system-site-packages .venv

    if ! has_cuda_torch "$VENV_PYTHON" && has_local_torch "$VENV_PYTHON"; then
        echo "Removing a non-CUDA torch from .venv because it shadows system CUDA PyTorch."
        uv pip uninstall --python "$VENV_PYTHON" torch
    fi
fi

# CPU PyTorch 不影响训练判定；找不到可用的 GPU PyTorch 时安装 cu124。
if ! has_cuda_torch "$VENV_PYTHON"; then
    echo "CUDA PyTorch was not found; installing the cu124 build."
    uv sync --no-dev --extra cu124 --extra train --inexact
fi

"$VENV_PYTHON" -c 'import sys, torch; print(f"Using torch={torch.__version__}, cuda={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else "GPU PyTorch is required for training")'

"$VENV_PYTHON" -m snake_ai.train "$@"
