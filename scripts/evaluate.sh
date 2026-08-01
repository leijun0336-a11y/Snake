#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_EVALUATE="$PROJECT_ROOT/.venv/bin/snake-evaluate"
SYSTEM_PYTHON="${SNAKE_SYSTEM_PYTHON:-}"

if [[ -z "$SYSTEM_PYTHON" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        SYSTEM_PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        SYSTEM_PYTHON="$(command -v python)"
    else
        echo "Python was not found."
        exit 1
    fi
fi

# Let a new environment reuse an existing CPU or CUDA PyTorch installation.
if [[ ! -x "$VENV_PYTHON" ]] \
    && "$SYSTEM_PYTHON" -c 'import torch' >/dev/null 2>&1; then
    echo "Detected system PyTorch; creating .venv with system site packages."
    uv venv --python "$SYSTEM_PYTHON" --system-site-packages .venv
fi

SYNC_ARGS=(sync --no-dev --inexact)
for argument in "$@"; do
    if [[ "$argument" == "--tensorboard" ]]; then
        SYNC_ARGS+=(--extra train)
        break
    fi
done
uv "${SYNC_ARGS[@]}"

# Also handle an existing .venv that cannot see system site packages yet.
if ! "$VENV_PYTHON" -c 'import torch' >/dev/null 2>&1 \
    && "$SYSTEM_PYTHON" -c 'import torch' >/dev/null 2>&1; then
    echo "Detected system PyTorch; enabling system site packages for the existing .venv."
    uv venv --python "$SYSTEM_PYTHON" --allow-existing --system-site-packages .venv
fi

if ! "$VENV_PYTHON" -c 'import torch' >/dev/null 2>&1; then
    echo "PyTorch was not found; installing the CPU build."
    uv "${SYNC_ARGS[@]}" --extra cpu
else
    "$VENV_PYTHON" -c \
        'import torch; print(f"Using torch={torch.__version__}, cuda={torch.cuda.is_available()}")'
fi

"$VENV_EVALUATE" "$@"
