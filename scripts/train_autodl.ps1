$ErrorActionPreference = "Stop"

$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

# 游戏使用 cpu extra；训练必须显式选择 CUDA 12.4 的 cu124 extra。
uv sync --extra cu124

# 不允许静默退回 CPU 训练。
uv run --extra cu124 python -c "import sys, torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}'); sys.exit(0 if torch.cuda.is_available() else 'GPU PyTorch is required for training')"
uv run --extra cu124 python -m snake_ai.train @args
