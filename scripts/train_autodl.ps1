$ErrorActionPreference = "Stop"

$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

uv sync
uv run python -m snake_ai.train @args
