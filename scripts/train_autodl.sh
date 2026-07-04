#!/usr/bin/env bash
set -euo pipefail

uv sync
uv run python -m snake_ai.train "$@"
