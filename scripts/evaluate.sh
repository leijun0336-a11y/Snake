#!/usr/bin/env bash
set -euo pipefail

uv run python -m snake_ai.evaluate "$@"
