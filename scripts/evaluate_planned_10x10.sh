#!/usr/bin/env bash

set -euo pipefail

uv run --extra cpu python -m snake_ai.evaluate_planned_10x10 "$@"
