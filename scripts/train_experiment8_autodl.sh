#!/usr/bin/env bash

# Strict configuration reproduction for dqn_20260712_130642.
# This script intentionally accepts no overrides: changing one value would no longer
# be a strict reproduction of experiment 8.
set -euo pipefail

if (( $# != 0 )); then
    echo "train_experiment8_autodl.sh accepts no arguments" >&2
    exit 2
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8

uv sync

uv run python -m snake_ai.train \
    --reward-profile experiment8 \
    --max-episodes 15000 \
    --width 6 \
    --height 6 \
    --state-mode hybrid \
    --batch-size 128 \
    --gamma 0.99 \
    --learning-rate 0.001 \
    --replay-buffer-size 100000 \
    --epsilon-start 1.0 \
    --epsilon-end 0.01 \
    --epsilon-decay 0.995 \
    --epsilon-decay-episodes 7500 \
    --target-update-interval 1000 \
    --hidden-size 256 \
    --cnn-channels 32 \
    --cnn-output-channels 8 \
    --cnn-dilations 1 1 2 \
    --cnn-pool-size 10 10 \
    --seed 42
