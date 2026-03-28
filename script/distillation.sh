#!/usr/bin/env bash
# set -euo pipefail

uv run distill-pathmnist-tokens --dataset pathmnist \
    --data-root ./data \
    --output-dir ./artifacts/pathmnist_tokens \
    --epochs 16 \
    --batch-size 256 \
    --patch-size 4 \
    --overlap 0.2 \
    --codebook-size 2048 \
    --code-dim 256 \
    --hidden-dim 384 \
    --cls-weight 0.5 \
    --cls-weight-end 0.2 \
    --diversity-weight 0.05 \
    --quant-temperature 1.0 \
    --warmup-ratio 0.05 \
    --min-lr 1e-5 \
    --image-size 28
