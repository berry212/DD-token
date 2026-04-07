#!/usr/bin/env bash
# set -euo pipefail

IPC=100


# DermaMNIST official 224 split distillation (MedMNIST+ native size)
# uv run distill-tokens --dataset dermamnist \
#     --data-root ./data \
#     --output-dir ./artifacts/dermamnist_tokens \
#     --epochs 5 \
#     --batch-size 16 \
#     --rvq-stages 1 \
#     --cls-weight 0.8 \
#     --cls-weight-end 0.5 \
#     --diversity-weight 0.05 \
#     --quant-temperature 1.0 \
#     --warmup-ratio 0.05 \
#     --min-lr 1e-5 \
#     --image-size 224 \
#     --num-workers 0 \
#     --ipc "$IPC"

# PathMNIST example:
uv run distill-tokens --dataset pathmnist \
    --data-root ./data \
    --output-dir ./artifacts/pathmnist_tokens \
    --epochs 5 \
    --batch-size 256 \
    --rvq-stages 1 \
    --cls-weight 0.8 \
    --cls-weight-end 0.5 \
    --diversity-weight 0.05 \
    --quant-temperature 1.0 \
    --warmup-ratio 0.05 \
    --min-lr 1e-5 \
    --image-size 28 \
    --ipc "$IPC"
