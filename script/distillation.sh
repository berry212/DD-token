#!/usr/bin/env bash
# set -euo pipefail

# uv run distill-pathmnist-tokens --dataset pathmnist \
#     --data-root ./data \
#     --output-dir ./artifacts/pathmnist_tokens \
#     --epochs 5 \
#     --batch-size 256 \
#     --patch-size 4 \
#     --overlap 0.2 \
#     --codebook-size 2048 \
#     --code-dim 256 \
#     --hidden-dim 384 \
#     --cls-weight 0.5 \
#     --cls-weight-end 0.2 \
#     --diversity-weight 0.05 \
#     --quant-temperature 1.0 \
#     --warmup-ratio 0.05 \
#     --min-lr 1e-5 \
#     --image-size 28

# Skin lesions (Hugging Face dataset) example:
uv run distill-pathmnist-tokens --dataset skin-lesions \
    --data-root ./data \
    --output-dir ./artifacts/skin_lesions_tokens \
    --epochs 5 \
    --batch-size 64 \
    --patch-size 16 \
    --overlap 0.2 \
    --codebook-size 2048 \
    --code-dim 256 \
    --hidden-dim 384 \
    --cls-weight 0.4 \
    --cls-weight-end 0.2 \
    --diversity-weight 0.05 \
    --quant-temperature 1.0 \
    --warmup-ratio 0.05 \
    --min-lr 1e-5 \
    --image-size 256 \
    --num-workers 0
