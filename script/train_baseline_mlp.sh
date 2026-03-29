#!/usr/bin/env sh
set -eu

DATASET="${1:-pathmnist}"

case "$DATASET" in
    pathmnist)
        uv run train-baseline-mlp \
            --dataset pathmnist \
            --data-root ./data \
            --output-dir ./artifacts/pathmnist_mlp_baseline \
            --image-size 64 \
            --hidden-dim 512 \
            --epochs 10 \
            --batch-size 256 \
            --lr 3e-4 \
            --num-workers 6
        ;;
    skin-lesions|skin_lesions|ahmed-ai/skin-lesions-classification-dataset)
        uv run train-baseline-mlp \
            --dataset skin-lesions \
            --data-root ./data \
            --output-dir ./artifacts/skin_lesions_mlp_baseline \
            --image-size 64 \
            --hidden-dim 512 \
            --epochs 10 \
            --batch-size 128 \
            --lr 3e-4 \
            --num-workers 6
        ;;
    *)
        echo "Usage: sh script/train_baseline_mlp.sh [pathmnist|skin-lesions]" >&2
        exit 1
        ;;
esac
