#!/usr/bin/env sh
set -eu

DATASET="${1:-pathmnist}"

case "$DATASET" in
    pathmnist)
        uv run train-baseline-vit \
            --dataset pathmnist \
            --data-root ./data \
            --output-dir ./artifacts/pathmnist_vit_baseline \
            --epochs 10 \
            --batch-size 256 \
            --lr 3e-4 \
            --num-workers 6 \
            --precision bf16-mixed
        ;;
    skin-lesions|skin_lesions|ahmed-ai/skin-lesions-classification-dataset)
        uv run train-baseline-vit \
            --dataset skin-lesions \
            --data-root ./data \
            --output-dir ./artifacts/skin_lesions_vit_baseline \
            --epochs 10 \
            --batch-size 128 \
            --lr 3e-4 \
            --num-workers 6 \
            --precision bf16-mixed
        ;;
    *)
        echo "Usage: sh script/train_baseline_vit.sh [pathmnist|skin-lesions]" >&2
        exit 1
        ;;
esac