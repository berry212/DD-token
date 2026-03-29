#!/usr/bin/env sh
set -eu

DATASET="${1:-pathmnist}"

case "$DATASET" in
    pathmnist)
        uv run train-pathmnist-vit \
            --dataset pathmnist \
            --data-root ./data \
            --output-dir ./artifacts/pathmnist_vit \
            --epochs 10 \
            --batch-size 256 \
            --model-name vit_tiny_patch16_224 \
            --image-size 224 \
            --lr 3e-4 \
            --num-workers 6
        ;;
    skin-lesions|skin_lesions|ahmed-ai/skin-lesions-classification-dataset)
        uv run train-pathmnist-vit \
            --dataset skin-lesions \
            --data-root ./data \
            --output-dir ./artifacts/skin_lesions_vit \
            --epochs 10 \
            --batch-size 128 \
            --model-name vit_tiny_patch16_224 \
            --image-size 224 \
            --lr 3e-4 \
            --num-workers 6
        ;;
    *)
        echo "Usage: sh script/train_baseline.sh [pathmnist|skin-lesions]" >&2
        exit 1
        ;;
esac