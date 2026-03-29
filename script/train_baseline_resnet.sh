#!/usr/bin/env sh
set -eu

DATASET="${1:-pathmnist}"

case "$DATASET" in
    pathmnist)
        uv run train-baseline-resnet \
            --dataset pathmnist \
            --data-root ./data \
            --output-dir ./artifacts/pathmnist_resnet_baseline \
            --epochs 10 \
            --batch-size 256 \
            --model-name resnet18 \
            --lr 3e-4 \
            --num-workers 6
        ;;
    skin-lesions|skin_lesions|ahmed-ai/skin-lesions-classification-dataset)
        uv run train-baseline-resnet \
            --dataset skin-lesions \
            --data-root ./data \
            --output-dir ./artifacts/skin_lesions_resnet_baseline \
            --epochs 10 \
            --batch-size 128 \
            --model-name resnet18 \
            --lr 3e-4 \
            --num-workers 6
        ;;
    *)
        echo "Usage: sh script/train_baseline_resnet.sh [pathmnist|skin-lesions]" >&2
        exit 1
        ;;
esac
