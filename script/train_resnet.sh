#!/usr/bin/env sh
set -eu

DATASET="${1:-pathmnist}"

case "$DATASET" in
    pathmnist)
        uv run train-eval-token-resnet \
            --dataset pathmnist \
            --token-dir ./artifacts/pathmnist_tokens \
            --output-dir ./artifacts/token_resnet_classifier \
            --model-name resnet18 \
            --epochs 10 \
            --batch-size 256 \
            --lr 3e-4
        ;;
    skin-lesions|skin_lesions|ahmed-ai/skin-lesions-classification-dataset)
        uv run train-eval-token-resnet \
            --dataset skin-lesions \
            --token-dir ./artifacts/skin_lesions_tokens \
            --output-dir ./artifacts/skin_lesions_token_resnet_classifier \
            --model-name resnet18 \
            --epochs 10 \
            --batch-size 128 \
            --lr 3e-4
        ;;
    *)
        echo "Usage: sh script/train_resnet.sh [pathmnist|skin-lesions]" >&2
        exit 1
        ;;
esac