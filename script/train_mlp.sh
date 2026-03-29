#!/usr/bin/env sh
set -eu

DATASET="${1:-pathmnist}"

case "$DATASET" in
    pathmnist)
        uv run train-eval-token-mlp \
            --dataset pathmnist \
            --token-dir ./artifacts/pathmnist_tokens \
            --output-dir ./artifacts/pathmnist_mlp \
            --embed-dim 128 \
            --hidden-dim 256 \
            --epochs 10 \
            --batch-size 256 \
            --lr 3e-4
        ;;
    skin-lesions|skin_lesions|ahmed-ai/skin-lesions-classification-dataset)
        uv run train-eval-token-mlp \
            --dataset skin-lesions \
            --token-dir ./artifacts/skin_lesions_tokens \
            --output-dir ./artifacts/skin_lesions_mlp \
            --embed-dim 128 \
            --hidden-dim 256 \
            --epochs 10 \
            --batch-size 128 \
            --lr 3e-4
        ;;
    *)
        echo "Usage: sh script/train_mlp.sh [pathmnist|skin-lesions]" >&2
        exit 1
        ;;
esac
