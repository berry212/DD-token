#!/usr/bin/env sh
set -eu

DATASET="${1:-pathmnist}"

case "$DATASET" in
	pathmnist)
		uv run train-eval-token-classifier \
			--dataset pathmnist \
			--token-dir ./artifacts/pathmnist_tokens \
			--output-dir ./artifacts/pathmnist_vit \
			--model-name vit_tiny_patch16_224 \
			--epochs 10 \
			--batch-size 256 \
			--lr 3e-4
		;;
	dermamnist|derma|dermamnist+)
		uv run train-eval-token-classifier \
			--dataset dermamnist \
			--token-dir ./artifacts/dermamnist_tokens \
			--output-dir ./artifacts/dermamnist_vit \
			--model-name vit_tiny_patch16_224 \
			--epochs 10 \
			--batch-size 128 \
			--lr 3e-4
		;;
	skin-lesions|skin_lesions|ahmed-ai/skin-lesions-classification-dataset)
		uv run train-eval-token-classifier \
			--dataset skin-lesions \
			--token-dir ./artifacts/skin_lesions_tokens \
			--output-dir ./artifacts/skin_lesions_vit \
			--model-name vit_tiny_patch16_224 \
			--epochs 10 \
			--batch-size 128 \
			--lr 3e-4
		;;
	*)
		echo "Usage: sh script/train_vit.sh [pathmnist|dermamnist|skin-lesions]" >&2
		exit 1
		;;
esac
