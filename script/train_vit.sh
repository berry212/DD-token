#!/usr/bin/env sh
set -eu

DATASET="${1:-pathmnist}"

case "$DATASET" in
	pathmnist)
		uv run train-eval-token-classifier \
			--dataset pathmnist \
			--token-dir ./artifacts/pathmnist_tokens \
			--output-dir ./artifacts/token_vit \
			--model-name vit_tiny_patch16_224 \
			--pseudo-image-mode native \
			--native-patch-size 1 \
			--epochs 10 \
			--batch-size 256 \
			--lr 3e-4
		;;
	skin-lesions|skin_lesions|ahmed-ai/skin-lesions-classification-dataset)
		uv run train-eval-token-classifier \
			--dataset skin-lesions \
			--token-dir ./artifacts/skin_lesions_tokens \
			--output-dir ./artifacts/skin_lesions_token_vit \
			--model-name vit_tiny_patch16_224 \
			--pseudo-image-mode native \
			--native-patch-size 1 \
			--epochs 10 \
			--batch-size 128 \
			--lr 3e-4
		;;
	*)
		echo "Usage: sh script/train_classifier.sh [pathmnist|skin-lesions]" >&2
		exit 1
		;;
esac
