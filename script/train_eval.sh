uv run train-eval-token-classifier \
	--token-dir ./artifacts/pathmnist_tokens \
	--output-dir ./artifacts/token_classifier \
	--model-name vit_tiny_patch16_224 \
	--epochs 10 \
	--batch-size 256 \
	--lr 3e-4
