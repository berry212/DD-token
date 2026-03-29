# uv run train-eval-token-classifier \
# 	--dataset pathmnist \
# 	--token-dir ./artifacts/pathmnist_tokens \
# 	--output-dir ./artifacts/token_classifier \
# 	--model-name vit_tiny_patch16_224 \
# 	--pseudo-image-mode native \
# 	--native-patch-size 1 \
# 	--epochs 10 \
# 	--batch-size 256 \
# 	--lr 3e-4

# Skin lesions distilled tokens example:
uv run train-eval-token-classifier \
	--dataset skin-lesions \
	--token-dir ./artifacts/skin_lesions_tokens \
	--output-dir ./artifacts/skin_lesions_token_classifier \
	--model-name vit_tiny_patch16_224 \
	--pseudo-image-mode native \
	--native-patch-size 1 \
	--epochs 10 \
	--batch-size 128 \
	--lr 3e-4
