uv run distill-pathmnist-tokens --dataset skin-lesions \
    --data-root ./data \
    --output-dir ./artifacts/skin_lesions_tokens \
    --epochs 5 \
    --batch-size 64 \
    --patch-size 16 \
    --overlap 0.2 \
    --codebook-size 2048 \
    --code-dim 256 \
    --hidden-dim 384 \
    --cls-weight 0.4 \
    --cls-weight-end 0.2 \
    --diversity-weight 0.05 \
    --quant-temperature 1.0 \
    --warmup-ratio 0.05 \
    --min-lr 1e-5 \
    --image-size 256 \
    --num-workers 0

uv run train-eval-token-classifier \
	--dataset skin-lesions \
	--token-dir ./artifacts/skin_lesions_tokens \
	--output-dir ./artifacts/skin_lesions_token_classifier \
	--model-name vit_tiny_patch16_224 \
	--epochs 10 \
	--batch-size 128 \
	--lr 3e-4

uv run train-eval-token-resnet \
    --dataset skin-lesions \
    --token-dir ./artifacts/skin_lesions_tokens \
    --output-dir ./artifacts/skin_lesions_token_resnet_classifier \
    --model-name resnet18 \
    --image-size 224 \
    --epochs 10 \
    --batch-size 128 \
    --lr 3e-4

uv run train-pathmnist-vit \
    --dataset ahmed-ai/skin-lesions-classification-dataset \
    --data-root ./data \
    --output-dir ./artifacts/skin_lesions_vit \
    --epochs 10 \
    --batch-size 64 \
    --model-name vit_tiny_patch16_224 \
    --image-size 224 \
    --lr 3e-4


