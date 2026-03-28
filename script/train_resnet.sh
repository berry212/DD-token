uv run train-eval-token-resnet \
    --dataset pathmnist \
    --token-dir ./artifacts/pathmnist_tokens \
    --output-dir ./artifacts/token_resnet_classifier \
    --model-name resnet18 \
    --image-size 224 \
    --epochs 10 \
    --batch-size 256 \
    --lr 3e-4

# Skin lesions distilled tokens example:
# uv run train-eval-token-resnet \
#     --dataset skin-lesions \
#     --token-dir ./artifacts/skin_lesions_tokens \
#     --output-dir ./artifacts/skin_lesions_token_resnet_classifier \
#     --model-name resnet18 \
#     --image-size 224 \
#     --epochs 10 \
#     --batch-size 128 \
#     --lr 3e-4