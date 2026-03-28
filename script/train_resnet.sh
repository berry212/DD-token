uv run train-eval-token-resnet \
    --token-dir ./artifacts/pathmnist_tokens \
    --output-dir ./artifacts/token_resnet_classifier \
    --model-name resnet18 \
    --image-size 224 \
    --epochs 10 \
    --batch-size 256 \
    --lr 3e-4