uv run train-pathmnist-vit \
    --data-root ./data \
    --output-dir ./artifacts/pathmnist_vit \
    --epochs 10 \
    --batch-size 256 \
    --model-name vit_tiny_patch16_224 \
    --image-size 224 \
    --lr 3e-4