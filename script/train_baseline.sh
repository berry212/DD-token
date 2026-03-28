uv run train-pathmnist-vit \
    --dataset pathmnist \
    --data-root ./data \
    --output-dir ./artifacts/pathmnist_vit \
    --epochs 10 \
    --batch-size 256 \
    --model-name vit_tiny_patch16_224 \
    --image-size 224 \
    --lr 3e-4

# Skin lesions baseline ViT example:
# uv run train-pathmnist-vit \
#     --dataset ahmed-ai/skin-lesions-classification-dataset \
#     --data-root ./data \
#     --output-dir ./artifacts/skin_lesions_vit \
#     --epochs 10 \
#     --batch-size 128 \
#     --model-name vit_tiny_patch16_224 \
#     --image-size 224 \
#     --lr 3e-4