uv run train-eval-token-mlp \
    --dataset pathmnist \
    --token-dir ./artifacts/pathmnist_tokens \
    --output-dir ./artifacts/token_mlp_classifier \
    --embed-dim 128 \
    --hidden-dim 256 \
    --epochs 10 \
    --batch-size 256 \
    --lr 3e-4

# Skin lesions distilled tokens example:
# uv run train-eval-token-mlp \
#     --dataset skin-lesions \
#     --token-dir ./artifacts/skin_lesions_tokens \
#     --output-dir ./artifacts/skin_lesions_token_mlp_classifier \
#     --embed-dim 128 \
#     --hidden-dim 256 \
#     --epochs 10 \
#     --batch-size 128 \
#     --lr 3e-4
