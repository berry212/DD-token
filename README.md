# DD-token: Token Distillation

This project provides a practical token distillation pipeline for:

- PathMNIST (multiclass)
- Skin Lesions (Hugging Face: ahmed-ai/skin-lesions-classification-dataset, multiclass)

1. Image preprocessing (normalization + optional denoise)
2. Overlapping patch extraction
3. VQ-style codebook discretization (VQGAN-inspired tokenizer)
4. Export train/val/test token datasets for downstream models

## Quick Start

Install dependencies:

```bash
uv sync
```

Run PathMNIST distillation:

```bash
uv run distill-pathmnist-tokens \
  --dataset pathmnist \
  --data-root ./data \
  --output-dir ./artifacts/pathmnist_tokens \
  --epochs 16 \
  --batch-size 256 \
  --patch-size 4 \
  --overlap 0.2 \
  --codebook-size 2048 \
  --code-dim 256 \
  --hidden-dim 384 \
  --cls-weight 0.4
```

Run Skin Lesions distillation:

```bash
uv run distill-pathmnist-tokens \
  --dataset skin-lesions \
  --data-root ./data \
  --output-dir ./artifacts/skin_lesions_tokens \
  --epochs 12 \
  --batch-size 64 \
  --patch-size 16 \
  --overlap 0.2 \
  --codebook-size 2048 \
  --code-dim 256 \
  --hidden-dim 384 \
  --cls-weight 0.4
```

Equivalent dataset identifier for `--dataset`:

- `skin-lesions`
- `ahmed-ai/skin-lesions-classification-dataset`

Run TensorBoard for distillation logs:

```bash
tensorboard --logdir ./artifacts/pathmnist_tokens/tb_logs
```

## Outputs

The script writes the following files under the output directory:

- `train_tokens.npz`
- `val_tokens.npz`
- `test_tokens.npz`
- `vq_tokenizer.pt`
- `metadata.json`

Each `*_tokens.npz` contains:

- `tokens`: shape `[N, L]`, where `L` is patch token length per image
- `labels`: multiclass `[N]`

## Notes

- This implementation uses a lightweight VQ tokenizer (VQGAN-style discretization) to prioritize a reproducible token pipeline.
- Distillation now includes an auxiliary image-level classification loss (`--cls-weight`) so generated tokens carry more class-discriminative information.

## Train And Evaluate On Distilled Tokens

After token distillation, train and evaluate a token classifier.

PathMNIST example:

```bash
uv run train-eval-token-classifier \
  --dataset pathmnist \
  --token-dir ./artifacts/pathmnist_tokens \
  --output-dir ./artifacts/token_classifier \
  --model-name vit_tiny_patch16_224 \
  --epochs 10 \
  --batch-size 256 \
  --lr 3e-4
```

Skin lesions token-classifier example:

```bash
uv run train-eval-token-classifier \
  --dataset skin-lesions \
  --token-dir ./artifacts/skin_lesions_tokens \
  --output-dir ./artifacts/skin_lesions_token_classifier \
  --model-name vit_tiny_patch16_224 \
  --epochs 10 \
  --batch-size 128 \
  --lr 3e-4
```

Run TensorBoard for training logs:

```bash
tensorboard --logdir ./artifacts/token_classifier/tb_logs
```

## Train A Simple MLP Head On Distilled Tokens

Train a lightweight classifier with distilled token sequences directly (token embedding + pooled MLP head), and record ACC/AUC, token size, GPU memory peak, and training time:

```bash
uv run train-eval-token-mlp \
  --dataset pathmnist \
  --token-dir ./artifacts/pathmnist_tokens \
  --output-dir ./artifacts/token_mlp_classifier \
  --embed-dim 128 \
  --hidden-dim 256 \
  --epochs 10 \
  --batch-size 256 \
  --lr 3e-4
```

Skin lesions token-MLP example:

```bash
uv run train-eval-token-mlp \
  --dataset skin-lesions \
  --token-dir ./artifacts/skin_lesions_tokens \
  --output-dir ./artifacts/skin_lesions_token_mlp_classifier \
  --embed-dim 128 \
  --hidden-dim 256 \
  --epochs 10 \
  --batch-size 128 \
  --lr 3e-4
```

Run TensorBoard for MLP training logs:

```bash
tensorboard --logdir ./artifacts/token_mlp_classifier/tb_logs
```

## Train A ResNet On Token Pseudo-Images

Train a ResNet classifier by reshaping distilled token sequences to pseudo-images (with automatic resize), while recording ACC/AUC, token size, GPU memory peak, and training time:

```bash
uv run train-eval-token-resnet \
  --dataset pathmnist \
  --token-dir ./artifacts/pathmnist_tokens \
  --output-dir ./artifacts/token_resnet_classifier \
  --model-name resnet18 \
  --image-size 224 \
  --epochs 10 \
  --batch-size 256 \
  --lr 3e-4
```

Skin lesions token-ResNet example:

```bash
uv run train-eval-token-resnet \
  --dataset skin-lesions \
  --token-dir ./artifacts/skin_lesions_tokens \
  --output-dir ./artifacts/skin_lesions_token_resnet_classifier \
  --model-name resnet18 \
  --image-size 224 \
  --epochs 10 \
  --batch-size 128 \
  --lr 3e-4
```

Run TensorBoard for ResNet training logs:

```bash
tensorboard --logdir ./artifacts/token_resnet_classifier/tb_logs
```

## Train A ViT Baseline On PathMNIST

Train an image-level ViT baseline with Lightning and TensorBoard, while profiling token size usage, GPU memory, and training time:

```bash
uv run train-pathmnist-vit \
  --dataset pathmnist \
  --data-root ./data \
  --output-dir ./artifacts/pathmnist_vit \
  --epochs 20 \
  --batch-size 256 \
  --model-name vit_tiny_patch16_224 \
  --image-size 224 \
  --lr 3e-4
```

Skin lesions baseline ViT example:

```bash
uv run train-pathmnist-vit \
  --dataset ahmed-ai/skin-lesions-classification-dataset \
  --data-root ./data \
  --output-dir ./artifacts/skin_lesions_vit \
  --epochs 20 \
  --batch-size 128 \
  --model-name vit_tiny_patch16_224 \
  --image-size 224 \
  --lr 3e-4
```

Run TensorBoard for ViT training logs:

```bash
tensorboard --logdir ./artifacts/pathmnist_vit/tb_logs
```

`./artifacts/pathmnist_vit/metrics.json` includes:

- full training config
- token statistics (patch token count, token bytes per image/split/total)
- training profile (total and per-epoch train time, peak GPU memory allocated/reserved)
- best checkpoint and test metrics (including ACC and AUC)
- TensorBoard log directory

Outputs:

- `best_token_classifier.pt`: best checkpoint selected by validation metric
- `metrics.json`: final test metrics + best checkpoint info + TensorBoard log path
- `tb_logs/`: TensorBoard event files from Lightning
- `checkpoints/`: best and last checkpoint from Lightning callbacks
