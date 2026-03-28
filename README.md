# DD-token: Token Distillation (PathMNIST + NIH Chest X-ray)

This project provides a practical token distillation pipeline for:

- PathMNIST (multiclass)
- NIH Chest X-ray (alkzar90/NIH-Chest-X-ray-dataset, 14 disease labels, multilabel)

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

Run NIH Chest X-ray distillation (resize to 256x256):

```bash
uv run distill-pathmnist-tokens \
  --dataset nih-chest-xray \
  --output-dir ./artifacts/nih_tokens_256 \
  --epochs 8 \
  --batch-size 64 \
  --patch-size 8 \
  --overlap 0.2 \
  --codebook-size 2048 \
  --code-dim 256 \
  --hidden-dim 384 \
  --cls-weight 0.4 \
  --image-size 256
```

Run NIH Chest X-ray distillation without resize (keep original resolution):

```bash
uv run distill-pathmnist-tokens \
  --dataset nih-chest-xray \
  --output-dir ./artifacts/nih_tokens_native \
  --epochs 8 \
  --batch-size 16 \
  --patch-size 8 \
  --overlap 0.2 \
  --codebook-size 2048 \
  --code-dim 256 \
  --hidden-dim 384 \
  --cls-weight 0.4 \
  --image-size 0
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
- `labels`: multiclass `[N]` or multilabel `[N, C]`

## Notes

- This implementation uses a lightweight VQ tokenizer (VQGAN-style discretization) to prioritize a reproducible token pipeline.
- Distillation now includes an auxiliary image-level classification loss (`--cls-weight`) so generated tokens carry more class-discriminative information.

## Train And Evaluate On Distilled Tokens

After token distillation, train and evaluate a token classifier.

PathMNIST example:

```bash
uv run train-eval-token-classifier \
  --token-dir ./artifacts/pathmnist_tokens \
  --output-dir ./artifacts/token_classifier \
  --epochs 20 \
  --batch-size 512 \
  --lr 3e-4 \
  --d-model 128 \
  --num-layers 4 \
  --ff-dim 256 \
  --dropout 0.1 \
  --label-smoothing 0.0 \
  --warmup-ratio 0.0 \
  --class-balance-power 0.0
```

NIH multilabel example:

```bash
uv run train-eval-token-classifier \
  --token-dir ./artifacts/nih_tokens_256 \
  --output-dir ./artifacts/nih_token_classifier \
  --task-type multilabel \
  --threshold 0.5 \
  --epochs 20 \
  --batch-size 256 \
  --lr 3e-4
```

Run TensorBoard for training logs:

```bash
tensorboard --logdir ./artifacts/token_classifier/tb_logs
```

Outputs:

- `best_token_classifier.pt`: best checkpoint selected by validation metric
- `metrics.json`: final test metrics + best checkpoint info + TensorBoard log path
- `tb_logs/`: TensorBoard event files from Lightning
- `checkpoints/`: best and last checkpoint from Lightning callbacks
