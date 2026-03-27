# DD-token: PathMNIST Token Distillation

This project provides a practical token distillation pipeline for PathMNIST:

1. Image preprocessing (normalization + optional denoise)
2. Overlapping patch extraction
3. VQ-style codebook discretization (VQGAN-inspired tokenizer)
4. Export train/val/test token datasets for downstream models

## Quick Start

Install dependencies:

```bash
uv sync
```

Run distillation:

```bash
uv run distill-pathmnist-tokens \
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

## Outputs

The script writes the following files under the output directory:

- `train_tokens.npz`
- `val_tokens.npz`
- `test_tokens.npz`
- `vq_tokenizer.pt`
- `metadata.json`

Each `*_tokens.npz` contains:

- `tokens`: shape `[N, L]`, where `L` is patch token length per image
- `labels`: shape `[N]`

## Notes

- This implementation uses a lightweight VQ tokenizer (VQGAN-style discretization) to prioritize a reproducible token pipeline.
- Distillation now includes an auxiliary image-level classification loss (`--cls-weight`) so generated tokens carry more class-discriminative information.

## Train And Evaluate On Distilled Tokens

After token distillation, train and evaluate a token classifier:

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

Outputs:

- `best_token_classifier.pt`: best checkpoint selected by validation accuracy
- `metrics.json`: training history + final test metrics (`acc`, `macro_f1`, `loss`)
