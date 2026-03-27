import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


@dataclass
class TrainConfig:
    token_dir: str
    output_dir: str
    epochs: int = 20
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-2
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    ff_dim: int = 256
    dropout: float = 0.1
    label_smoothing: float = 0.0
    warmup_ratio: float = 0.0
    min_lr: float = 1e-5
    grad_clip_norm: float = 1.0
    class_balance_power: float = 0.0
    early_stop_patience: int = 10
    num_workers: int = 0
    seed: int = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        tokens = data["tokens"].astype(np.int64)
        labels = data["labels"].astype(np.int64)
    return tokens, labels


def macro_f1_score(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    f1_values = []
    for c in range(num_classes):
        tp = torch.sum((y_pred == c) & (y_true == c)).item()
        fp = torch.sum((y_pred == c) & (y_true != c)).item()
        fn = torch.sum((y_pred != c) & (y_true == c)).item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))
    return float(sum(f1_values) / len(f1_values))


class TokenClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        ff_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(tokens)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb
        x = self.encoder(x)
        x = self.norm(x)
        x = x[:, 0]
        return self.head(x)


def build_weighted_sampler(labels: np.ndarray, num_classes: int, balance_power: float) -> WeightedRandomSampler:
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    class_counts[class_counts == 0] = 1.0
    class_weights = (1.0 / class_counts) ** balance_power
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


def build_class_weights(labels: np.ndarray, num_classes: int, balance_power: float) -> torch.Tensor:
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    class_counts[class_counts == 0] = 1.0
    weights = (1.0 / class_counts) ** balance_power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(
    tokens: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(tokens), torch.from_numpy(labels))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, num_classes: int) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    all_preds = []
    all_labels = []

    for tokens, labels in loader:
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(tokens)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        count += bs

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds)
        all_labels.append(labels)

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_labels)
    acc = float((y_pred == y_true).float().mean().item())
    macro_f1 = macro_f1_score(y_true, y_pred, num_classes=num_classes)

    return {
        "loss": total_loss / max(1, count),
        "acc": acc,
        "macro_f1": macro_f1,
    }


def train(cfg: TrainConfig) -> None:
    token_dir = Path(cfg.token_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tokens, train_labels = load_split(token_dir / "train_tokens.npz")
    val_tokens, val_labels = load_split(token_dir / "val_tokens.npz")
    test_tokens, test_labels = load_split(token_dir / "test_tokens.npz")

    vocab_size = int(max(train_tokens.max(), val_tokens.max(), test_tokens.max()) + 1)
    seq_len = int(train_tokens.shape[1])
    num_classes = int(max(train_labels.max(), val_labels.max(), test_labels.max()) + 1)

    train_sampler = build_weighted_sampler(train_labels, num_classes, balance_power=cfg.class_balance_power)
    class_weights = build_class_weights(train_labels, num_classes, balance_power=cfg.class_balance_power)

    train_loader = make_loader(
        train_tokens,
        train_labels,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        sampler=train_sampler,
    )
    val_loader = make_loader(val_tokens, val_labels, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = make_loader(test_tokens, test_labels, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TokenClassifier(
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_classes=num_classes,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=cfg.label_smoothing)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    min_lr_ratio = cfg.min_lr / cfg.lr
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    best_val_acc = -1.0
    best_val_macro_f1 = -1.0
    best_state = None
    history = []
    no_improve_epochs = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        seen = 0

        for tokens, labels in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(tokens)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            bs = labels.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs

        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"epoch={epoch + 1:02d} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6e}"
        )

        improved = (val_metrics["macro_f1"] > best_val_macro_f1) or (
            val_metrics["macro_f1"] == best_val_macro_f1 and val_metrics["acc"] > best_val_acc
        )
        if improved:
            best_val_acc = val_metrics["acc"]
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= cfg.early_stop_patience:
            print(f"early_stop at epoch={epoch + 1:02d} (patience={cfg.early_stop_patience})")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)

    torch.save(best_state, output_dir / "best_token_classifier.pt")

    result = {
        "config": {
            "token_dir": str(token_dir),
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "d_model": cfg.d_model,
            "nhead": cfg.nhead,
            "num_layers": cfg.num_layers,
            "ff_dim": cfg.ff_dim,
            "dropout": cfg.dropout,
            "label_smoothing": cfg.label_smoothing,
            "warmup_ratio": cfg.warmup_ratio,
            "min_lr": cfg.min_lr,
            "grad_clip_norm": cfg.grad_clip_norm,
            "class_balance_power": cfg.class_balance_power,
            "early_stop_patience": cfg.early_stop_patience,
            "num_workers": cfg.num_workers,
            "seed": cfg.seed,
        },
        "dataset": {
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            "num_classes": num_classes,
            "train_size": int(train_tokens.shape[0]),
            "val_size": int(val_tokens.shape[0]),
            "test_size": int(test_tokens.shape[0]),
        },
        "best_val_acc": best_val_acc,
        "best_val_macro_f1": best_val_macro_f1,
        "test": test_metrics,
        "history": history,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"test_loss={test_metrics['loss']:.5f} "
        f"test_acc={test_metrics['acc']:.4f} "
        f"test_macro_f1={test_metrics['macro_f1']:.4f}"
    )
    print(f"saved model + metrics to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a token classifier on distilled PathMNIST tokens.")
    parser.add_argument("--token-dir", type=str, default="./artifacts/pathmnist_tokens")
    parser.add_argument("--output-dir", type=str, default="./artifacts/token_classifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--class-balance-power", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        token_dir=args.token_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_lr,
        grad_clip_norm=args.grad_clip_norm,
        class_balance_power=args.class_balance_power,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    set_seed(cfg.seed)
    train(cfg)


if __name__ == "__main__":
    main()
