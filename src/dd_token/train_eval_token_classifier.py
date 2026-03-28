import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
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
    task_type: str = "auto"
    threshold: float = 0.5
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
    log_every_n_steps: int = 20


def set_seed(seed: int) -> None:
    L.seed_everything(seed, workers=True)
    np.random.seed(seed)


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        tokens = data["tokens"].astype(np.int64)
        labels = data["labels"]
    return tokens, labels


def macro_f1_multiclass(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
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


def macro_f1_multilabel(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    f1_values = []
    num_classes = y_true.size(1)
    for c in range(num_classes):
        yt = y_true[:, c]
        yp = y_pred[:, c]

        tp = torch.sum((yp == 1) & (yt == 1)).item()
        fp = torch.sum((yp == 1) & (yt == 0)).item()
        fn = torch.sum((yp == 0) & (yt == 1)).item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1_values.append(0.0)
        else:
            f1_values.append(2 * precision * recall / (precision + recall))
    return float(sum(f1_values) / len(f1_values))


def micro_f1_multilabel(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


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


def build_class_weights_multiclass(labels: np.ndarray, num_classes: int, balance_power: float) -> torch.Tensor:
    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    class_counts[class_counts == 0] = 1.0
    weights = (1.0 / class_counts) ** balance_power
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_pos_weight_multilabel(labels: np.ndarray, balance_power: float) -> torch.Tensor:
    pos = labels.sum(axis=0).astype(np.float64)
    neg = labels.shape[0] - pos
    pos[pos == 0] = 1.0
    ratio = (neg / pos) ** balance_power
    ratio = np.clip(ratio, 1e-3, 1e3)
    return torch.tensor(ratio, dtype=torch.float32)


def make_loader(
    tokens: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: WeightedRandomSampler | None = None,
    label_dtype: torch.dtype = torch.long,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(tokens), torch.from_numpy(labels).to(label_dtype))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def infer_task_type(train_labels: np.ndarray, cfg_task_type: str, token_dir: Path) -> str:
    if cfg_task_type in {"multiclass", "multilabel"}:
        return cfg_task_type

    metadata_path = token_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("task_type") in {"multiclass", "multilabel"}:
            return str(meta["task_type"])

    if train_labels.ndim == 2 and train_labels.shape[1] > 1:
        return "multilabel"
    return "multiclass"


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


class TokenDataModule(L.LightningDataModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.token_dir = Path(cfg.token_dir)

        self.task_type: str = "multiclass"
        self.vocab_size: int = 0
        self.seq_len: int = 0
        self.num_classes: int = 0

        self.class_weights: torch.Tensor | None = None
        self.pos_weight: torch.Tensor | None = None

        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    def setup(self, stage: str | None = None):
        train_tokens, train_labels = load_split(self.token_dir / "train_tokens.npz")
        val_tokens, val_labels = load_split(self.token_dir / "val_tokens.npz")
        test_tokens, test_labels = load_split(self.token_dir / "test_tokens.npz")

        self.task_type = infer_task_type(train_labels, self.cfg.task_type, self.token_dir)

        self.vocab_size = int(max(train_tokens.max(), val_tokens.max(), test_tokens.max()) + 1)
        self.seq_len = int(train_tokens.shape[1])

        if self.task_type == "multiclass":
            if train_labels.ndim == 2 and train_labels.shape[1] == 1:
                train_labels = train_labels.reshape(-1)
                val_labels = val_labels.reshape(-1)
                test_labels = test_labels.reshape(-1)

            train_labels = train_labels.astype(np.int64)
            val_labels = val_labels.astype(np.int64)
            test_labels = test_labels.astype(np.int64)
            self.num_classes = int(max(train_labels.max(), val_labels.max(), test_labels.max()) + 1)

            train_sampler = build_weighted_sampler(train_labels, self.num_classes, balance_power=self.cfg.class_balance_power)
            self.class_weights = build_class_weights_multiclass(
                train_labels,
                self.num_classes,
                balance_power=self.cfg.class_balance_power,
            )

            self.train_loader = make_loader(
                train_tokens,
                train_labels,
                self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                sampler=train_sampler,
                label_dtype=torch.long,
            )
            self.val_loader = make_loader(
                val_tokens,
                val_labels,
                self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                label_dtype=torch.long,
            )
            self.test_loader = make_loader(
                test_tokens,
                test_labels,
                self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                label_dtype=torch.long,
            )
        else:
            train_labels = train_labels.astype(np.float32)
            val_labels = val_labels.astype(np.float32)
            test_labels = test_labels.astype(np.float32)
            self.num_classes = int(train_labels.shape[1])
            self.pos_weight = build_pos_weight_multilabel(train_labels, balance_power=self.cfg.class_balance_power)

            self.train_loader = make_loader(
                train_tokens,
                train_labels,
                self.cfg.batch_size,
                shuffle=True,
                num_workers=self.cfg.num_workers,
                sampler=None,
                label_dtype=torch.float32,
            )
            self.val_loader = make_loader(
                val_tokens,
                val_labels,
                self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                label_dtype=torch.float32,
            )
            self.test_loader = make_loader(
                test_tokens,
                test_labels,
                self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                label_dtype=torch.float32,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_loader is None:
            raise RuntimeError("DataModule is not setup yet.")
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        if self.val_loader is None:
            raise RuntimeError("DataModule is not setup yet.")
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        if self.test_loader is None:
            raise RuntimeError("DataModule is not setup yet.")
        return self.test_loader


class LitTokenClassifier(L.LightningModule):
    def __init__(self, cfg: TrainConfig, dm: TokenDataModule):
        super().__init__()
        self.cfg = cfg
        self.dm = dm

        self.model = TokenClassifier(
            vocab_size=dm.vocab_size,
            seq_len=dm.seq_len,
            num_classes=dm.num_classes,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
        )

        if dm.task_type == "multiclass":
            weights = dm.class_weights if dm.class_weights is not None else None
            if weights is not None:
                weights = weights.clone().detach()
            self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
            self.monitor_metric = "val_macro_f1"
        else:
            pos_weight = dm.pos_weight if dm.pos_weight is not None else None
            if pos_weight is not None:
                pos_weight = pos_weight.clone().detach()
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.monitor_metric = "val_micro_f1"

        self.val_preds: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []
        self.test_preds: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []

        self.save_hyperparameters(asdict(cfg))
        self.save_hyperparameters({
            "task_type": dm.task_type,
            "vocab_size": dm.vocab_size,
            "seq_len": dm.seq_len,
            "num_classes": dm.num_classes,
        })

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.model(tokens)

    @torch.no_grad()
    def _evaluate_on_loader(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        count = 0
        preds_buffer: list[torch.Tensor] = []
        targets_buffer: list[torch.Tensor] = []

        for tokens, labels in loader:
            tokens = tokens.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self(tokens)
            loss = self.criterion(logits, labels)

            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            count += bs

            if self.dm.task_type == "multiclass":
                preds = torch.argmax(logits, dim=1)
                targets = labels.long()
            else:
                preds = (torch.sigmoid(logits) >= self.cfg.threshold).long()
                targets = labels.long()

            preds_buffer.append(preds.detach().cpu())
            targets_buffer.append(targets.detach().cpu())

        y_pred = torch.cat(preds_buffer) if preds_buffer else torch.empty(0)
        y_true = torch.cat(targets_buffer) if targets_buffer else torch.empty(0)
        metrics = {"loss": total_loss / max(1, count)}

        if y_pred.numel() == 0:
            return metrics

        if self.dm.task_type == "multiclass":
            acc = float((y_pred == y_true).float().mean().item())
            macro_f1 = macro_f1_multiclass(y_true, y_pred, num_classes=self.dm.num_classes)
            metrics.update({"acc": acc, "macro_f1": macro_f1})
        else:
            subset_acc = float((y_pred == y_true).all(dim=1).float().mean().item())
            micro_f1 = micro_f1_multilabel(y_true, y_pred)
            macro_f1 = macro_f1_multilabel(y_true, y_pred)
            metrics.update({"subset_acc": subset_acc, "micro_f1": micro_f1, "macro_f1": macro_f1})

        return metrics

    def _shared_step(self, batch, stage: str):
        tokens, labels = batch
        logits = self(tokens)
        loss = self.criterion(logits, labels)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=tokens.size(0))
        return logits, labels, loss

    def training_step(self, batch, batch_idx):
        logits, labels, loss = self._shared_step(batch, stage="train")
        _ = logits
        _ = labels
        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels, loss = self._shared_step(batch, stage="val")

        if self.dm.task_type == "multiclass":
            preds = torch.argmax(logits, dim=1)
            targets = labels.long()
        else:
            preds = (torch.sigmoid(logits) >= self.cfg.threshold).long()
            targets = labels.long()

        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(targets.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        y_pred = torch.cat(self.val_preds) if self.val_preds else torch.empty(0)
        y_true = torch.cat(self.val_targets) if self.val_targets else torch.empty(0)

        if y_pred.numel() == 0:
            return

        if self.dm.task_type == "multiclass":
            acc = float((y_pred == y_true).float().mean().item())
            macro_f1 = macro_f1_multiclass(y_true, y_pred, num_classes=self.dm.num_classes)
            self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_macro_f1", macro_f1, prog_bar=True, on_step=False, on_epoch=True)
        else:
            subset_acc = float((y_pred == y_true).all(dim=1).float().mean().item())
            micro_f1 = micro_f1_multilabel(y_true, y_pred)
            macro_f1 = macro_f1_multilabel(y_true, y_pred)
            self.log("val_subset_acc", subset_acc, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_micro_f1", micro_f1, prog_bar=True, on_step=False, on_epoch=True)
            self.log("val_macro_f1", macro_f1, prog_bar=True, on_step=False, on_epoch=True)

        # Record test metrics every epoch for easier debugging curves in TensorBoard.
        if self.trainer is not None and not self.trainer.sanity_checking:
            test_metrics = self._evaluate_on_loader(self.dm.test_dataloader())
            self.log("test_epoch_loss", test_metrics["loss"], prog_bar=False, on_step=False, on_epoch=True)
            if self.dm.task_type == "multiclass":
                self.log("test_epoch_acc", test_metrics.get("acc", 0.0), prog_bar=False, on_step=False, on_epoch=True)
                self.log(
                    "test_epoch_macro_f1",
                    test_metrics.get("macro_f1", 0.0),
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                )
                self.print(
                    f"epoch_test: loss={test_metrics['loss']:.5f} "
                    f"acc={test_metrics.get('acc', 0.0):.4f} "
                    f"macro_f1={test_metrics.get('macro_f1', 0.0):.4f}"
                )
            else:
                self.log(
                    "test_epoch_subset_acc",
                    test_metrics.get("subset_acc", 0.0),
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    "test_epoch_micro_f1",
                    test_metrics.get("micro_f1", 0.0),
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                )
                self.log(
                    "test_epoch_macro_f1",
                    test_metrics.get("macro_f1", 0.0),
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                )
                self.print(
                    f"epoch_test: loss={test_metrics['loss']:.5f} "
                    f"subset_acc={test_metrics.get('subset_acc', 0.0):.4f} "
                    f"micro_f1={test_metrics.get('micro_f1', 0.0):.4f} "
                    f"macro_f1={test_metrics.get('macro_f1', 0.0):.4f}"
                )

        self.val_preds.clear()
        self.val_targets.clear()

    def test_step(self, batch, batch_idx):
        logits, labels, loss = self._shared_step(batch, stage="test")

        if self.dm.task_type == "multiclass":
            preds = torch.argmax(logits, dim=1)
            targets = labels.long()
        else:
            preds = (torch.sigmoid(logits) >= self.cfg.threshold).long()
            targets = labels.long()

        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        return loss

    def on_test_epoch_end(self):
        y_pred = torch.cat(self.test_preds) if self.test_preds else torch.empty(0)
        y_true = torch.cat(self.test_targets) if self.test_targets else torch.empty(0)

        if y_pred.numel() == 0:
            return

        if self.dm.task_type == "multiclass":
            acc = float((y_pred == y_true).float().mean().item())
            macro_f1 = macro_f1_multiclass(y_true, y_pred, num_classes=self.dm.num_classes)
            self.log("test_acc", acc, on_step=False, on_epoch=True)
            self.log("test_macro_f1", macro_f1, on_step=False, on_epoch=True)
        else:
            subset_acc = float((y_pred == y_true).all(dim=1).float().mean().item())
            micro_f1 = micro_f1_multilabel(y_true, y_pred)
            macro_f1 = macro_f1_multilabel(y_true, y_pred)
            self.log("test_subset_acc", subset_acc, on_step=False, on_epoch=True)
            self.log("test_micro_f1", micro_f1, on_step=False, on_epoch=True)
            self.log("test_macro_f1", macro_f1, on_step=False, on_epoch=True)

        self.test_preds.clear()
        self.test_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        steps_per_epoch = max(1, len(self.dm.train_dataloader()))
        total_steps = self.cfg.epochs * steps_per_epoch
        warmup_steps = int(self.cfg.warmup_ratio * total_steps)
        min_lr_ratio = self.cfg.min_lr / self.cfg.lr

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


def train(cfg: TrainConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = TokenDataModule(cfg)
    dm.setup()

    model = LitTokenClassifier(cfg, dm)

    monitor_metric = model.monitor_metric
    mode = "max"

    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="tb_logs")
    ckpt_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:02d}-{" + monitor_metric + ":.4f}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor=monitor_metric, mode=mode, patience=cfg.early_stop_patience)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        logger=tb_logger,
        callbacks=[ckpt_callback, early_stop, lr_monitor],
        gradient_clip_val=cfg.grad_clip_norm,
        log_every_n_steps=cfg.log_every_n_steps,
    )

    trainer.fit(model, datamodule=dm)

    best_ckpt = ckpt_callback.best_model_path
    if best_ckpt:
        test_results = trainer.test(model=None, datamodule=dm, ckpt_path=best_ckpt)
    else:
        test_results = trainer.test(model=model, datamodule=dm)

    metrics = {
        "config": asdict(cfg),
        "dataset": {
            "task_type": dm.task_type,
            "vocab_size": dm.vocab_size,
            "seq_len": dm.seq_len,
            "num_classes": dm.num_classes,
        },
        "best_checkpoint": best_ckpt,
        "best_score": float(ckpt_callback.best_model_score.item()) if ckpt_callback.best_model_score is not None else None,
        "test": test_results[0] if test_results else {},
        "tensorboard_log_dir": tb_logger.log_dir,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if dm.task_type == "multiclass":
        print(
            f"test_loss={metrics['test'].get('test_loss', float('nan')):.5f} "
            f"test_acc={metrics['test'].get('test_acc', float('nan')):.4f} "
            f"test_macro_f1={metrics['test'].get('test_macro_f1', float('nan')):.4f}"
        )
    else:
        print(
            f"test_loss={metrics['test'].get('test_loss', float('nan')):.5f} "
            f"test_subset_acc={metrics['test'].get('test_subset_acc', float('nan')):.4f} "
            f"test_micro_f1={metrics['test'].get('test_micro_f1', float('nan')):.4f} "
            f"test_macro_f1={metrics['test'].get('test_macro_f1', float('nan')):.4f}"
        )
    print(f"saved model + metrics to {output_dir}")
    print(f"tensorboard logs at {tb_logger.log_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a token classifier on distilled tokens with Lightning.")
    parser.add_argument("--token-dir", type=str, default="./artifacts/pathmnist_tokens")
    parser.add_argument("--output-dir", type=str, default="./artifacts/token_classifier")
    parser.add_argument("--task-type", type=str, default="auto", choices=["auto", "multiclass", "multilabel"])
    parser.add_argument("--threshold", type=float, default=0.5)
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

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        token_dir=args.token_dir,
        output_dir=args.output_dir,
        task_type=args.task_type,
        threshold=args.threshold,
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
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
    )
    set_seed(cfg.seed)
    train(cfg)


if __name__ == "__main__":
    main()
