import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


@dataclass
class TrainConfig:
    dataset: str
    token_dir: str
    output_dir: str
    embed_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    label_smoothing: float = 0.0
    warmup_ratio: float = 0.05
    min_lr: float = 1e-5
    grad_clip_norm: float = 1.0
    class_balance_power: float = 0.0
    early_stop_patience: int = 10
    num_workers: int = 4
    seed: int = 42
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
    log_every_n_steps: int = 20


HF_SKIN_LESIONS_REPO_ID = "ahmed-ai/skin-lesions-classification-dataset"


def normalize_dataset_name(dataset: str) -> str:
    normalized = dataset.strip().lower()
    if normalized == "pathmnist":
        return "pathmnist"
    if normalized in {"skin-lesions", "skin_lesions", "skin-lesions-classification", HF_SKIN_LESIONS_REPO_ID.lower()}:
        return "skin-lesions"
    raise ValueError(
        "Unsupported dataset name. Supported values: pathmnist, skin-lesions, "
        f"{HF_SKIN_LESIONS_REPO_ID}."
    )


def default_token_dir_for_dataset(dataset: str) -> str:
    if dataset == "skin-lesions":
        return "./artifacts/skin_lesions_tokens"
    return "./artifacts/pathmnist_tokens"


def default_output_dir_for_dataset(dataset: str) -> str:
    if dataset == "skin-lesions":
        return "./artifacts/skin_lesions_token_mlp_classifier"
    return "./artifacts/token_mlp_classifier"


def set_seed(seed: int) -> None:
    L.seed_everything(seed, workers=True)
    np.random.seed(seed)


def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024.0:.2f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / (1024.0**2):.2f} MB"
    return f"{num_bytes / (1024.0**3):.2f} GB"


def load_split(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        tokens = data["tokens"].astype(np.int64)
        labels = data["labels"]
    return tokens, labels


def multiclass_auc_ovr(y_true: torch.Tensor, y_prob: torch.Tensor) -> float | None:
    if y_true.numel() == 0 or y_prob.numel() == 0:
        return None

    y_true_np = y_true.detach().cpu().numpy()
    y_prob_np = y_prob.detach().cpu().numpy()

    try:
        auc = roc_auc_score(y_true_np, y_prob_np, average="macro", multi_class="ovr")
    except ValueError:
        return None
    return float(auc)


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


def build_split_token_size_stats(tokens: np.ndarray, labels: np.ndarray, npz_path: Path) -> dict:
    raw_tokens_bytes = int(tokens.nbytes)
    raw_labels_bytes = int(labels.nbytes)
    raw_total_bytes = raw_tokens_bytes + raw_labels_bytes
    npz_file_bytes = int(npz_path.stat().st_size)
    return {
        "num_samples": int(tokens.shape[0]),
        "tokens_shape": list(tokens.shape),
        "labels_shape": list(labels.shape),
        "raw_tokens_bytes": raw_tokens_bytes,
        "raw_tokens_human": format_bytes(raw_tokens_bytes),
        "raw_labels_bytes": raw_labels_bytes,
        "raw_labels_human": format_bytes(raw_labels_bytes),
        "raw_total_bytes": raw_total_bytes,
        "raw_total_human": format_bytes(raw_total_bytes),
        "npz_file_bytes": npz_file_bytes,
        "npz_file_human": format_bytes(npz_file_bytes),
    }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


class TokenSequenceDataset(Dataset):
    def __init__(self, tokens: np.ndarray, labels: np.ndarray):
        self.tokens = tokens.astype(np.int64)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def __getitem__(self, idx):
        token_ids = torch.from_numpy(self.tokens[idx]).to(torch.long)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return token_ids, label


class TokenMLPHead(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Pool sequence tokens and classify with a lightweight MLP head.
        x = self.token_emb(token_ids)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class TrainProfileCallback(Callback):
    def __init__(self):
        super().__init__()
        self.fit_start_time: float | None = None
        self.epoch_start_time: float | None = None
        self.total_train_time_sec: float = 0.0
        self.epoch_train_time_sec: list[float] = []
        self.epoch_peak_allocated_mb: list[float] = []
        self.epoch_peak_reserved_mb: list[float] = []

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.fit_start_time = time.perf_counter()
        self.epoch_train_time_sec.clear()
        self.epoch_peak_allocated_mb.clear()
        self.epoch_peak_reserved_mb.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.epoch_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.epoch_start_time is None:
            return

        duration = float(time.perf_counter() - self.epoch_start_time)
        self.epoch_train_time_sec.append(duration)

        if torch.cuda.is_available():
            allocated_mb = float(torch.cuda.max_memory_allocated() / (1024.0**2))
            reserved_mb = float(torch.cuda.max_memory_reserved() / (1024.0**2))
            self.epoch_peak_allocated_mb.append(allocated_mb)
            self.epoch_peak_reserved_mb.append(reserved_mb)

            if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
                experiment = trainer.logger.experiment
                if hasattr(experiment, "add_scalar"):
                    epoch_idx = int(trainer.current_epoch + 1)
                    experiment.add_scalar("profile/epoch_peak_allocated_mb", allocated_mb, epoch_idx)
                    experiment.add_scalar("profile/epoch_peak_reserved_mb", reserved_mb, epoch_idx)

        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            experiment = trainer.logger.experiment
            if hasattr(experiment, "add_scalar"):
                epoch_idx = int(trainer.current_epoch + 1)
                experiment.add_scalar("profile/epoch_train_time_sec", duration, epoch_idx)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.fit_start_time is None:
            return
        self.total_train_time_sec = float(time.perf_counter() - self.fit_start_time)

    def summary(self) -> dict:
        out = {
            "total_train_time_sec": self.total_train_time_sec,
            "epoch_train_time_sec": self.epoch_train_time_sec,
            "avg_epoch_train_time_sec": float(sum(self.epoch_train_time_sec) / max(1, len(self.epoch_train_time_sec))),
        }

        if self.epoch_peak_allocated_mb:
            out["epoch_peak_allocated_mb"] = self.epoch_peak_allocated_mb
            out["peak_allocated_mb"] = float(max(self.epoch_peak_allocated_mb))
        else:
            out["epoch_peak_allocated_mb"] = []
            out["peak_allocated_mb"] = None

        if self.epoch_peak_reserved_mb:
            out["epoch_peak_reserved_mb"] = self.epoch_peak_reserved_mb
            out["peak_reserved_mb"] = float(max(self.epoch_peak_reserved_mb))
        else:
            out["epoch_peak_reserved_mb"] = []
            out["peak_reserved_mb"] = None

        return out


class TokenDataModule(L.LightningDataModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.token_dir = Path(cfg.token_dir)

        self.vocab_size: int = 0
        self.seq_len: int = 0
        self.num_classes: int = 0

        self.class_weights: torch.Tensor | None = None
        self.token_size_stats: dict | None = None

        self.train_loader: DataLoader | None = None
        self.val_loader: DataLoader | None = None
        self.test_loader: DataLoader | None = None

    def setup(self, stage: str | None = None):
        _ = stage
        train_path = self.token_dir / "train_tokens.npz"
        val_path = self.token_dir / "val_tokens.npz"
        test_path = self.token_dir / "test_tokens.npz"
        train_tokens, train_labels = load_split(train_path)
        val_tokens, val_labels = load_split(val_path)
        test_tokens, test_labels = load_split(test_path)

        self.vocab_size = int(max(train_tokens.max(), val_tokens.max(), test_tokens.max()) + 1)
        self.seq_len = int(train_tokens.shape[1])

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

        split_stats = {
            "train": build_split_token_size_stats(train_tokens, train_labels, train_path),
            "val": build_split_token_size_stats(val_tokens, val_labels, val_path),
            "test": build_split_token_size_stats(test_tokens, test_labels, test_path),
        }
        total_raw_bytes = sum(v["raw_total_bytes"] for v in split_stats.values())
        total_npz_file_bytes = sum(v["npz_file_bytes"] for v in split_stats.values())
        self.token_size_stats = {
            "splits": split_stats,
            "total": {
                "raw_total_bytes": int(total_raw_bytes),
                "raw_total_human": format_bytes(int(total_raw_bytes)),
                "npz_file_bytes": int(total_npz_file_bytes),
                "npz_file_human": format_bytes(int(total_npz_file_bytes)),
            },
        }

        train_dataset = TokenSequenceDataset(train_tokens, train_labels)
        val_dataset = TokenSequenceDataset(val_tokens, val_labels)
        test_dataset = TokenSequenceDataset(test_tokens, test_labels)

        self.train_loader = make_loader(
            train_dataset,
            self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            sampler=train_sampler,
        )
        self.val_loader = make_loader(
            val_dataset,
            self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )
        self.test_loader = make_loader(
            test_dataset,
            self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
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


class LitTokenMLPClassifier(L.LightningModule):
    def __init__(self, cfg: TrainConfig, dm: TokenDataModule):
        super().__init__()
        self.cfg = cfg
        self.dm = dm

        self.model = TokenMLPHead(
            vocab_size=dm.vocab_size,
            num_classes=dm.num_classes,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )

        weights = dm.class_weights if dm.class_weights is not None else None
        if weights is not None:
            weights = weights.clone().detach()
        self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
        self.monitor_metric = "val_acc"

        self.val_preds: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []
        self.val_probs: list[torch.Tensor] = []
        self.test_preds: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []
        self.test_probs: list[torch.Tensor] = []

        self.save_hyperparameters(asdict(cfg))
        self.save_hyperparameters(
            {
                "vocab_size": dm.vocab_size,
                "seq_len": dm.seq_len,
                "num_classes": dm.num_classes,
            }
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model(token_ids)

    @torch.no_grad()
    def _evaluate_on_loader(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        count = 0
        preds_buffer: list[torch.Tensor] = []
        targets_buffer: list[torch.Tensor] = []
        probs_buffer: list[torch.Tensor] = []

        for token_ids, labels in loader:
            token_ids = token_ids.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self(token_ids)
            loss = self.criterion(logits, labels)

            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            count += bs

            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            targets = labels.long()

            preds_buffer.append(preds.detach().cpu())
            targets_buffer.append(targets.detach().cpu())
            probs_buffer.append(probs.detach().cpu())

        y_pred = torch.cat(preds_buffer) if preds_buffer else torch.empty(0)
        y_true = torch.cat(targets_buffer) if targets_buffer else torch.empty(0)
        y_prob = torch.cat(probs_buffer) if probs_buffer else torch.empty(0)
        metrics = {"loss": total_loss / max(1, count)}

        if y_pred.numel() == 0:
            return metrics

        acc = float((y_pred == y_true).float().mean().item())
        auc = multiclass_auc_ovr(y_true, y_prob)
        metrics["acc"] = acc
        if auc is not None:
            metrics["auc"] = auc

        return metrics

    def _shared_step(self, batch, stage: str):
        token_ids, labels = batch
        logits = self(token_ids)
        loss = self.criterion(logits, labels)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=token_ids.size(0))
        return logits, labels, loss

    def training_step(self, batch, batch_idx):
        logits, labels, loss = self._shared_step(batch, stage="train")
        _ = logits
        _ = labels
        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels, loss = self._shared_step(batch, stage="val")

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        targets = labels.long()

        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(targets.detach().cpu())
        self.val_probs.append(probs.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        y_pred = torch.cat(self.val_preds) if self.val_preds else torch.empty(0)
        y_true = torch.cat(self.val_targets) if self.val_targets else torch.empty(0)
        y_prob = torch.cat(self.val_probs) if self.val_probs else torch.empty(0)

        if y_pred.numel() == 0:
            return

        acc = float((y_pred == y_true).float().mean().item())
        auc = multiclass_auc_ovr(y_true, y_prob)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        if auc is not None:
            self.log("val_auc", auc, prog_bar=True, on_step=False, on_epoch=True)

        if self.trainer is not None and not self.trainer.sanity_checking:
            test_metrics = self._evaluate_on_loader(self.dm.test_dataloader())
            self.log("test_epoch_loss", test_metrics["loss"], prog_bar=False, on_step=False, on_epoch=True)
            self.log("test_epoch_acc", test_metrics.get("acc", 0.0), prog_bar=False, on_step=False, on_epoch=True)
            if "auc" in test_metrics:
                self.log("test_epoch_auc", test_metrics["auc"], prog_bar=False, on_step=False, on_epoch=True)
                self.print(
                    f"epoch_test: loss={test_metrics['loss']:.5f} "
                    f"acc={test_metrics.get('acc', 0.0):.4f} "
                    f"auc={test_metrics.get('auc', float('nan')):.4f}"
                )
            else:
                self.print(
                    f"epoch_test: loss={test_metrics['loss']:.5f} "
                    f"acc={test_metrics.get('acc', 0.0):.4f}"
                )

        self.val_preds.clear()
        self.val_targets.clear()
        self.val_probs.clear()

    def test_step(self, batch, batch_idx):
        logits, labels, loss = self._shared_step(batch, stage="test")

        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        targets = labels.long()

        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(targets.detach().cpu())
        self.test_probs.append(probs.detach().cpu())
        return loss

    def on_test_epoch_end(self):
        y_pred = torch.cat(self.test_preds) if self.test_preds else torch.empty(0)
        y_true = torch.cat(self.test_targets) if self.test_targets else torch.empty(0)
        y_prob = torch.cat(self.test_probs) if self.test_probs else torch.empty(0)

        if y_pred.numel() == 0:
            return

        acc = float((y_pred == y_true).float().mean().item())
        auc = multiclass_auc_ovr(y_true, y_prob)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        if auc is not None:
            self.log("test_auc", auc, on_step=False, on_epoch=True)

        self.test_preds.clear()
        self.test_targets.clear()
        self.test_probs.clear()

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

    model = LitTokenMLPClassifier(cfg, dm)

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
    profile_callback = TrainProfileCallback()

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        logger=tb_logger,
        callbacks=[ckpt_callback, early_stop, lr_monitor, profile_callback],
        gradient_clip_val=cfg.grad_clip_norm,
        log_every_n_steps=cfg.log_every_n_steps,
    )

    trainer.fit(model, datamodule=dm)

    best_ckpt = ckpt_callback.best_model_path
    if best_ckpt:
        test_results = trainer.test(model=None, datamodule=dm, ckpt_path=best_ckpt)
    else:
        test_results = trainer.test(model=model, datamodule=dm)

    profile_summary = profile_callback.summary()

    metrics = {
        "config": asdict(cfg),
        "dataset": {
            "name": cfg.dataset,
            "vocab_size": dm.vocab_size,
            "seq_len": dm.seq_len,
            "num_classes": dm.num_classes,
        },
        "token_size_stats": dm.token_size_stats,
        "training_profile": profile_summary,
        "best_checkpoint": best_ckpt,
        "best_score": float(ckpt_callback.best_model_score.item()) if ckpt_callback.best_model_score is not None else None,
        "test": test_results[0] if test_results else {},
        "tensorboard_log_dir": tb_logger.log_dir,
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(
        f"test_loss={metrics['test'].get('test_loss', float('nan')):.5f} "
        f"test_acc={metrics['test'].get('test_acc', float('nan')):.4f} "
        f"test_auc={metrics['test'].get('test_auc', float('nan')):.4f}"
    )
    total_token_stats = metrics.get("token_size_stats", {}).get("total", {})
    print(
        f"token_size(raw={total_token_stats.get('raw_total_human')}, "
        f"npz={total_token_stats.get('npz_file_human')})"
    )
    print(
        f"train_time={profile_summary.get('total_train_time_sec', 0.0):.2f}s "
        f"peak_allocated={profile_summary.get('peak_allocated_mb')}MB "
        f"peak_reserved={profile_summary.get('peak_reserved_mb')}MB"
    )
    print(f"saved model + metrics to {output_dir}")
    print(f"tensorboard logs at {tb_logger.log_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a distilled-token MLP classifier with Lightning.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="pathmnist",
        choices=["pathmnist", "skin-lesions", "skin_lesions", "skin-lesions-classification", HF_SKIN_LESIONS_REPO_ID],
    )
    parser.add_argument("--token-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
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
    dataset = normalize_dataset_name(args.dataset)
    token_dir = args.token_dir if args.token_dir is not None else default_token_dir_for_dataset(dataset)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir_for_dataset(dataset)

    cfg = TrainConfig(
        dataset=dataset,
        token_dir=token_dir,
        output_dir=output_dir,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
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
