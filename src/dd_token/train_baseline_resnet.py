import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from dd_token.train_baseline_vit import (
    HF_SKIN_LESIONS_REPO_ID,
    ImageClassificationDataModule,
    TrainProfileCallback,
    macro_f1_multiclass,
    multiclass_auc_ovr,
    normalize_dataset_name,
    set_seed,
)


@dataclass
class ResNetTrainConfig:
    data_root: str
    dataset: str
    output_dir: str
    model_name: str = "resnet18"
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    image_size: int = 0
    label_smoothing: float = 0.0
    warmup_ratio: float = 0.05
    min_lr: float = 1e-5
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 10
    num_workers: int = 4
    train_augment: bool = True
    seed: int = 42
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
    log_every_n_steps: int = 20


def default_output_dir_for_dataset(dataset: str) -> str:
    if dataset == "skin-lesions":
        return "./artifacts/skin_lesions_resnet_baseline"
    return "./artifacts/pathmnist_resnet_baseline"


def collate_pad_variable_images(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([int(item[1]) for item in batch], dtype=torch.long)

    if not images:
        return torch.empty((0, 0, 0, 0), dtype=torch.float32), labels

    max_h = max(int(img.shape[-2]) for img in images)
    max_w = max(int(img.shape[-1]) for img in images)

    padded_images = []
    for image in images:
        if image.ndim != 3:
            raise ValueError(f"Expected image tensor [C,H,W], got shape={tuple(image.shape)}")
        pad_h = max_h - int(image.shape[-2])
        pad_w = max_w - int(image.shape[-1])
        padded_images.append(F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0))

    return torch.stack(padded_images, dim=0), labels


class ResNetImageClassificationDataModule(ImageClassificationDataModule):
    def _loader(self, dataset, shuffle: bool):
        collate_fn = collate_pad_variable_images if int(self.cfg.image_size) <= 0 else None
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
            collate_fn=collate_fn,
        )


class LitPathMNISTResNet(L.LightningModule):
    def __init__(self, cfg: ResNetTrainConfig, in_channels: int, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(num_classes)

        self.model = timm.create_model(
            cfg.model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_channels,
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        self.monitor_metric = "val_acc"

        self.val_preds: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []
        self.val_probs: list[torch.Tensor] = []
        self.test_preds: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []
        self.test_probs: list[torch.Tensor] = []

        self.save_hyperparameters(asdict(cfg))
        self.save_hyperparameters({"in_channels": in_channels, "num_classes": num_classes})

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)

    @torch.no_grad()
    def _evaluate_on_loader(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        count = 0
        preds_buffer: list[torch.Tensor] = []
        targets_buffer: list[torch.Tensor] = []
        probs_buffer: list[torch.Tensor] = []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.view(-1).long().to(self.device, non_blocking=True)

            logits = self(images)
            loss = self.criterion(logits, labels)

            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            count += bs

            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            preds_buffer.append(preds.detach().cpu())
            targets_buffer.append(labels.detach().cpu())
            probs_buffer.append(probs.detach().cpu())

        y_pred = torch.cat(preds_buffer) if preds_buffer else torch.empty(0)
        y_true = torch.cat(targets_buffer) if targets_buffer else torch.empty(0)
        y_prob = torch.cat(probs_buffer) if probs_buffer else torch.empty(0)

        metrics = {"loss": total_loss / max(1, count)}
        if y_pred.numel() == 0:
            return metrics

        acc = float((y_pred == y_true).float().mean().item())
        macro_f1 = macro_f1_multiclass(y_true, y_pred, num_classes=self.num_classes)
        auc = multiclass_auc_ovr(y_true, y_prob)
        metrics["acc"] = acc
        metrics["macro_f1"] = macro_f1
        if auc is not None:
            metrics["auc"] = auc
        return metrics

    def _step(self, batch, stage: str):
        images, labels = batch
        labels = labels.view(-1).long()
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = float((preds == labels).float().mean().item())

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log(f"{stage}_acc", acc, prog_bar=(stage != "train"), on_step=False, on_epoch=True, batch_size=images.size(0))
        return logits, labels, loss

    def training_step(self, batch, batch_idx):
        logits, labels, loss = self._step(batch, stage="train")
        _ = logits
        _ = labels
        return loss

    def validation_step(self, batch, batch_idx):
        logits, labels, loss = self._step(batch, stage="val")
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(labels.detach().cpu())
        self.val_probs.append(probs.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        y_pred = torch.cat(self.val_preds) if self.val_preds else torch.empty(0)
        y_true = torch.cat(self.val_targets) if self.val_targets else torch.empty(0)
        y_prob = torch.cat(self.val_probs) if self.val_probs else torch.empty(0)

        if y_pred.numel() == 0:
            return

        acc = float((y_pred == y_true).float().mean().item())
        macro_f1 = macro_f1_multiclass(y_true, y_pred, num_classes=self.num_classes)
        auc = multiclass_auc_ovr(y_true, y_prob)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        if auc is not None:
            self.log("val_auc", auc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_macro_f1", macro_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.print(
            f"epoch_val: acc={acc:.4f} "
            f"auc={auc if auc is not None else float('nan'):.4f} "
            f"macro_f1={macro_f1:.4f}"
        )

        if self.trainer is not None and not self.trainer.sanity_checking and self.trainer.datamodule is not None:
            test_metrics = self._evaluate_on_loader(self.trainer.datamodule.test_dataloader())
            self.log("test_epoch_loss", test_metrics["loss"], prog_bar=False, on_step=False, on_epoch=True)
            self.log("test_epoch_acc", test_metrics.get("acc", 0.0), prog_bar=False, on_step=False, on_epoch=True)
            if "auc" in test_metrics:
                self.log("test_epoch_auc", test_metrics["auc"], prog_bar=False, on_step=False, on_epoch=True)
            if "macro_f1" in test_metrics:
                self.log(
                    "test_epoch_macro_f1",
                    test_metrics["macro_f1"],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                )
            self.print(
                f"epoch_test: loss={test_metrics['loss']:.5f} "
                f"acc={test_metrics.get('acc', float('nan')):.4f} "
                f"auc={test_metrics.get('auc', float('nan')):.4f} "
                f"macro_f1={test_metrics.get('macro_f1', float('nan')):.4f}"
            )

        self.val_preds.clear()
        self.val_targets.clear()
        self.val_probs.clear()

    def test_step(self, batch, batch_idx):
        logits, labels, loss = self._step(batch, stage="test")
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)
        self.test_preds.append(preds.detach().cpu())
        self.test_targets.append(labels.detach().cpu())
        self.test_probs.append(probs.detach().cpu())
        return loss

    def on_test_epoch_end(self):
        y_pred = torch.cat(self.test_preds) if self.test_preds else torch.empty(0)
        y_true = torch.cat(self.test_targets) if self.test_targets else torch.empty(0)
        y_prob = torch.cat(self.test_probs) if self.test_probs else torch.empty(0)

        if y_pred.numel() == 0:
            return

        acc = float((y_pred == y_true).float().mean().item())
        macro_f1 = macro_f1_multiclass(y_true, y_pred, num_classes=self.num_classes)
        auc = multiclass_auc_ovr(y_true, y_prob)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        if auc is not None:
            self.log("test_auc", auc, on_step=False, on_epoch=True)
        self.log("test_macro_f1", macro_f1, on_step=False, on_epoch=True)

        self.test_preds.clear()
        self.test_targets.clear()
        self.test_probs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        train_loader = self.trainer.datamodule.train_dataloader()
        total_steps = max(1, self.cfg.epochs * len(train_loader))
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


def train(cfg: ResNetTrainConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = ResNetImageClassificationDataModule(cfg)
    dm.setup()

    model = LitPathMNISTResNet(cfg=cfg, in_channels=dm.in_channels, num_classes=dm.num_classes)

    monitor_metric = model.monitor_metric
    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="tb_logs")

    ckpt_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:02d}-{val_acc:.4f}",
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor=monitor_metric, mode="max", patience=cfg.early_stop_patience)
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
            "hf_repo_id": HF_SKIN_LESIONS_REPO_ID if cfg.dataset == "skin-lesions" else None,
            "num_classes": dm.num_classes,
            "label_names": dm.label_names,
            "in_channels": dm.in_channels,
            "num_train": len(dm.train_set),
            "num_val": len(dm.val_set),
            "num_test": len(dm.test_set),
        },
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
        f"test_auc={metrics['test'].get('test_auc', float('nan')):.4f} "
        f"test_macro_f1={metrics['test'].get('test_macro_f1', float('nan')):.4f}"
    )
    print(
        f"train_time={profile_summary.get('total_train_time_sec', 0.0):.2f}s "
        f"peak_allocated={profile_summary.get('peak_allocated_mb')}MB "
        f"peak_reserved={profile_summary.get('peak_reserved_mb')}MB"
    )
    print(f"saved metrics to {output_dir / 'metrics.json'}")
    print(f"tensorboard logs at {tb_logger.log_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet baseline on original datasets with Lightning.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="pathmnist",
        choices=["pathmnist", "skin-lesions", "skin_lesions", "skin-lesions-classification", HF_SKIN_LESIONS_REPO_ID],
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Resize target for input images. Use <=0 to keep original image size (no resize).",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-train-augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = normalize_dataset_name(args.dataset)
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir_for_dataset(dataset)

    cfg = ResNetTrainConfig(
        data_root=args.data_root,
        dataset=dataset,
        output_dir=output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        label_smoothing=args.label_smoothing,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_lr,
        grad_clip_norm=args.grad_clip_norm,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        train_augment=not args.no_train_augment,
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
