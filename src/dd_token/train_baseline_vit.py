import argparse
from bisect import bisect_right
from io import BytesIO
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightning as L
from huggingface_hub import HfApi, snapshot_download
import medmnist
import numpy as np
import timm
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from medmnist import PathMNIST
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class ViTTrainConfig:
    data_root: str
    dataset: str
    output_dir: str
    model_name: str = "vit_tiny_patch16_224"
    epochs: int = 20
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    image_size: int = 224
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


def default_output_dir_for_dataset(dataset: str) -> str:
    if dataset == "skin-lesions":
        return "./artifacts/skin_lesions_vit"
    return "./artifacts/pathmnist_vit"


def _import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for Hugging Face parquet datasets. Install it with `uv add pyarrow`."
        ) from exc
    return pq


def set_seed(seed: int) -> None:
    L.seed_everything(seed, workers=True)
    np.random.seed(seed)


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


def scalar_bytes_from_precision(precision: str) -> int:
    normalized = precision.lower()
    if "bf16" in normalized:
        return 2
    if "16" in normalized:
        return 2
    return 4


def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024.0:.2f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / (1024.0**2):.2f} MB"
    return f"{num_bytes / (1024.0**3):.2f} GB"


def build_train_transform(cfg: ViTTrainConfig):
    transform_list = [transforms.Resize((cfg.image_size, cfg.image_size), antialias=True)]
    if cfg.train_augment:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(transform_list)


def build_eval_transform(cfg: ViTTrainConfig):
    return transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


class HFParquetImageDataset(Dataset):
    def __init__(
        self,
        parquet_paths: list[Path],
        transform,
        image_column: str = "image",
        label_column: str = "label",
    ):
        if not parquet_paths:
            raise ValueError("parquet_paths cannot be empty.")

        self.parquet_paths = [Path(p) for p in parquet_paths]
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column

        self._index_map: list[tuple[int, int, int]] = []
        self._cumulative_rows: list[int] = []
        self._length = 0

        self._parquet_files: dict[int, Any] = {}
        self._cached_group_key: tuple[int, int] | None = None
        self._cached_images: list[Any] | None = None
        self._cached_labels: list[Any] | None = None

        pq = _import_pyarrow_parquet()
        for file_idx, parquet_path in enumerate(self.parquet_paths):
            parquet_file = pq.ParquetFile(str(parquet_path))
            for row_group_idx in range(parquet_file.num_row_groups):
                num_rows = int(parquet_file.metadata.row_group(row_group_idx).num_rows)
                if num_rows <= 0:
                    continue
                self._length += num_rows
                self._index_map.append((file_idx, row_group_idx, num_rows))
                self._cumulative_rows.append(self._length)

        if self._length <= 0:
            raise ValueError("No rows found in parquet files.")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_parquet_files"] = {}
        state["_cached_group_key"] = None
        state["_cached_images"] = None
        state["_cached_labels"] = None
        return state

    def __len__(self) -> int:
        return self._length

    def _decode_image(self, value: Any) -> Image.Image:
        if isinstance(value, dict):
            img_bytes = value.get("bytes")
            img_path = value.get("path")
            if img_bytes is not None:
                with Image.open(BytesIO(img_bytes)) as img:
                    return img.convert("RGB")
            if img_path:
                with Image.open(img_path) as img:
                    return img.convert("RGB")

        if isinstance(value, (bytes, bytearray)):
            with Image.open(BytesIO(value)) as img:
                return img.convert("RGB")

        if isinstance(value, str):
            with Image.open(value) as img:
                return img.convert("RGB")

        raise ValueError(f"Unsupported image value type: {type(value)}")

    def _load_row_group(self, file_idx: int, row_group_idx: int) -> None:
        key = (file_idx, row_group_idx)
        if self._cached_group_key == key:
            return

        parquet_file = self._parquet_files.get(file_idx)
        if parquet_file is None:
            pq = _import_pyarrow_parquet()
            parquet_file = pq.ParquetFile(str(self.parquet_paths[file_idx]))
            self._parquet_files[file_idx] = parquet_file

        table = parquet_file.read_row_group(row_group_idx, columns=[self.image_column, self.label_column])
        self._cached_images = table.column(self.image_column).to_pylist()
        self._cached_labels = table.column(self.label_column).to_pylist()
        self._cached_group_key = key

    def __getitem__(self, index):
        if index < 0:
            index = self._length + index

        if index < 0 or index >= self._length:
            raise IndexError(f"Index {index} out of range for dataset of length {self._length}.")

        group_idx = bisect_right(self._cumulative_rows, index)
        prev_cumulative = 0 if group_idx == 0 else self._cumulative_rows[group_idx - 1]
        row_in_group = int(index - prev_cumulative)

        file_idx, row_group_idx, _ = self._index_map[group_idx]
        self._load_row_group(file_idx, row_group_idx)

        if self._cached_images is None or self._cached_labels is None:
            raise RuntimeError("Parquet row group cache is unexpectedly empty.")

        image = self._decode_image(self._cached_images[row_in_group])
        label = int(self._cached_labels[row_in_group])
        image_tensor = self.transform(image) if self.transform is not None else image
        return image_tensor, torch.tensor(label, dtype=torch.long)


def _extract_hf_label_names(repo_id: str) -> list[str]:
    info = HfApi().dataset_info(repo_id=repo_id)
    card_data = info.cardData.to_dict() if info.cardData is not None else {}
    dataset_info = card_data.get("dataset_info") if isinstance(card_data, dict) else None
    features = dataset_info.get("features", []) if isinstance(dataset_info, dict) else []

    for feature in features:
        if not isinstance(feature, dict) or feature.get("name") != "label":
            continue
        dtype = feature.get("dtype")
        if not isinstance(dtype, dict):
            continue
        class_label = dtype.get("class_label")
        if not isinstance(class_label, dict):
            continue

        names = class_label.get("names")
        if isinstance(names, list):
            return [str(name) for name in names]
        if isinstance(names, dict):
            indexed = []
            for key, value in names.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                indexed.append((idx, str(value)))
            if indexed:
                indexed.sort(key=lambda x: x[0])
                return [name for _, name in indexed]
    return []


def _collect_parquet_split_files(snapshot_dir: Path, split_name: str) -> list[Path]:
    data_dir = snapshot_dir / "data"
    patterns = [f"{split_name}-*.parquet"]
    if split_name == "validation":
        patterns.append("val-*.parquet")

    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(data_dir.glob(pattern)):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(path)
    return files


class ImageClassificationDataModule(L.LightningDataModule):
    def __init__(self, cfg: ViTTrainConfig):
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.in_channels = 3
        self.num_classes = 0
        self.label_names: list[str] = []

    def setup(self, stage: str | None = None):
        _ = stage
        Path(self.cfg.data_root).mkdir(parents=True, exist_ok=True)

        train_transform = build_train_transform(self.cfg)
        eval_transform = build_eval_transform(self.cfg)

        if self.cfg.dataset == "pathmnist":
            self.train_set = PathMNIST(root=self.cfg.data_root, split="train", transform=train_transform, download=True)
            self.val_set = PathMNIST(root=self.cfg.data_root, split="val", transform=eval_transform, download=True)
            self.test_set = PathMNIST(root=self.cfg.data_root, split="test", transform=eval_transform, download=True)
            self.num_classes = len(medmnist.INFO["pathmnist"]["label"])
            self.label_names = [medmnist.INFO["pathmnist"]["label"][str(i)] for i in range(self.num_classes)]
        else:
            snapshot_dir = Path(
                snapshot_download(
                    repo_id=HF_SKIN_LESIONS_REPO_ID,
                    repo_type="dataset",
                    allow_patterns=["README.md", "data/*.parquet"],
                    cache_dir=str(Path(self.cfg.data_root) / "hf_cache"),
                )
            )

            train_files = _collect_parquet_split_files(snapshot_dir, "train")
            val_files = _collect_parquet_split_files(snapshot_dir, "validation")
            test_files = _collect_parquet_split_files(snapshot_dir, "test")
            if not train_files or not val_files or not test_files:
                raise ValueError("Failed to find train/validation/test parquet splits for skin lesions dataset.")

            self.train_set = HFParquetImageDataset(train_files, transform=train_transform)
            self.val_set = HFParquetImageDataset(val_files, transform=eval_transform)
            self.test_set = HFParquetImageDataset(test_files, transform=eval_transform)

            self.label_names = _extract_hf_label_names(HF_SKIN_LESIONS_REPO_ID)
            self.num_classes = len(self.label_names) if self.label_names else 14
            if not self.label_names:
                self.label_names = [str(i) for i in range(self.num_classes)]

        sample_img, _ = self.train_set[0]
        self.in_channels = int(sample_img.shape[0])

    def _loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_set, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_set, shuffle=False)


def _as_square_patch_size(model: nn.Module) -> int:
    patch = getattr(model.patch_embed, "patch_size", 16)
    if isinstance(patch, tuple):
        if len(patch) != 2 or patch[0] != patch[1]:
            raise ValueError(f"Only square patch size is supported, got {patch}.")
        return int(patch[0])
    return int(patch)


class LitPathMNISTViT(L.LightningModule):
    def __init__(self, cfg: ViTTrainConfig, in_channels: int, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        self.model = timm.create_model(
            cfg.model_name,
            pretrained=False,
            num_classes=num_classes,
            in_chans=in_channels,
            img_size=cfg.image_size,
        )
        for param in self.model.parameters():
            param.requires_grad = True
        self.patch_size = _as_square_patch_size(self.model)
        self.embed_dim = int(getattr(self.model, "embed_dim"))
        self.num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
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


def estimate_token_stats(cfg: ViTTrainConfig, dm: ImageClassificationDataModule, model: LitPathMNISTViT) -> dict:
    patch_size = model.patch_size
    embed_dim = model.embed_dim
    num_prefix_tokens = model.num_prefix_tokens

    if cfg.image_size % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size.")

    num_patch_tokens = (cfg.image_size // patch_size) ** 2
    seq_len_with_cls = num_patch_tokens + num_prefix_tokens
    raw_patch_dim = dm.in_channels * patch_size * patch_size

    scalar_bytes = scalar_bytes_from_precision(cfg.precision)
    scalar_dtype = "fp16/bf16" if scalar_bytes == 2 else "fp32"

    raw_patch_token_bytes = raw_patch_dim * scalar_bytes
    embed_token_bytes = embed_dim * scalar_bytes

    splits = {
        "train": len(dm.train_set),
        "val": len(dm.val_set),
        "test": len(dm.test_set),
    }

    split_stats = {}
    for split_name, sample_count in splits.items():
        raw_total = sample_count * num_patch_tokens * raw_patch_token_bytes
        embed_total = sample_count * seq_len_with_cls * embed_token_bytes
        split_stats[split_name] = {
            "num_samples": int(sample_count),
            "raw_patch_tokens_total": int(sample_count * num_patch_tokens),
            "embedded_tokens_total": int(sample_count * seq_len_with_cls),
            "raw_patch_tokens_bytes": int(raw_total),
            "raw_patch_tokens_human": format_bytes(int(raw_total)),
            "embedded_tokens_bytes": int(embed_total),
            "embedded_tokens_human": format_bytes(int(embed_total)),
        }

    total_raw_bytes = sum(v["raw_patch_tokens_bytes"] for v in split_stats.values())
    total_embed_bytes = sum(v["embedded_tokens_bytes"] for v in split_stats.values())

    return {
        "assumption": {
            "model_name": cfg.model_name,
            "scalar_dtype": scalar_dtype,
            "scalar_bytes": scalar_bytes,
            "image_size": cfg.image_size,
            "patch_size": patch_size,
            "num_prefix_tokens": num_prefix_tokens,
            "in_channels": dm.in_channels,
            "embed_dim": embed_dim,
        },
        "sequence": {
            "patch_tokens_per_image": num_patch_tokens,
            "tokens_per_image_with_cls": seq_len_with_cls,
            "raw_patch_dim": raw_patch_dim,
        },
        "per_image": {
            "raw_patch_tokens_bytes": int(num_patch_tokens * raw_patch_token_bytes),
            "raw_patch_tokens_human": format_bytes(int(num_patch_tokens * raw_patch_token_bytes)),
            "embedded_tokens_bytes": int(seq_len_with_cls * embed_token_bytes),
            "embedded_tokens_human": format_bytes(int(seq_len_with_cls * embed_token_bytes)),
        },
        "splits": split_stats,
        "total": {
            "raw_patch_tokens_bytes": int(total_raw_bytes),
            "raw_patch_tokens_human": format_bytes(int(total_raw_bytes)),
            "embedded_tokens_bytes": int(total_embed_bytes),
            "embedded_tokens_human": format_bytes(int(total_embed_bytes)),
        },
    }


def train(cfg: ViTTrainConfig) -> None:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = ImageClassificationDataModule(cfg)
    dm.setup()

    model = LitPathMNISTViT(cfg=cfg, in_channels=dm.in_channels, num_classes=dm.num_classes)
    token_stats = estimate_token_stats(cfg, dm, model)

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
        "token_stats": token_stats,
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
    parser = argparse.ArgumentParser(description="Train a ViT baseline with Lightning and profile token/memory/time stats.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="pathmnist",
        choices=["pathmnist", "skin-lesions", "skin_lesions", "skin-lesions-classification", HF_SKIN_LESIONS_REPO_ID],
    )
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="vit_tiny_patch16_224")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--image-size", type=int, default=224)
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

    cfg = ViTTrainConfig(
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