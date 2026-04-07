import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, override

import lightning as L
import medmnist
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from medmnist import DermaMNIST, PathMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import VQModel

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

SUPPORTED_DATASET_CHOICES = ["pathmnist", "dermamnist"]
VIT_CLS_BACKBONE = "vit_tiny_patch16_224"
VIT_CLS_IMAGE_SIZE = 224


@dataclass
class DistillConfig:
    data_root: str
    output_dir: str
    dataset: str = "pathmnist"
    vqvae_model: str = "CompVis/ldm-celebahq-256"
    vqvae_subfolder: str | None = "vqvae"
    vqvae_revision: str | None = None
    encode_batch_size: int = 8
    quantize_chunk_size: int = 2048
    epochs: int = 12
    batch_size: int = 128
    lr: float = 3e-4
    rvq_stages: int = 2
    num_workers: int = 0
    recon_weight: float = 1.0
    commit_weight: float = 0.25
    cls_weight: float = 0.4
    cls_weight_end: float = 0.2
    diversity_weight: float = 0.05
    quant_temperature: float = 1.0
    warmup_ratio: float = 0.05
    min_lr: float = 1e-5
    blur: bool = True
    seed: int = 42
    image_size: int = 128
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "bf16-mixed"
    log_every_n_steps: int = 20
    ipc: int = 0


def check_dataset_name(dataset: str) -> str:
    normalized = dataset.strip().lower()
    if normalized not in SUPPORTED_DATASET_CHOICES:
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported values: {SUPPORTED_DATASET_CHOICES}.")
    return normalized


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


def _iter_with_progress(loader: DataLoader, split_name: str):
    if tqdm is None:
        return loader
    return tqdm(loader, desc=f"encode-{split_name}", leave=False)


class PretrainedRVQTokenizer(nn.Module):
    def __init__(
        self,
        model_name: str,
        rvq_stages: int,
        subfolder: str | None = "vqvae",
        revision: str | None = None,
        cache_dir: str | None = None,
        beta: float = 0.25,
        temperature: float = 1.0,
        quantize_chunk_size: int = 4096,
    ):
        super().__init__()
        if rvq_stages < 1:
            raise ValueError("rvq_stages must be >= 1.")

        load_kwargs: dict[str, Any] = {}
        if subfolder:
            load_kwargs["subfolder"] = subfolder
        if revision:
            load_kwargs["revision"] = revision
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        self.vqvae = VQModel.from_pretrained(model_name, **load_kwargs)
        _ = self._codebook_weight()

        self.rvq_stages = int(rvq_stages)
        self.beta = float(beta)
        self.temperature = float(temperature)
        self.quantize_chunk_size = int(max(1, quantize_chunk_size))

    def _codebook_weight(self) -> torch.Tensor:
        quantize = self.vqvae.quantize
        embedding = getattr(quantize, "embedding", None)
        if embedding is not None and hasattr(embedding, "weight"):
            return embedding.weight

        embed = getattr(quantize, "embed", None)
        if isinstance(embed, torch.Tensor):
            if embed.shape[0] < embed.shape[1]:
                return embed.t().contiguous()
            return embed.contiguous()

        raise RuntimeError("Could not locate codebook embedding weights in pretrained VQ-VAE.")

    @property
    def codebook_size(self) -> int:
        return int(self._codebook_weight().shape[0])

    @property
    def code_dim(self) -> int:
        return int(self._codebook_weight().shape[1])

    def codebook_weight(self) -> torch.Tensor:
        return self._codebook_weight()

    def _nearest_codebook(self, flat_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        codebook = self._codebook_weight()
        flat_latents = flat_latents.to(device=codebook.device, dtype=codebook.dtype)

        codebook_t = codebook.t()
        codebook_norm = (codebook**2).sum(dim=1)

        all_indices: list[torch.Tensor] = []
        all_quantized: list[torch.Tensor] = []
        prob_sums = torch.zeros(codebook.size(0), device=codebook.device, dtype=codebook.dtype)
        total_rows = 0

        for start in range(0, flat_latents.size(0), self.quantize_chunk_size):
            end = min(start + self.quantize_chunk_size, flat_latents.size(0))
            chunk = flat_latents[start:end]
            distances = (chunk**2).sum(dim=1, keepdim=True) + codebook_norm.unsqueeze(0) - 2 * chunk @ codebook_t
            indices = torch.argmin(distances, dim=1)
            quantized = codebook.index_select(0, indices)

            soft_assign = F.softmax(-distances / max(self.temperature, 1e-6), dim=1)
            prob_sums += soft_assign.sum(dim=0)
            total_rows += int(chunk.size(0))

            all_indices.append(indices)
            all_quantized.append(quantized)

        avg_probs = prob_sums / float(max(1, total_rows))
        entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
        max_entropy = math.log(max(2, codebook.size(0)))
        diversity = 1.0 - entropy / max_entropy

        return torch.cat(all_indices, dim=0), torch.cat(all_quantized, dim=0), diversity

    def quantize_latents(self, latents: torch.Tensor):
        bsz, channels, grid_h, grid_w = latents.shape

        residual = latents.permute(0, 2, 3, 1).reshape(-1, channels)
        z_q_total = torch.zeros_like(residual)
        indices_per_stage: list[torch.Tensor] = []
        commit_terms: list[torch.Tensor] = []
        diversity_terms: list[torch.Tensor] = []

        for _ in range(self.rvq_stages):
            stage_indices, stage_quantized, diversity = self._nearest_codebook(residual)
            stage_quantized_st = residual + (stage_quantized - residual).detach()
            z_q_total = z_q_total + stage_quantized_st

            commit = F.mse_loss(residual, stage_quantized.detach()) + self.beta * F.mse_loss(
                stage_quantized,
                residual.detach(),
            )
            commit_terms.append(commit)
            diversity_terms.append(diversity)
            indices_per_stage.append(stage_indices)

            residual = residual - stage_quantized.detach()

        z_q = z_q_total.view(bsz, grid_h, grid_w, channels).permute(0, 3, 1, 2).contiguous()
        indices = torch.stack(indices_per_stage, dim=1).view(bsz, grid_h, grid_w, self.rvq_stages)
        indices = indices.permute(0, 3, 1, 2).contiguous()

        commit_loss = torch.stack(commit_terms).mean()
        diversity_loss = torch.stack(diversity_terms).mean()
        return z_q, indices, commit_loss, diversity_loss

    def forward(self, images: torch.Tensor):
        latents = self.vqvae.encode(images).latents
        z_q, indices, commit_loss, diversity_loss = self.quantize_latents(latents)
        recon = self.vqvae.decode(z_q).sample
        recon_loss = F.mse_loss(recon, images)
        return recon, z_q, indices, recon_loss, commit_loss, diversity_loss

    @torch.no_grad()
    def encode_tokens(self, images: torch.Tensor) -> torch.Tensor:
        latents = self.vqvae.encode(images).latents
        _, indices, _, _ = self.quantize_latents(latents)
        return indices


def build_transforms(blur: bool, image_size: int, enable_resize: bool = True):
    transforms_list = []
    if enable_resize and image_size > 0:
        transforms_list.append(transforms.Resize((image_size, image_size)))
    transforms_list.append(transforms.ToTensor())
    if blur:
        transforms_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)))
    transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms.Compose(transforms_list)


def _build_medmnist_split(dataset_cls, cfg: DistillConfig, split: str, transform, native_size: int | None = None):
    kwargs = {
        "root": cfg.data_root,
        "split": split,
        "transform": transform,
        "download": True,
    }
    if native_size is not None:
        kwargs["size"] = native_size
    elif cfg.image_size > 0:
        kwargs["size"] = cfg.image_size
    return dataset_cls(**kwargs)


def make_pathmnist_datasets(cfg: DistillConfig):
    Path(cfg.data_root).mkdir(parents=True, exist_ok=True)
    transform = build_transforms(cfg.blur, cfg.image_size, enable_resize=True)
    train_set = _build_medmnist_split(PathMNIST, cfg, split="train", transform=transform)
    val_set = _build_medmnist_split(PathMNIST, cfg, split="val", transform=transform)
    test_set = _build_medmnist_split(PathMNIST, cfg, split="test", transform=transform)

    num_targets = len(medmnist.INFO["pathmnist"]["label"])
    label_names = [medmnist.INFO["pathmnist"]["label"][str(i)] for i in range(num_targets)]
    metadata = {"task_type": "multiclass", "num_targets": num_targets, "label_names": label_names}
    return train_set, val_set, test_set, metadata


def make_dermamnist_datasets(cfg: DistillConfig):
    Path(cfg.data_root).mkdir(parents=True, exist_ok=True)
    transform = build_transforms(cfg.blur, cfg.image_size, enable_resize=False)
    train_set = _build_medmnist_split(DermaMNIST, cfg, split="train", transform=transform, native_size=cfg.image_size)
    val_set = _build_medmnist_split(DermaMNIST, cfg, split="val", transform=transform, native_size=cfg.image_size)
    test_set = _build_medmnist_split(DermaMNIST, cfg, split="test", transform=transform, native_size=cfg.image_size)

    num_targets = len(medmnist.INFO["dermamnist"]["label"])
    label_names = [medmnist.INFO["dermamnist"]["label"][str(i)] for i in range(num_targets)]
    metadata = {"task_type": "multiclass", "num_targets": num_targets, "label_names": label_names}
    return train_set, val_set, test_set, metadata


def make_datasets(cfg: DistillConfig):
    if cfg.dataset == "pathmnist":
        return make_pathmnist_datasets(cfg)
    if cfg.dataset == "dermamnist":
        return make_dermamnist_datasets(cfg)
    raise Exception("critical error")


class DistillDataModule(L.LightningDataModule):
    def __init__(self, cfg: DistillConfig):
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.task_type = "multiclass"
        self.num_targets = 0
        self.label_names: list[str] = []

    def setup(self, stage: str | None = None):
        self.train_set, self.val_set, self.test_set, meta = make_datasets(self.cfg)
        self.task_type = meta["task_type"]
        self.num_targets = meta["num_targets"]
        self.label_names = meta["label_names"]

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


class LitPretrainedVQDistiller(L.LightningModule):
    def __init__(self, cfg: DistillConfig, num_targets: int):
        super().__init__()
        self.cfg = cfg

        self.tokenizer = PretrainedRVQTokenizer(
            model_name=cfg.vqvae_model,
            rvq_stages=cfg.rvq_stages,
            subfolder=cfg.vqvae_subfolder,
            revision=cfg.vqvae_revision,
            cache_dir=str(Path(cfg.data_root) / "hf_cache"),
            beta=cfg.commit_weight,
            temperature=cfg.quant_temperature,
            quantize_chunk_size=cfg.quantize_chunk_size,
        )

        self.vit_classifier = timm.create_model(
            VIT_CLS_BACKBONE,
            pretrained=True,
            num_classes=num_targets,
            in_chans=3,
            img_size=VIT_CLS_IMAGE_SIZE,
        )
        # Freeze ViT backbone to save memory, only keep the classification head trainable.
        for name, param in self.vit_classifier.named_parameters():
            param.requires_grad = name.startswith("head")

        self.register_buffer(
            "vit_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "vit_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.cls_criterion = nn.CrossEntropyLoss()
        self.monitor_metric = "val_acc"

        self.val_preds: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []

        self.save_hyperparameters(asdict(cfg))
        self.save_hyperparameters(
            {
                "task_type": "multiclass",
                "num_targets": num_targets,
                "tokenizer_type": "pretrained_diffusers_vqvae_rvq",
            }
        )

    def _current_cls_weight(self) -> float:
        if self.cfg.epochs <= 1:
            return self.cfg.cls_weight_end
        ratio = self.current_epoch / float(max(1, self.cfg.epochs - 1))
        return self.cfg.cls_weight + ratio * (self.cfg.cls_weight_end - self.cfg.cls_weight)

    def _shared_forward(self, images: torch.Tensor, labels: torch.Tensor):
        recon, z_q, _, recon_loss, commit_loss, diversity_loss = self.tokenizer(images)

        vit_inputs = ((recon + 1.0) * 0.5).clamp(0.0, 1.0)
        if vit_inputs.shape[-1] != VIT_CLS_IMAGE_SIZE or vit_inputs.shape[-2] != VIT_CLS_IMAGE_SIZE:
            vit_inputs = F.interpolate(
                vit_inputs,
                size=(VIT_CLS_IMAGE_SIZE, VIT_CLS_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )
        vit_inputs = (vit_inputs - self.vit_mean) / self.vit_std

        cls_logits = self.vit_classifier(vit_inputs)
        cls_loss = self.cls_criterion(cls_logits, labels.view(-1).long())

        cls_weight_now = self._current_cls_weight()
        total_loss = (
            self.cfg.recon_weight * recon_loss
            + commit_loss
            + cls_weight_now * cls_loss
            + self.cfg.diversity_weight * diversity_loss
        )
        return cls_logits, total_loss, recon_loss, commit_loss, cls_loss, diversity_loss, cls_weight_now

    def training_step(self, batch, batch_idx):
        _ = batch_idx
        images, labels = batch
        logits, total_loss, recon_loss, commit_loss, cls_loss, diversity_loss, cls_weight_now = self._shared_forward(
            images,
            labels,
        )
        _ = logits

        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train_commit_loss", commit_loss, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train_cls_loss", cls_loss, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train_diversity_loss", diversity_loss, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train_cls_weight", cls_weight_now, on_step=False, on_epoch=True, batch_size=images.size(0))
        return total_loss

    def validation_step(self, batch, batch_idx):
        _ = batch_idx
        images, labels = batch
        logits, total_loss, _, _, _, _, _ = self._shared_forward(images, labels)

        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.size(0))

        preds = torch.argmax(logits, dim=1)
        targets = labels.view(-1).long()
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return

        y_pred = torch.cat(self.val_preds)
        y_true = torch.cat(self.val_targets)
        metric = float((y_pred == y_true).float().mean().item())
        self.log("val_acc", metric, prog_bar=True, on_step=False, on_epoch=True)
        self.print(f"epoch={self.current_epoch + 1:02d} val_acc={metric:.4f}")

        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)

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


class DistillProfileCallback(Callback):
    def __init__(self):
        super().__init__()
        self.fit_start_time: float | None = None
        self.epoch_start_time: float | None = None
        self.total_distill_time_sec: float = 0.0
        self.epoch_distill_time_sec: list[float] = []
        self.epoch_peak_allocated_mb: list[float] = []
        self.epoch_peak_reserved_mb: list[float] = []

    @override
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        _ = trainer
        _ = pl_module
        self.fit_start_time = time.perf_counter()
        self.epoch_distill_time_sec.clear()
        self.epoch_peak_allocated_mb.clear()
        self.epoch_peak_reserved_mb.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @override
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        _ = trainer
        _ = pl_module
        self.epoch_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @override
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        _ = pl_module
        if self.epoch_start_time is None:
            return

        duration = float(time.perf_counter() - self.epoch_start_time)
        self.epoch_distill_time_sec.append(duration)

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
                experiment.add_scalar("profile/epoch_distill_time_sec", duration, epoch_idx)

    @override
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        _ = trainer
        _ = pl_module
        if self.fit_start_time is None:
            return
        self.total_distill_time_sec = float(time.perf_counter() - self.fit_start_time)

    def summary(self) -> dict:
        out = {
            "total_distill_time_sec": self.total_distill_time_sec,
            "epoch_distill_time_sec": self.epoch_distill_time_sec,
            "avg_epoch_distill_time_sec": float(sum(self.epoch_distill_time_sec) / max(1, len(self.epoch_distill_time_sec))),
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


@torch.no_grad()
def export_split_tokens(
    split_name: str,
    loader: DataLoader,
    tokenizer: Any,
    output_dir: Path,
    device: torch.device,
):
    tokenizer.eval()
    all_tokens = []
    all_labels = []

    for images, labels in _iter_with_progress(loader, split_name):
        images = images.to(device, non_blocking=True)
        token_ids = tokenizer.encode_tokens(images)
        all_tokens.append(token_ids.cpu().numpy().astype(np.int32))
        all_labels.append(labels.view(-1).cpu().numpy().astype(np.int64))

    tokens_np = np.concatenate(all_tokens, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    output_path = output_dir / f"{split_name}_tokens.npz"
    np.savez_compressed(output_path, tokens=tokens_np, labels=labels_np)
    print(f"saved {split_name}: {output_path} | pseudo_tokens shape={tokens_np.shape} | labels shape={labels_np.shape}")

    return _build_npz_size_stats(tokens_np, labels_np, output_path)


def _build_npz_size_stats(tokens_np: np.ndarray, labels_np: np.ndarray, npz_path: Path) -> dict[str, Any]:
    raw_tokens_bytes = int(tokens_np.nbytes)
    raw_labels_bytes = int(labels_np.nbytes)
    raw_total_bytes = raw_tokens_bytes + raw_labels_bytes
    file_bytes = int(npz_path.stat().st_size)
    return {
        "num_samples": int(tokens_np.shape[0]),
        "tokens_shape": list(tokens_np.shape),
        "labels_shape": list(labels_np.shape),
        "raw_tokens_bytes": raw_tokens_bytes,
        "raw_tokens_human": format_bytes(raw_tokens_bytes),
        "raw_labels_bytes": raw_labels_bytes,
        "raw_labels_human": format_bytes(raw_labels_bytes),
        "raw_total_bytes": raw_total_bytes,
        "raw_total_human": format_bytes(raw_total_bytes),
        "npz_file_bytes": file_bytes,
        "npz_file_human": format_bytes(file_bytes),
    }


def _compress_train_tokens_with_classwise_kmeans(
    output_dir: Path,
    ipc: int,
    seed: int,
    codebook_size: int,
) -> dict[str, Any]:
    from sklearn.cluster import MiniBatchKMeans

    train_path = output_dir / "train_tokens.npz"
    with np.load(train_path) as data:
        train_tokens = data["tokens"]
        train_labels = data["labels"].reshape(-1).astype(np.int64)

    flat_tokens = train_tokens.reshape(train_tokens.shape[0], -1).astype(np.float32)
    token_tail_shape = tuple(int(v) for v in train_tokens.shape[1:])

    unique_classes = sorted(int(c) for c in np.unique(train_labels))
    selected_tokens: list[np.ndarray] = []
    selected_labels: list[int] = []
    per_class_before: dict[str, int] = {}
    per_class_after: dict[str, int] = {}

    for class_id in unique_classes:
        class_indices = np.where(train_labels == class_id)[0]
        class_count = int(class_indices.size)
        per_class_before[str(class_id)] = class_count

        class_vectors = flat_tokens[class_indices]
        n_clusters = int(min(ipc, class_count))
        per_class_after[str(class_id)] = n_clusters

        if n_clusters >= class_count:
            chosen_local = np.arange(class_count, dtype=np.int64)
        else:
            batch_size = int(min(max(256, 8 * n_clusters), class_count))
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=seed,
                batch_size=batch_size,
                n_init="auto",
            )
            kmeans.fit(class_vectors)

            centers = kmeans.cluster_centers_
            cluster_assign = kmeans.labels_
            chosen_local_list: list[int] = []

            for cluster_id in range(n_clusters):
                members = np.where(cluster_assign == cluster_id)[0]
                center = centers[cluster_id]
                if members.size > 0:
                    candidate_vectors = class_vectors[members]
                    nearest_member_idx = int(np.argmin(np.sum((candidate_vectors - center) ** 2, axis=1)))
                    chosen_local_list.append(int(members[nearest_member_idx]))
                else:
                    fallback_idx = int(np.argmin(np.sum((class_vectors - center) ** 2, axis=1)))
                    chosen_local_list.append(fallback_idx)

            chosen_local = np.asarray(chosen_local_list, dtype=np.int64)

        chosen_global = class_indices[chosen_local]
        selected_tokens.append(flat_tokens[chosen_global])
        selected_labels.extend([class_id] * int(chosen_global.size))

    compressed_flat = np.concatenate(selected_tokens, axis=0)
    compressed_labels = np.asarray(selected_labels, dtype=np.int64)
    compressed_tokens = compressed_flat.reshape((compressed_flat.shape[0],) + token_tail_shape)

    compressed_tokens = np.rint(compressed_tokens).astype(np.int32)
    compressed_tokens = np.clip(compressed_tokens, 0, int(codebook_size) - 1)

    full_backup_path = output_dir / "train_tokens_full.npz"
    if not full_backup_path.exists():
        np.savez_compressed(full_backup_path, tokens=train_tokens, labels=train_labels)

    np.savez_compressed(train_path, tokens=compressed_tokens, labels=compressed_labels)

    train_size_stats = _build_npz_size_stats(compressed_tokens, compressed_labels, train_path)
    ratio = float(train_tokens.shape[0] / max(1, compressed_tokens.shape[0]))
    return {
        "enabled": True,
        "ipc_requested": int(ipc),
        "num_classes": int(len(unique_classes)),
        "train_samples_before": int(train_tokens.shape[0]),
        "train_samples_after": int(compressed_tokens.shape[0]),
        "train_sample_compression_ratio": ratio,
        "train_tokens_backup_path": str(full_backup_path),
        "per_class_samples_before": per_class_before,
        "per_class_samples_after": per_class_after,
        "train_size_stats": train_size_stats,
    }


def make_export_loader(dataset, cfg: DistillConfig, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(1, cfg.encode_batch_size),
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )


def train_tokenizer(cfg: DistillConfig):
    pipeline_start_time = time.perf_counter()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = DistillDataModule(cfg)
    dm.setup()
    model = LitPretrainedVQDistiller(cfg=cfg, num_targets=dm.num_targets)

    tb_logger = TensorBoardLogger(save_dir=str(output_dir), name="tb_logs")
    monitor_metric = model.monitor_metric

    ckpt_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:02d}-{" + monitor_metric + ":.4f}",
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        save_last=True,
        save_weights_only=True,
    )
    early_stop = EarlyStopping(monitor=monitor_metric, mode="max", patience=max(3, cfg.epochs // 4))
    lr_monitor = LearningRateMonitor(logging_interval="step")
    profile_callback = DistillProfileCallback()

    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision=cfg.precision,
        logger=tb_logger,
        callbacks=[ckpt_callback, early_stop, lr_monitor, profile_callback],
        log_every_n_steps=cfg.log_every_n_steps,
    )

    trainer.fit(model, datamodule=dm)

    best_ckpt = ckpt_callback.best_model_path
    best_score = float(ckpt_callback.best_model_score.item()) if ckpt_callback.best_model_score is not None else None
    if best_ckpt:
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = model.tokenizer

    export_start_time = time.perf_counter()
    token_size_splits = {
        "train": export_split_tokens(
            "train",
            make_export_loader(dm.train_set, cfg, shuffle=False),
            tokenizer,
            output_dir,
            device,
        ),
        "val": export_split_tokens(
            "val",
            make_export_loader(dm.val_set, cfg, shuffle=False),
            tokenizer,
            output_dir,
            device,
        ),
        "test": export_split_tokens(
            "test",
            make_export_loader(dm.test_set, cfg, shuffle=False),
            tokenizer,
            output_dir,
            device,
        ),
    }

    token_size_splits_pre_kmeans = None
    kmeans_ipc_summary = None
    if cfg.ipc > 0:
        token_size_splits_pre_kmeans = json.loads(json.dumps(token_size_splits))
        kmeans_ipc_summary = _compress_train_tokens_with_classwise_kmeans(
            output_dir=output_dir,
            ipc=cfg.ipc,
            seed=cfg.seed,
            codebook_size=tokenizer.codebook_size,
        )
        token_size_splits["train"] = kmeans_ipc_summary["train_size_stats"]

    token_export_time_sec = float(time.perf_counter() - export_start_time)
    total_pipeline_time_sec = float(time.perf_counter() - pipeline_start_time)

    profile_summary = profile_callback.summary()

    tokenizer_state = {
        "type": "pretrained_diffusers_vqvae_rvq",
        "vqvae_model": cfg.vqvae_model,
        "vqvae_subfolder": cfg.vqvae_subfolder,
        "vqvae_revision": cfg.vqvae_revision,
        "rvq_stages": cfg.rvq_stages,
        "ipc": cfg.ipc,
        "codebook_size": tokenizer.codebook_size,
        "code_dim": tokenizer.code_dim,
        "state_dict": {k: v.detach().cpu() for k, v in tokenizer.state_dict().items()},
    }
    torch.save(tokenizer_state, output_dir / "vq_tokenizer.pt")

    codebook = tokenizer.codebook_weight().detach().cpu()
    torch.save(codebook, output_dir / "vqvae_codebook.pt")
    np.save(output_dir / "vqvae_codebook.npy", codebook.numpy())

    metadata = {
        "dataset": cfg.dataset,
        "task_type": dm.task_type,
        "num_targets": dm.num_targets,
        "label_names": dm.label_names,
        "tokenizer_type": "pretrained_diffusers_vqvae_rvq",
        "pretrained_vqvae": {
            "model": cfg.vqvae_model,
            "subfolder": cfg.vqvae_subfolder,
            "revision": cfg.vqvae_revision,
        },
        "export_token_layout": "[num_samples, rvq_stages, grid_h, grid_w]",
        "codebook_size": tokenizer.codebook_size,
        "code_dim": tokenizer.code_dim,
        "rvq_stages": cfg.rvq_stages,
        "ipc": cfg.ipc,
        "training_loss_terms": [
            "train_loss",
            "train_recon_loss",
            "train_commit_loss",
            "train_cls_loss",
            "train_diversity_loss",
        ],
        "cls_weight": cfg.cls_weight,
        "cls_weight_end": cfg.cls_weight_end,
        "diversity_weight": cfg.diversity_weight,
        "quant_temperature": cfg.quant_temperature,
        "warmup_ratio": cfg.warmup_ratio,
        "min_lr": cfg.min_lr,
        "epochs": cfg.epochs,
        "blur": cfg.blur,
        "image_size": cfg.image_size,
        "encode_batch_size": cfg.encode_batch_size,
        "quantize_chunk_size": cfg.quantize_chunk_size,
        "best_checkpoint": best_ckpt,
        "best_score": best_score,
        "tensorboard_log_dir": tb_logger.log_dir,
    }

    total_raw_bytes = sum(v["raw_total_bytes"] for v in token_size_splits.values())
    total_npz_file_bytes = sum(v["npz_file_bytes"] for v in token_size_splits.values())
    metadata["token_size_stats"] = {
        "splits": token_size_splits,
        "total": {
            "raw_total_bytes": int(total_raw_bytes),
            "raw_total_human": format_bytes(int(total_raw_bytes)),
            "npz_file_bytes": int(total_npz_file_bytes),
            "npz_file_human": format_bytes(int(total_npz_file_bytes)),
        },
    }

    if token_size_splits_pre_kmeans is not None:
        pre_raw_total_bytes = sum(v["raw_total_bytes"] for v in token_size_splits_pre_kmeans.values())
        pre_npz_total_bytes = sum(v["npz_file_bytes"] for v in token_size_splits_pre_kmeans.values())
        metadata["token_size_stats_before_kmeans"] = {
            "splits": token_size_splits_pre_kmeans,
            "total": {
                "raw_total_bytes": int(pre_raw_total_bytes),
                "raw_total_human": format_bytes(int(pre_raw_total_bytes)),
                "npz_file_bytes": int(pre_npz_total_bytes),
                "npz_file_human": format_bytes(int(pre_npz_total_bytes)),
            },
        }

    if kmeans_ipc_summary is not None:
        metadata["kmeans_ipc"] = kmeans_ipc_summary

    metadata["distill_profile"] = {
        **profile_summary,
        "token_export_time_sec": token_export_time_sec,
        "total_pipeline_time_sec": total_pipeline_time_sec,
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"saved model + metadata to {output_dir}")
    print(f"tensorboard logs at {tb_logger.log_dir}")
    total_stats = metadata["token_size_stats"]["total"]
    distill_stats = metadata["distill_profile"]
    print(
        f"{cfg.dataset} token_size(raw={total_stats['raw_total_human']}, npz={total_stats['npz_file_human']}) "
        f"distill_time={distill_stats['total_distill_time_sec']:.2f}s "
        f"peak_allocated={distill_stats.get('peak_allocated_mb')}MB "
        f"peak_reserved={distill_stats.get('peak_reserved_mb')}MB"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill PathMNIST/DermaMNIST images into discrete tokens with pretrained VQ-VAE."
    )
    parser.add_argument("--dataset", type=str, default="pathmnist", choices=SUPPORTED_DATASET_CHOICES)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./artifacts/pathmnist_tokens")
    parser.add_argument("--vqvae-model", type=str, default="CompVis/ldm-celebahq-256")
    parser.add_argument("--vqvae-subfolder", type=str, default="vqvae")
    parser.add_argument("--vqvae-revision", type=str, default=None)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--quantize-chunk-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rvq-stages", type=int, default=2)
    parser.add_argument("--cls-weight", type=float, default=0.4)
    parser.add_argument("--cls-weight-end", type=float, default=0.2)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--quant-temperature", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-blur", action="store_true")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ipc", type=int, default=0, help=(
            "If > 0, run class-wise kmeans on train tokens after distillation. "
            "Number of clusters per class equals ipc."
        ),
    )
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = check_dataset_name(args.dataset)
    cfg = DistillConfig(
        dataset=dataset,
        data_root=args.data_root,
        output_dir=args.output_dir,
        vqvae_model=args.vqvae_model,
        vqvae_subfolder=args.vqvae_subfolder,
        vqvae_revision=args.vqvae_revision,
        encode_batch_size=args.encode_batch_size,
        quantize_chunk_size=args.quantize_chunk_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        rvq_stages=args.rvq_stages,
        cls_weight=args.cls_weight,
        cls_weight_end=args.cls_weight_end,
        diversity_weight=args.diversity_weight,
        quant_temperature=args.quant_temperature,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_lr,
        num_workers=args.num_workers,
        blur=not args.no_blur,
        image_size=args.image_size,
        seed=args.seed,
        ipc=args.ipc,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
    )

    set_seed(cfg.seed)
    train_tokenizer(cfg)


if __name__ == "__main__":
    main()
