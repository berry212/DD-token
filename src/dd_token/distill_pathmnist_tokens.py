import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightning as L
import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from medmnist import PathMNIST
from torch.utils.data import DataLoader
from torchvision import transforms


@dataclass
class DistillConfig:
    data_root: str
    output_dir: str
    dataset: str = "pathmnist"
    epochs: int = 12
    batch_size: int = 256
    lr: float = 3e-4
    patch_size: int = 4
    overlap: float = 0.2
    codebook_size: int = 2048
    code_dim: int = 256
    hidden_dim: int = 384
    encoder_layers: int = 3
    decoder_layers: int = 2
    attention_heads: int = 8
    ff_mult: int = 4
    rvq_stages: int = 2
    num_workers: int = 4
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
    image_size: int = 256
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
    log_every_n_steps: int = 20


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


def build_sincos_position(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(1, length, dim, device=device)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        num_quantizers: int,
        codebook_size: int,
        code_dim: int,
        beta: float = 0.25,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.beta = beta
        self.temperature = temperature
        self.codebooks = nn.ModuleList([nn.Embedding(codebook_size, code_dim) for _ in range(num_quantizers)])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for cb in self.codebooks:
            nn.init.uniform_(cb.weight, -1.0, 1.0)

    def _distances(self, x: torch.Tensor, codebook: nn.Embedding) -> torch.Tensor:
        x_sq = (x**2).sum(dim=1, keepdim=True)
        c_sq = (codebook.weight**2).sum(dim=1)
        return x_sq + c_sq - 2 * x @ codebook.weight.t()

    def forward(self, z_e: torch.Tensor):
        residual = z_e
        z_q_total = torch.zeros_like(z_e)
        all_indices = []
        commit_terms = []
        diversity_terms = []

        for codebook in self.codebooks:
            distances = self._distances(residual, codebook)
            indices = torch.argmin(distances, dim=1)

            z_q = codebook(indices)
            z_q_st = residual + (z_q - residual).detach()
            z_q_total = z_q_total + z_q_st

            commit = F.mse_loss(residual, z_q.detach()) + self.beta * F.mse_loss(z_q, residual.detach())
            commit_terms.append(commit)

            soft_assign = F.softmax(-distances / self.temperature, dim=1)
            avg_probs = soft_assign.mean(dim=0)
            entropy = -(avg_probs * torch.log(avg_probs + 1e-8)).sum()
            max_entropy = math.log(codebook.num_embeddings)
            diversity = 1.0 - entropy / max_entropy
            diversity_terms.append(diversity)

            all_indices.append(indices)
            residual = residual - z_q.detach()

        indices_stacked = torch.stack(all_indices, dim=1)
        commit_loss = torch.stack(commit_terms).mean()
        diversity_loss = torch.stack(diversity_terms).mean()
        return z_q_total, indices_stacked, commit_loss, diversity_loss


class ContextualPatchTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        hidden_dim: int,
        code_dim: int,
        codebook_size: int,
        commit_weight: float,
        quant_temperature: float,
        encoder_layers: int,
        decoder_layers: int,
        attention_heads: int,
        ff_mult: int,
        rvq_stages: int,
    ):
        super().__init__()
        patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size
        self.rvq_stages = rvq_stages

        self.patch_embed = nn.Linear(patch_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)
        self.latent_proj = nn.Linear(hidden_dim, code_dim)

        self.quantizer = ResidualVectorQuantizer(
            num_quantizers=rvq_stages,
            codebook_size=codebook_size,
            code_dim=code_dim,
            beta=commit_weight,
            temperature=quant_temperature,
        )

        self.decode_in = nn.Linear(code_dim, hidden_dim)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_layers)
        self.decode_out = nn.Linear(hidden_dim, patch_dim)

    def encode_latents(self, patches_seq: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(patches_seq)
        x = x + build_sincos_position(x.size(1), x.size(2), x.device)
        x = self.encoder(x)
        z_e = self.latent_proj(x)
        return z_e

    def forward(self, patches_seq: torch.Tensor):
        bsz, seq_len, code_in = patches_seq.shape
        _ = code_in

        z_e_seq = self.encode_latents(patches_seq)
        z_e_flat = z_e_seq.reshape(-1, z_e_seq.size(-1))

        z_q_flat, indices_flat, commit_loss, diversity_loss = self.quantizer(z_e_flat)
        z_q_seq = z_q_flat.view(bsz, seq_len, -1)
        indices = indices_flat.view(bsz, seq_len, self.rvq_stages)

        dec = self.decode_in(z_q_seq)
        dec = dec + build_sincos_position(dec.size(1), dec.size(2), dec.device)
        dec = self.decoder(dec)
        recon = torch.tanh(self.decode_out(dec))
        recon_loss = F.mse_loss(recon, patches_seq)

        return recon, z_q_seq, indices, recon_loss, commit_loss, diversity_loss

    @torch.no_grad()
    def encode_tokens(self, images: torch.Tensor, overlap: float):
        patches_seq, _ = extract_patches(images, self.patch_size, overlap)
        z_e_seq = self.encode_latents(patches_seq)
        bsz, seq_len, _ = z_e_seq.shape
        z_e_flat = z_e_seq.reshape(-1, z_e_seq.size(-1))

        _, indices_flat, _, _ = self.quantizer(z_e_flat)
        indices = indices_flat.view(bsz, seq_len, self.rvq_stages)
        return indices.reshape(bsz, -1)


def extract_patches(images: torch.Tensor, patch_size: int, overlap: float):
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")

    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    unfolded = F.unfold(images, kernel_size=patch_size, stride=stride)
    patches = unfolded.transpose(1, 2).contiguous()
    return patches, unfolded.size(-1)


def build_transforms(blur: bool, image_size: int):
    transforms_list = []
    if image_size > 0:
        transforms_list.append(transforms.Resize((image_size, image_size)))
    transforms_list.append(transforms.ToTensor())
    if blur:
        transforms_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)))
    transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms.Compose(transforms_list)


def make_pathmnist_datasets(cfg: DistillConfig):
    Path(cfg.data_root).mkdir(parents=True, exist_ok=True)
    transform = build_transforms(cfg.blur, cfg.image_size)
    train_set = PathMNIST(root=cfg.data_root, split="train", transform=transform, download=True)
    val_set = PathMNIST(root=cfg.data_root, split="val", transform=transform, download=True)
    test_set = PathMNIST(root=cfg.data_root, split="test", transform=transform, download=True)

    num_targets = len(medmnist.INFO["pathmnist"]["label"])
    label_names = [medmnist.INFO["pathmnist"]["label"][str(i)] for i in range(num_targets)]
    metadata = {"task_type": "multiclass", "num_targets": num_targets, "label_names": label_names}
    return train_set, val_set, test_set, metadata


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
        self.in_channels = 3

    def setup(self, stage: str | None = None):
        train_set, val_set, test_set, meta = make_pathmnist_datasets(self.cfg)
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.task_type = meta["task_type"]
        self.num_targets = meta["num_targets"]
        self.label_names = meta["label_names"]

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


class LitVQDistiller(L.LightningModule):
    def __init__(self, cfg: DistillConfig, in_channels: int, task_type: str, num_targets: int):
        super().__init__()
        self.cfg = cfg
        self.task_type = task_type

        self.tokenizer = ContextualPatchTokenizer(
            in_channels=in_channels,
            patch_size=cfg.patch_size,
            hidden_dim=cfg.hidden_dim,
            code_dim=cfg.code_dim,
            codebook_size=cfg.codebook_size,
            commit_weight=cfg.commit_weight,
            quant_temperature=cfg.quant_temperature,
            encoder_layers=cfg.encoder_layers,
            decoder_layers=cfg.decoder_layers,
            attention_heads=cfg.attention_heads,
            ff_mult=cfg.ff_mult,
            rvq_stages=cfg.rvq_stages,
        )
        self.aux_head = nn.Linear(cfg.code_dim, num_targets)

        if task_type == "multiclass":
            self.cls_criterion = nn.CrossEntropyLoss()
            self.monitor_metric = "val_acc"
        else:
            self.cls_criterion = nn.BCEWithLogitsLoss()
            self.monitor_metric = "val_micro_f1"

        self.val_preds: list[torch.Tensor] = []
        self.val_targets: list[torch.Tensor] = []

        self.save_hyperparameters(asdict(cfg))
        self.save_hyperparameters({
            "in_channels": in_channels,
            "task_type": task_type,
            "num_targets": num_targets,
        })

    def _current_cls_weight(self) -> float:
        if self.cfg.epochs <= 1:
            return self.cfg.cls_weight_end
        ratio = self.current_epoch / float(max(1, self.cfg.epochs - 1))
        return self.cfg.cls_weight + ratio * (self.cfg.cls_weight_end - self.cfg.cls_weight)

    def _shared_forward(self, images: torch.Tensor, labels: torch.Tensor):
        patches_seq, patch_count = extract_patches(images, self.cfg.patch_size, self.cfg.overlap)
        recon, z_q, _, recon_loss, commit_loss, diversity_loss = self.tokenizer(patches_seq)
        _ = recon
        _ = patch_count

        image_repr = z_q.mean(dim=1)
        cls_logits = self.aux_head(image_repr)

        if self.task_type == "multiclass":
            cls_loss = self.cls_criterion(cls_logits, labels.view(-1).long())
        else:
            cls_loss = self.cls_criterion(cls_logits, labels.float())

        cls_weight_now = self._current_cls_weight()
        total_loss = (
            self.cfg.recon_weight * recon_loss
            + commit_loss
            + cls_weight_now * cls_loss
            + self.cfg.diversity_weight * diversity_loss
        )
        return cls_logits, total_loss, recon_loss, commit_loss, cls_loss, diversity_loss, cls_weight_now

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits, total_loss, recon_loss, commit_loss, cls_loss, diversity_loss, cls_weight_now = self._shared_forward(images, labels)
        _ = logits

        self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train_recon_loss", recon_loss, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train_commit_loss", commit_loss, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train_cls_loss", cls_loss, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train_diversity_loss", diversity_loss, on_step=False, on_epoch=True, batch_size=images.size(0))
        self.log("train_cls_weight", cls_weight_now, on_step=False, on_epoch=True, batch_size=images.size(0))
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits, total_loss, _, _, _, _, _ = self._shared_forward(images, labels)

        self.log("val_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=images.size(0))

        if self.task_type == "multiclass":
            preds = torch.argmax(logits, dim=1)
            targets = labels.view(-1).long()
        else:
            preds = (torch.sigmoid(logits) >= 0.5).long()
            targets = labels.long()

        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(targets.detach().cpu())

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return

        y_pred = torch.cat(self.val_preds)
        y_true = torch.cat(self.val_targets)

        if self.task_type == "multiclass":
            metric = float((y_pred == y_true).float().mean().item())
            self.log("val_acc", metric, prog_bar=True, on_step=False, on_epoch=True)
            self.print(f"epoch={self.current_epoch + 1:02d} val_acc={metric:.4f}")
        else:
            tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
            fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
            fn = torch.sum((y_pred == 0) & (y_true == 1)).item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metric = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            self.log("val_micro_f1", metric, prog_bar=True, on_step=False, on_epoch=True)
            self.print(f"epoch={self.current_epoch + 1:02d} val_micro_f1={metric:.4f}")

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

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.fit_start_time = time.perf_counter()
        self.epoch_distill_time_sec.clear()
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

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
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
    tokenizer: ContextualPatchTokenizer,
    overlap: float,
    output_dir: Path,
    device: torch.device,
):
    tokenizer.eval()
    all_tokens = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        tokens = tokenizer.encode_tokens(images, overlap=overlap)
        all_tokens.append(tokens.cpu().numpy().astype(np.int32))

        labels_np = labels.cpu().numpy()
        if labels_np.ndim == 1:
            all_labels.append(labels_np.astype(np.int64))
        elif labels_np.ndim == 2 and labels_np.shape[1] == 1:
            all_labels.append(labels_np.reshape(-1).astype(np.int64))
        else:
            all_labels.append(labels_np.astype(np.float32))

    tokens_np = np.concatenate(all_tokens, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    output_path = output_dir / f"{split_name}_tokens.npz"
    np.savez_compressed(output_path, tokens=tokens_np, labels=labels_np)
    print(f"saved {split_name}: {output_path} | tokens shape={tokens_np.shape} | labels shape={labels_np.shape}")

    raw_tokens_bytes = int(tokens_np.nbytes)
    raw_labels_bytes = int(labels_np.nbytes)
    raw_total_bytes = raw_tokens_bytes + raw_labels_bytes
    file_bytes = int(output_path.stat().st_size)
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


def train_tokenizer(cfg: DistillConfig):
    pipeline_start_time = time.perf_counter()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dm = DistillDataModule(cfg)
    dm.setup()

    model = LitVQDistiller(
        cfg=cfg,
        in_channels=dm.in_channels,
        task_type=dm.task_type,
        num_targets=dm.num_targets,
    )

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
    if best_ckpt:
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = model.tokenizer

    export_start_time = time.perf_counter()
    token_size_splits = {
        "train": export_split_tokens("train", dm.train_dataloader(), tokenizer, cfg.overlap, output_dir, device),
        "val": export_split_tokens("val", dm.val_dataloader(), tokenizer, cfg.overlap, output_dir, device),
        "test": export_split_tokens("test", dm.test_dataloader(), tokenizer, cfg.overlap, output_dir, device),
    }
    token_export_time_sec = float(time.perf_counter() - export_start_time)
    total_pipeline_time_sec = float(time.perf_counter() - pipeline_start_time)

    profile_summary = profile_callback.summary()

    torch.save(tokenizer.state_dict(), output_dir / "vq_tokenizer.pt")

    metadata = {
        "dataset": cfg.dataset,
        "task_type": dm.task_type,
        "num_targets": dm.num_targets,
        "label_names": dm.label_names,
        "patch_size": cfg.patch_size,
        "overlap": cfg.overlap,
        "codebook_size": cfg.codebook_size,
        "code_dim": cfg.code_dim,
        "hidden_dim": cfg.hidden_dim,
        "encoder_layers": cfg.encoder_layers,
        "decoder_layers": cfg.decoder_layers,
        "attention_heads": cfg.attention_heads,
        "ff_mult": cfg.ff_mult,
        "rvq_stages": cfg.rvq_stages,
        "cls_weight": cfg.cls_weight,
        "cls_weight_end": cfg.cls_weight_end,
        "diversity_weight": cfg.diversity_weight,
        "quant_temperature": cfg.quant_temperature,
        "warmup_ratio": cfg.warmup_ratio,
        "min_lr": cfg.min_lr,
        "epochs": cfg.epochs,
        "blur": cfg.blur,
        "image_size": cfg.image_size,
        "best_checkpoint": best_ckpt,
        "best_score": float(ckpt_callback.best_model_score.item()) if ckpt_callback.best_model_score is not None else None,
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
        f"pathmnist token_size(raw={total_stats['raw_total_human']}, npz={total_stats['npz_file_human']}) "
        f"distill_time={distill_stats['total_distill_time_sec']:.2f}s "
        f"peak_allocated={distill_stats.get('peak_allocated_mb')}MB "
        f"peak_reserved={distill_stats.get('peak_reserved_mb')}MB"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Distill image datasets into discrete tokens with a Lightning VQ tokenizer.")
    parser.add_argument("--dataset", type=str, default="pathmnist", choices=["pathmnist"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./artifacts/pathmnist_tokens")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--codebook-size", type=int, default=2048)
    parser.add_argument("--code-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--rvq-stages", type=int, default=2)
    parser.add_argument("--cls-weight", type=float, default=0.4)
    parser.add_argument("--cls-weight-end", type=float, default=0.2)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--quant-temperature", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-blur", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DistillConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=args.patch_size,
        overlap=args.overlap,
        codebook_size=args.codebook_size,
        code_dim=args.code_dim,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        attention_heads=args.attention_heads,
        ff_mult=args.ff_mult,
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
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
    )

    set_seed(cfg.seed)
    train_tokenizer(cfg)


if __name__ == "__main__":
    main()
