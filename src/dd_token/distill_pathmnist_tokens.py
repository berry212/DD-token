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
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from medmnist import DermaMNIST, PathMNIST
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


HF_SKIN_LESIONS_REPO_ID = "ahmed-ai/skin-lesions-classification-dataset"
_SKIN_LESIONS_ALIASES = {
    HF_SKIN_LESIONS_REPO_ID,
    "skin-lesions",
    "skin_lesions",
    "skin-lesions-classification",
}
_DERMAMNIST_ALIASES = {
    "dermamnist",
    "derma",
    "dermamnist+",
}
SUPPORTED_DATASET_CHOICES = ["pathmnist", "dermamnist", "skin-lesions", HF_SKIN_LESIONS_REPO_ID]


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


def _import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for Hugging Face parquet datasets. Install it with `uv add pyarrow`."
        ) from exc
    return pq


def _import_diffusers_vqmodel():
    try:
        from diffusers import VQModel
    except ImportError as exc:
        raise ImportError(
            "diffusers is required for pretrained VQ-VAE loading. Install it with `uv add diffusers[torch]`."
        ) from exc
    return VQModel


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

        VQModel = _import_diffusers_vqmodel()
        load_kwargs: dict[str, Any] = {}
        if subfolder is not None and len(subfolder.strip()) > 0:
            load_kwargs["subfolder"] = subfolder
        if revision is not None and len(str(revision).strip()) > 0:
            load_kwargs["revision"] = revision
        if cache_dir is not None:
            load_kwargs["cache_dir"] = cache_dir

        self.vqvae = VQModel.from_pretrained(model_name, **load_kwargs)
        _ = self._codebook_weight()

        self.model_name = model_name
        self.model_subfolder = subfolder
        self.model_revision = revision
        self.rvq_stages = int(rvq_stages)
        self.beta = float(beta)
        self.temperature = float(temperature)
        self.quantize_chunk_size = int(max(1, quantize_chunk_size))

    def _codebook_weight(self) -> torch.Tensor:
        quantize = getattr(self.vqvae, "quantize", None)
        if quantize is None:
            raise RuntimeError("Loaded VQ-VAE does not expose a quantize module.")

        embedding = getattr(quantize, "embedding", None)
        if embedding is not None and hasattr(embedding, "weight"):
            return embedding.weight

        embed = getattr(quantize, "embed", None)
        if isinstance(embed, torch.Tensor):
            if embed.ndim != 2:
                raise RuntimeError(f"Unsupported quantizer embed shape: {tuple(embed.shape)}")
            # Some checkpoints store [dim, n_embed] instead of [n_embed, dim].
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
        if flat_latents.device != codebook.device:
            flat_latents = flat_latents.to(codebook.device)
        if flat_latents.dtype != codebook.dtype:
            flat_latents = flat_latents.to(codebook.dtype)

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

            temperature = max(self.temperature, 1e-6)
            soft_assign = F.softmax(-distances / temperature, dim=1)
            prob_sums = prob_sums + soft_assign.sum(dim=0)
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


def make_pathmnist_datasets(cfg: DistillConfig):
    return make_medmnist_multiclass_datasets(
        cfg,
        dataset_key="pathmnist",
        dataset_cls=PathMNIST,
        require_native_size=False,
    )


def _build_medmnist_split(
    dataset_cls,
    cfg: DistillConfig,
    split: str,
    transform,
    require_native_size: bool = False,
):
    base_kwargs = {
        "root": cfg.data_root,
        "split": split,
        "transform": transform,
        "download": True,
    }
    try:
        return dataset_cls(size=cfg.image_size, **base_kwargs)
    except TypeError as exc:
        if require_native_size:
            raise RuntimeError(
                "DermaMNIST official high-resolution split requires medmnist with `size` support. "
                "Please upgrade medmnist (e.g., `uv add -U medmnist`) and retry."
            ) from exc
        return dataset_cls(**base_kwargs)


def make_medmnist_multiclass_datasets(
    cfg: DistillConfig,
    dataset_key: str,
    dataset_cls,
    require_native_size: bool,
):
    Path(cfg.data_root).mkdir(parents=True, exist_ok=True)
    # DermaMNIST official 224 should come from MedMNIST+ native split, not transform resize.
    transform = build_transforms(cfg.blur, cfg.image_size, enable_resize=not require_native_size)
    train_set = _build_medmnist_split(
        dataset_cls,
        cfg,
        split="train",
        transform=transform,
        require_native_size=require_native_size,
    )
    val_set = _build_medmnist_split(
        dataset_cls,
        cfg,
        split="val",
        transform=transform,
        require_native_size=require_native_size,
    )
    test_set = _build_medmnist_split(
        dataset_cls,
        cfg,
        split="test",
        transform=transform,
        require_native_size=require_native_size,
    )

    num_targets = len(medmnist.INFO[dataset_key]["label"])
    label_names = [medmnist.INFO[dataset_key]["label"][str(i)] for i in range(num_targets)]
    metadata = {"task_type": "multiclass", "num_targets": num_targets, "label_names": label_names}
    return train_set, val_set, test_set, metadata


def make_dermamnist_datasets(cfg: DistillConfig):
    return make_medmnist_multiclass_datasets(
        cfg,
        dataset_key="dermamnist",
        dataset_cls=DermaMNIST,
        require_native_size=True,
    )


def _dataset_name_kind(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower()
    if normalized == "pathmnist":
        return "pathmnist"
    if normalized in _DERMAMNIST_ALIASES:
        return "dermamnist"
    if normalized in {name.lower() for name in _SKIN_LESIONS_ALIASES}:
        return "hf_skin_lesions"
    raise ValueError(
        f"Unsupported dataset '{dataset_name}'. Supported values: {SUPPORTED_DATASET_CHOICES}."
    )


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
            pairs = []
            for key, value in names.items():
                try:
                    idx = int(key)
                except (TypeError, ValueError):
                    continue
                pairs.append((idx, str(value)))
            if pairs:
                pairs.sort(key=lambda x: x[0])
                return [name for _, name in pairs]
    return []


def _infer_num_targets_from_parquet(parquet_paths: list[Path], label_column: str = "label") -> int:
    max_label = -1
    pq = _import_pyarrow_parquet()
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(str(parquet_path))
        for row_group_idx in range(parquet_file.num_row_groups):
            label_table = parquet_file.read_row_group(row_group_idx, columns=[label_column])
            labels = label_table.column(label_column).to_pylist()
            for label in labels:
                max_label = max(max_label, int(label))
    if max_label < 0:
        raise ValueError("Could not infer target count from parquet label column.")
    return max_label + 1


def make_hf_skin_lesions_datasets(cfg: DistillConfig):
    Path(cfg.data_root).mkdir(parents=True, exist_ok=True)
    transform = build_transforms(cfg.blur, cfg.image_size)

    snapshot_dir = Path(
        snapshot_download(
            repo_id=HF_SKIN_LESIONS_REPO_ID,
            repo_type="dataset",
            allow_patterns=["README.md", "data/*.parquet"],
            cache_dir=str(Path(cfg.data_root) / "hf_cache"),
        )
    )

    train_files = _collect_parquet_split_files(snapshot_dir, "train")
    val_files = _collect_parquet_split_files(snapshot_dir, "validation")
    test_files = _collect_parquet_split_files(snapshot_dir, "test")

    if not train_files or not val_files or not test_files:
        raise ValueError(
            "Failed to find complete train/validation/test parquet splits in the downloaded Hugging Face dataset."
        )

    train_set = HFParquetImageDataset(train_files, transform=transform)
    val_set = HFParquetImageDataset(val_files, transform=transform)
    test_set = HFParquetImageDataset(test_files, transform=transform)

    label_names = _extract_hf_label_names(HF_SKIN_LESIONS_REPO_ID)
    if label_names:
        num_targets = len(label_names)
    else:
        num_targets = _infer_num_targets_from_parquet(train_files + val_files + test_files)
        label_names = [str(i) for i in range(num_targets)]

    metadata = {
        "task_type": "multiclass",
        "num_targets": num_targets,
        "label_names": label_names,
        "hf_repo_id": HF_SKIN_LESIONS_REPO_ID,
        "hf_snapshot_dir": str(snapshot_dir),
        "split_files": {
            "train": [str(path) for path in train_files],
            "validation": [str(path) for path in val_files],
            "test": [str(path) for path in test_files],
        },
    }
    return train_set, val_set, test_set, metadata


def make_datasets(cfg: DistillConfig):
    kind = _dataset_name_kind(cfg.dataset)
    if kind == "pathmnist":
        return make_pathmnist_datasets(cfg)
    if kind == "dermamnist":
        return make_dermamnist_datasets(cfg)
    if kind == "hf_skin_lesions":
        return make_hf_skin_lesions_datasets(cfg)
    raise ValueError(f"Unsupported dataset kind: {kind}")


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
        train_set, val_set, test_set, meta = make_datasets(self.cfg)
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

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
    def __init__(self, cfg: DistillConfig, task_type: str, num_targets: int):
        super().__init__()
        self.cfg = cfg
        self.task_type = task_type

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

        self.aux_head = nn.Linear(self.tokenizer.code_dim, num_targets)

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
            "task_type": task_type,
            "num_targets": num_targets,
            "tokenizer_type": "pretrained_diffusers_vqvae_rvq",
        })

    def _current_cls_weight(self) -> float:
        if self.cfg.epochs <= 1:
            return self.cfg.cls_weight_end
        ratio = self.current_epoch / float(max(1, self.cfg.epochs - 1))
        return self.cfg.cls_weight + ratio * (self.cfg.cls_weight_end - self.cfg.cls_weight)

    def _shared_forward(self, images: torch.Tensor, labels: torch.Tensor):
        _, z_q, _, recon_loss, commit_loss, diversity_loss = self.tokenizer(images)

        image_repr = z_q.mean(dim=(2, 3))
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
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for pretrained VQ-VAE training.")

        optimizer = torch.optim.AdamW(trainable_params, lr=self.cfg.lr)

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
    tokenizer: Any,
    output_dir: Path,
    device: torch.device,
):
    tokenizer.eval()
    all_pseudo_tokens = []
    all_labels = []

    for images, labels in _iter_with_progress(loader, split_name):
        images = images.to(device, non_blocking=True)
        pseudo_tokens = tokenizer.encode_tokens(images)
        all_pseudo_tokens.append(pseudo_tokens.cpu().numpy().astype(np.int32))

        labels_np = labels.cpu().numpy()
        if labels_np.ndim == 1:
            all_labels.append(labels_np.astype(np.int64))
        elif labels_np.ndim == 2 and labels_np.shape[1] == 1:
            all_labels.append(labels_np.reshape(-1).astype(np.int64))
        else:
            all_labels.append(labels_np.astype(np.float32))

    tokens_np = np.concatenate(all_pseudo_tokens, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    output_path = output_dir / f"{split_name}_tokens.npz"
    np.savez_compressed(output_path, tokens=tokens_np, labels=labels_np)
    print(f"saved {split_name}: {output_path} | pseudo_tokens shape={tokens_np.shape} | labels shape={labels_np.shape}")

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
    model = LitPretrainedVQDistiller(
        cfg=cfg,
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
    token_export_time_sec = float(time.perf_counter() - export_start_time)
    total_pipeline_time_sec = float(time.perf_counter() - pipeline_start_time)

    profile_summary = profile_callback.summary()

    tokenizer_state = {
        "type": "pretrained_diffusers_vqvae_rvq",
        "vqvae_model": cfg.vqvae_model,
        "vqvae_subfolder": cfg.vqvae_subfolder,
        "vqvae_revision": cfg.vqvae_revision,
        "rvq_stages": cfg.rvq_stages,
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
        description="Distill image datasets into discrete tokens by training a pretrained-initialized VQ-VAE tokenizer."
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

    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DistillConfig(
        dataset=args.dataset,
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
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps,
    )

    set_seed(cfg.seed)
    train_tokenizer(cfg)


if __name__ == "__main__":
    main()
