import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from medmnist import PathMNIST
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

NIH_14_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


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
    num_workers: int = 4
    recon_weight: float = 1.0
    commit_weight: float = 0.25
    cls_weight: float = 0.4
    blur: bool = True
    seed: int = 42
    image_size: int = 256
    hf_dataset_name: str = "alkzar90/NIH-Chest-X-ray-dataset"
    hf_config_name: str = "image-classification"
    hf_trust_remote_code: bool = True
    val_ratio: float = 0.1


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.beta = beta
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.codebook.weight, -1.0, 1.0)

    def forward(self, z_e: torch.Tensor):
        z_e_sq = (z_e**2).sum(dim=1, keepdim=True)
        code_sq = (self.codebook.weight**2).sum(dim=1)
        distances = z_e_sq + code_sq - 2 * z_e @ self.codebook.weight.t()

        indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(indices)
        z_q_st = z_e + (z_q - z_e).detach()

        commit_loss = F.mse_loss(z_e, z_q.detach()) + self.beta * F.mse_loss(z_q, z_e.detach())
        return z_q_st, indices, commit_loss


class PatchVQTokenizer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        hidden_dim: int,
        code_dim: int,
        codebook_size: int,
        commit_weight: float,
    ):
        super().__init__()
        patch_dim = in_channels * patch_size * patch_size
        self.patch_size = patch_size
        self.encoder = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, code_dim),
        )
        self.quantizer = VectorQuantizer(codebook_size=codebook_size, code_dim=code_dim, beta=commit_weight)
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, patch_dim),
            nn.Tanh(),
        )

    def forward(self, patches: torch.Tensor):
        z_e = self.encoder(patches)
        z_q_st, indices, commit_loss = self.quantizer(z_e)
        recon = self.decoder(z_q_st)
        recon_loss = F.mse_loss(recon, patches)
        return recon, z_q_st, indices, recon_loss, commit_loss

    @torch.no_grad()
    def encode_tokens(self, images: torch.Tensor, overlap: float):
        patches, patch_count = extract_patches(images, self.patch_size, overlap)
        z_e = self.encoder(patches)

        z_e_sq = (z_e**2).sum(dim=1, keepdim=True)
        code_sq = (self.quantizer.codebook.weight**2).sum(dim=1)
        distances = z_e_sq + code_sq - 2 * z_e @ self.quantizer.codebook.weight.t()

        indices = torch.argmin(distances, dim=1)
        tokens = indices.view(images.size(0), patch_count)
        return tokens


class HFDatasetAdapter(Dataset):
    def __init__(self, hf_split, transform, label_names: list[str]):
        self.hf_split = hf_split
        self.transform = transform
        self.label_names = label_names
        self.label_to_idx = {name: idx for idx, name in enumerate(label_names)}

    def __len__(self) -> int:
        return len(self.hf_split)

    def _to_multihot(self, raw_labels) -> torch.Tensor:
        out = torch.zeros(len(self.label_names), dtype=torch.float32)
        if raw_labels is None:
            return out

        for item in raw_labels:
            if isinstance(item, str):
                if item == "No Finding":
                    continue
                if item in self.label_to_idx:
                    out[self.label_to_idx[item]] = 1.0
            else:
                idx = int(item)
                if idx <= 0:
                    continue
                mapped_idx = idx - 1
                if 0 <= mapped_idx < len(self.label_names):
                    out[mapped_idx] = 1.0
        return out

    def __getitem__(self, idx):
        row = self.hf_split[idx]
        image = row["image"]
        labels = self._to_multihot(row.get("labels", []))
        image = image.convert("RGB")
        image = self.transform(image)
        return image, labels


def extract_patches(images: torch.Tensor, patch_size: int, overlap: float):
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")

    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    unfolded = F.unfold(images, kernel_size=patch_size, stride=stride)
    patches = unfolded.transpose(1, 2).reshape(-1, unfolded.size(1))
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


def make_pathmnist_loaders(cfg: DistillConfig):
    Path(cfg.data_root).mkdir(parents=True, exist_ok=True)
    transform = build_transforms(cfg.blur, cfg.image_size)
    train_set = PathMNIST(root=cfg.data_root, split="train", transform=transform, download=True)
    val_set = PathMNIST(root=cfg.data_root, split="val", transform=transform, download=True)
    test_set = PathMNIST(root=cfg.data_root, split="test", transform=transform, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    eval_loader_kwargs = dict(batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    metadata = {
        "task_type": "multiclass",
        "num_targets": len(medmnist.INFO["pathmnist"]["label"]),
        "label_names": [medmnist.INFO["pathmnist"]["label"][str(i)] for i in range(len(medmnist.INFO["pathmnist"]["label"]))],
    }
    return train_loader, DataLoader(val_set, **eval_loader_kwargs), DataLoader(test_set, **eval_loader_kwargs), metadata


def make_nih_loaders(cfg: DistillConfig):
    transform = build_transforms(cfg.blur, cfg.image_size)
    hf_train = load_dataset(
        cfg.hf_dataset_name,
        cfg.hf_config_name,
        split="train",
        trust_remote_code=cfg.hf_trust_remote_code,
    )
    hf_test = load_dataset(
        cfg.hf_dataset_name,
        cfg.hf_config_name,
        split="test",
        trust_remote_code=cfg.hf_trust_remote_code,
    )

    split_train_val = hf_train.train_test_split(test_size=cfg.val_ratio, seed=cfg.seed)
    train_set = HFDatasetAdapter(split_train_val["train"], transform, NIH_14_LABELS)
    val_set = HFDatasetAdapter(split_train_val["test"], transform, NIH_14_LABELS)
    test_set = HFDatasetAdapter(hf_test, transform, NIH_14_LABELS)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    eval_loader_kwargs = dict(batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    metadata = {
        "task_type": "multilabel",
        "num_targets": len(NIH_14_LABELS),
        "label_names": NIH_14_LABELS,
    }
    return train_loader, DataLoader(val_set, **eval_loader_kwargs), DataLoader(test_set, **eval_loader_kwargs), metadata


def make_loaders(cfg: DistillConfig):
    if cfg.dataset == "pathmnist":
        return make_pathmnist_loaders(cfg)
    if cfg.dataset == "nih-chest-xray":
        return make_nih_loaders(cfg)
    raise ValueError(f"Unsupported dataset: {cfg.dataset}")


def train_tokenizer(cfg: DistillConfig, device: torch.device):
    train_loader, val_loader, test_loader, ds_meta = make_loaders(cfg)
    first_images, _ = next(iter(train_loader))
    in_channels = int(first_images.size(1))

    model = PatchVQTokenizer(
        in_channels=in_channels,
        patch_size=cfg.patch_size,
        hidden_dim=cfg.hidden_dim,
        code_dim=cfg.code_dim,
        codebook_size=cfg.codebook_size,
        commit_weight=cfg.commit_weight,
    ).to(device)

    num_targets = ds_meta["num_targets"]
    aux_head = nn.Linear(cfg.code_dim, num_targets).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(aux_head.parameters()), lr=cfg.lr)
    if ds_meta["task_type"] == "multiclass":
        cls_criterion = nn.CrossEntropyLoss()
    else:
        cls_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        steps = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            patches, patch_count = extract_patches(images, cfg.patch_size, cfg.overlap)
            recon, z_q, _, recon_loss, commit_loss = model(patches)
            _ = recon

            image_repr = z_q.view(images.size(0), patch_count, -1).mean(dim=1)
            cls_logits = aux_head(image_repr)

            if ds_meta["task_type"] == "multiclass":
                cls_loss = cls_criterion(cls_logits, labels.view(-1).long())
            else:
                cls_loss = cls_criterion(cls_logits, labels.float())

            loss = cfg.recon_weight * recon_loss + commit_loss + cfg.cls_weight * cls_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            steps += 1

        avg = total / max(1, steps)
        print(f"epoch={epoch + 1:02d} train_loss={avg:.6f}")

    return model, (train_loader, val_loader, test_loader), ds_meta


@torch.no_grad()
def export_split_tokens(
    split_name: str,
    loader: DataLoader,
    model: PatchVQTokenizer,
    overlap: float,
    output_dir: Path,
    device: torch.device,
):
    model.eval()
    all_tokens = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        tokens = model.encode_tokens(images, overlap=overlap)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Distill image datasets into discrete tokens with a VQ tokenizer.")
    parser.add_argument("--dataset", type=str, default="pathmnist", choices=["pathmnist", "nih-chest-xray"])
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
    parser.add_argument("--cls-weight", type=float, default=0.4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-blur", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hf-dataset-name", type=str, default="alkzar90/NIH-Chest-X-ray-dataset")
    parser.add_argument("--hf-config-name", type=str, default="image-classification")
    parser.add_argument("--hf-no-trust-remote-code", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.1)
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
        cls_weight=args.cls_weight,
        num_workers=args.num_workers,
        blur=not args.no_blur,
        image_size=args.image_size,
        seed=args.seed,
        hf_dataset_name=args.hf_dataset_name,
        hf_config_name=args.hf_config_name,
        hf_trust_remote_code=not args.hf_no_trust_remote_code,
        val_ratio=args.val_ratio,
    )

    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    model, loaders, ds_meta = train_tokenizer(cfg, device)
    train_loader, val_loader, test_loader = loaders

    export_split_tokens("train", train_loader, model, cfg.overlap, output_dir, device)
    export_split_tokens("val", val_loader, model, cfg.overlap, output_dir, device)
    export_split_tokens("test", test_loader, model, cfg.overlap, output_dir, device)

    torch.save(model.state_dict(), output_dir / "vq_tokenizer.pt")
    metadata = {
        "dataset": cfg.dataset,
        "task_type": ds_meta["task_type"],
        "num_targets": ds_meta["num_targets"],
        "label_names": ds_meta["label_names"],
        "patch_size": cfg.patch_size,
        "overlap": cfg.overlap,
        "codebook_size": cfg.codebook_size,
        "code_dim": cfg.code_dim,
        "hidden_dim": cfg.hidden_dim,
        "cls_weight": cfg.cls_weight,
        "epochs": cfg.epochs,
        "blur": cfg.blur,
        "image_size": cfg.image_size,
        "hf_dataset_name": cfg.hf_dataset_name if cfg.dataset == "nih-chest-xray" else None,
        "hf_config_name": cfg.hf_config_name if cfg.dataset == "nih-chest-xray" else None,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"saved model + metadata to {output_dir}")


if __name__ == "__main__":
    main()
