import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from medmnist import PathMNIST
from torch.utils.data import DataLoader
from torchvision import transforms


@dataclass
class DistillConfig:
    data_root: str
    output_dir: str
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
        # z_e: [N, D]
        z_e_sq = (z_e ** 2).sum(dim=1, keepdim=True)
        code_sq = (self.codebook.weight ** 2).sum(dim=1)
        distances = z_e_sq + code_sq - 2 * z_e @ self.codebook.weight.t()

        indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(indices)

        # Straight-through estimator
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
        # patches: [N, C*P*P]
        z_e = self.encoder(patches)
        z_q_st, indices, commit_loss = self.quantizer(z_e)
        recon = self.decoder(z_q_st)
        recon_loss = F.mse_loss(recon, patches)
        return recon, z_q_st, indices, recon_loss, commit_loss

    @torch.no_grad()
    def encode_tokens(self, images: torch.Tensor, overlap: float):
        patches, patch_count = extract_patches(images, self.patch_size, overlap)
        z_e = self.encoder(patches)

        z_e_sq = (z_e ** 2).sum(dim=1, keepdim=True)
        code_sq = (self.quantizer.codebook.weight ** 2).sum(dim=1)
        distances = z_e_sq + code_sq - 2 * z_e @ self.quantizer.codebook.weight.t()

        indices = torch.argmin(distances, dim=1)
        tokens = indices.view(images.size(0), patch_count)
        return tokens


def extract_patches(images: torch.Tensor, patch_size: int, overlap: float):
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")

    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    unfolded = F.unfold(images, kernel_size=patch_size, stride=stride)
    # [B, C*P*P, L] -> [B*L, C*P*P]
    patches = unfolded.transpose(1, 2).reshape(-1, unfolded.size(1))
    return patches, unfolded.size(-1)


def build_transforms(blur: bool):
    transforms_list = [transforms.ToTensor()]
    if blur:
        transforms_list.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)))
    # Map [0, 1] to [-1, 1] without lambda so DataLoader workers can pickle transforms on Python 3.14+.
    transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    return transforms.Compose(transforms_list)


def make_loaders(cfg: DistillConfig):
    Path(cfg.data_root).mkdir(parents=True, exist_ok=True)
    transform = build_transforms(cfg.blur)
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
    return train_loader, DataLoader(val_set, **eval_loader_kwargs), DataLoader(test_set, **eval_loader_kwargs)


def train_tokenizer(cfg: DistillConfig, device: torch.device):
    train_loader, val_loader, test_loader = make_loaders(cfg)
    model = PatchVQTokenizer(
        in_channels=3,
        patch_size=cfg.patch_size,
        hidden_dim=cfg.hidden_dim,
        code_dim=cfg.code_dim,
        codebook_size=cfg.codebook_size,
        commit_weight=cfg.commit_weight,
    ).to(device)
    num_classes = len(medmnist.INFO["pathmnist"]["label"])
    aux_head = nn.Linear(cfg.code_dim, num_classes).to(device)

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(aux_head.parameters()), lr=cfg.lr)
    cls_criterion = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        steps = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).to(device, non_blocking=True)
            patches, patch_count = extract_patches(images, cfg.patch_size, cfg.overlap)

            recon, z_q, _, recon_loss, commit_loss = model(patches)
            _ = recon
            image_repr = z_q.view(images.size(0), patch_count, -1).mean(dim=1)
            cls_logits = aux_head(image_repr)
            cls_loss = cls_criterion(cls_logits, labels)

            loss = cfg.recon_weight * recon_loss + commit_loss + cfg.cls_weight * cls_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total += float(loss.item())
            steps += 1

        avg = total / max(1, steps)
        print(f"epoch={epoch + 1:02d} train_loss={avg:.6f}")

    return model, (train_loader, val_loader, test_loader)


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
        all_labels.append(labels.view(-1).cpu().numpy().astype(np.int64))

    tokens_np = np.concatenate(all_tokens, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    output_path = output_dir / f"{split_name}_tokens.npz"
    np.savez_compressed(output_path, tokens=tokens_np, labels=labels_np)
    print(f"saved {split_name}: {output_path} | tokens shape={tokens_np.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Distill PathMNIST into discrete tokens with a VQ tokenizer.")
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DistillConfig(
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
        seed=args.seed,
    )

    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    medmnist.INFO["pathmnist"]  # Trigger import guard and ensure metadata is available.

    model, loaders = train_tokenizer(cfg, device)
    train_loader, val_loader, test_loader = loaders

    export_split_tokens("train", train_loader, model, cfg.overlap, output_dir, device)
    export_split_tokens("val", val_loader, model, cfg.overlap, output_dir, device)
    export_split_tokens("test", test_loader, model, cfg.overlap, output_dir, device)

    torch.save(model.state_dict(), output_dir / "vq_tokenizer.pt")
    metadata = {
        "dataset": "PathMNIST",
        "patch_size": cfg.patch_size,
        "overlap": cfg.overlap,
        "codebook_size": cfg.codebook_size,
        "code_dim": cfg.code_dim,
        "hidden_dim": cfg.hidden_dim,
        "cls_weight": cfg.cls_weight,
        "epochs": cfg.epochs,
        "blur": cfg.blur,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"saved model + metadata to {output_dir}")


if __name__ == "__main__":
    main()
