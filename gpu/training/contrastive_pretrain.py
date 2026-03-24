"""SimCLR-style contrastive pre-training for CSI features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CSIResNet


class ContrastivePairDataset(Dataset):
    """Dataset yielding (node_a_spec, node_b_spec) as positive pairs."""

    def __init__(self, specs_a: np.ndarray, specs_b: np.ndarray) -> None:
        self.specs_a = torch.from_numpy(specs_a).float()
        self.specs_b = torch.from_numpy(specs_b).float()

    def __len__(self) -> int:
        return len(self.specs_a)

    def __getitem__(self, idx: int):
        return self.specs_a[idx], self.specs_b[idx]


class ContrastiveCSIPretrainer(nn.Module):
    """
    SimCLR-style contrastive pre-training for CSI features.

    Positive pairs: spectrograms from different RX nodes at the same time.
    Negative pairs: spectrograms from different time windows.

    After pre-training, discard the projection head and fine-tune the encoder.
    """

    def __init__(self, encoder: nn.Module, feature_dim: int = 512, projection_dim: int = 64) -> None:
        super().__init__()
        self.encoder = encoder

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        z1 = self.projector(self.encode(x1))
        z2 = self.projector(self.encode(x2))
        return z1, z2

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run encoder without projection head (for downstream)."""
        return self.encoder(x)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """NT-Xent loss (normalized temperature-scaled cross entropy)."""
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        z = F.normalize(z, dim=1)

        sim = torch.mm(z, z.T) / temperature  # [2B, 2B]

        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, -1e9)

        labels = torch.cat([
            torch.arange(B, 2 * B), torch.arange(0, B)
        ]).to(z.device)

        return F.cross_entropy(sim, labels)


def build_encoder(cfg: dict) -> nn.Module:
    """Build a ResNet encoder that outputs feature vectors (no classification head)."""
    n_channels = cfg.get("n_channels", 10)
    model = CSIResNet(n_input_channels=n_channels, n_classes=1, pretrained=cfg.get("pretrained", True))
    # Replace classification head with identity to get 512-d features
    model.fc = nn.Identity()
    return model


def train_contrastive(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Contrastive pre-training on {device}")

    data_dir = Path(cfg.get("data_dir", "data/contrastive"))
    if data_dir.exists():
        specs_a = np.load(data_dir / "node_a.npy")
        specs_b = np.load(data_dir / "node_b.npy")
    else:
        print("Data dir not found, generating synthetic pairs")
        n = cfg.get("n_synthetic", 500)
        c = cfg.get("n_channels", 10)
        h, w = cfg.get("spec_h", 65), cfg.get("spec_w", 32)
        specs_a = np.random.randn(n, c, h, w).astype(np.float32)
        specs_b = np.random.randn(n, c, h, w).astype(np.float32)

    dataset = ContrastivePairDataset(specs_a, specs_b)
    loader = DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    encoder = build_encoder(cfg)
    pretrainer = ContrastiveCSIPretrainer(
        encoder,
        feature_dim=512,
        projection_dim=cfg.get("projection_dim", 64),
    ).to(device)

    optimizer = torch.optim.AdamW(
        pretrainer.parameters(),
        lr=cfg.get("lr", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    temperature = cfg.get("temperature", 0.5)

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.get("epochs", 50)):
        pretrainer.train()
        total_loss = 0.0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = pretrainer(x1, x2)
            loss = pretrainer.contrastive_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch:3d} | contrastive_loss={avg_loss:.4f}")

    # Save encoder weights for downstream fine-tuning
    save_path = checkpoint_dir / "pretrained_encoder.pt"
    torch.save(pretrainer.encoder.state_dict(), save_path)
    print(f"Saved encoder weights to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive pre-training for CSI")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_contrastive(cfg)


if __name__ == "__main__":
    main()
