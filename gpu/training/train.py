"""Main training script for CSI activity recognition models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split

# Add project root so proto/ and gpu/ are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CSIResNet, CNNGRU, CSITransformer
from training.dataset import CSISpectrogramDataset


def build_model(cfg: dict) -> nn.Module:
    model_type = cfg["model"]
    if model_type == "resnet":
        return CSIResNet(
            n_input_channels=cfg.get("n_channels", 30),
            n_classes=cfg.get("n_classes", 7),
            pretrained=cfg.get("pretrained", True),
        )
    elif model_type == "cnn_gru":
        return CNNGRU(
            n_input_channels=cfg.get("n_channels", 3),
            n_classes=cfg.get("n_classes", 7),
            hidden_dim=cfg.get("hidden_dim", 128),
        )
    elif model_type == "transformer":
        return CSITransformer(
            input_dim=cfg.get("input_dim", 60),
            n_classes=cfg.get("n_classes", 7),
            d_model=cfg.get("d_model", 128),
            nhead=cfg.get("nhead", 8),
            num_layers=cfg.get("num_layers", 4),
            dim_feedforward=cfg.get("dim_feedforward", 256),
            dropout=cfg.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train(cfg: dict) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_dir = Path(cfg.get("data_dir", "data/processed"))
    if data_dir.exists():
        dataset = CSISpectrogramDataset.from_directory(data_dir)
    else:
        print(f"Data dir {data_dir} not found, generating synthetic data for testing")
        n_samples = cfg.get("n_synthetic", 200)
        n_channels = cfg.get("n_channels", 30)
        h, w = cfg.get("spec_h", 65), cfg.get("spec_w", 32)
        n_classes = cfg.get("n_classes", 7)
        specs = np.random.randn(n_samples, n_channels, h, w).astype(np.float32)
        labels = np.random.randint(0, n_classes, size=n_samples).astype(np.int64)
        dataset = CSISpectrogramDataset(specs, labels)

    # Split train/val
    val_ratio = cfg.get("val_ratio", 0.2)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=cfg.get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Build model
    model = build_model(cfg).to(device)
    print(f"Model: {cfg['model']} | Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 100)
    )
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.get("label_smoothing", 0.1)
    )

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(cfg.get("epochs", 100)):
        # Training
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)

            outputs = model(specs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += len(labels)

        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / max(val_total, 1)
        train_acc = correct / max(total, 1)
        avg_loss = train_loss / max(len(train_loader), 1)

        print(
            f"Epoch {epoch:3d}/{cfg.get('epochs', 100)} | "
            f"loss={avg_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = checkpoint_dir / f"best_{cfg['model']}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model (val_acc={val_acc:.3f})")

    print(f"Training complete. Best val_acc={best_val_acc:.3f}")
    return best_val_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CSI activity recognition model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
