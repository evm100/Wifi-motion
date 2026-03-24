"""CSI spectrogram dataset for PyTorch training."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CSISpectrogramDataset(Dataset):
    """
    Dataset of CSI spectrograms stored as .npy files with integer labels.

    Each sample is a (spectrogram_tensor, label) pair.
    """

    def __init__(self, spectrograms: np.ndarray, labels: np.ndarray) -> None:
        """
        Args:
            spectrograms: [N, C, H, W] float32 array of spectrograms
            labels: [N] int64 array of class labels
        """
        self.spectrograms = torch.from_numpy(spectrograms).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.spectrograms[idx], self.labels[idx]

    @classmethod
    def from_directory(cls, data_dir: str | Path) -> "CSISpectrogramDataset":
        """
        Load dataset from a processed/ folder structure.

        Expected layout:
            data_dir/
                spectrograms.npy   — [N, C, H, W] float32
                labels.npy         — [N] int64

        Args:
            data_dir: Path to directory containing .npy files

        Returns:
            CSISpectrogramDataset instance
        """
        data_dir = Path(data_dir)
        spectrograms = np.load(data_dir / "spectrograms.npy")
        labels = np.load(data_dir / "labels.npy")
        return cls(spectrograms, labels)
