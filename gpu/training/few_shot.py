"""Prototypical Networks for few-shot CSI activity recognition."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class PrototypicalCSINet(nn.Module):
    """
    Prototypical Network for few-shot CSI activity recognition.

    Maps spectrograms to an embedding space. At deployment, compute class
    prototypes from a small support set and classify by nearest prototype.

    Input: [B, C, H, W] spectrogram
    Output: [B, embedding_dim] embeddings
    """

    def __init__(
        self,
        n_input_channels: int = 30,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def classify_few_shot(
        self,
        support_set: Dict[int, torch.Tensor],
        query: torch.Tensor,
        n_classes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify queries by nearest prototype from the support set.

        Args:
            support_set: {class_label: [K, C, H, W] support examples}
            query: [B, C, H, W] query examples to classify
            n_classes: number of classes (used for ordering)

        Returns:
            predictions: [B] predicted class labels
            logits: [B, n_classes] negative distances (usable as logits)
        """
        prototypes = {}
        for label, examples in support_set.items():
            embeddings = self.encoder(examples)
            prototypes[label] = embeddings.mean(dim=0)

        proto_stack = torch.stack([prototypes[l] for l in sorted(prototypes)])  # [n_classes, D]

        query_emb = self.encoder(query)  # [B, D]

        # Euclidean distances to prototypes
        distances = torch.cdist(query_emb, proto_stack.unsqueeze(0)).squeeze(0)  # [B, n_classes]

        predictions = distances.argmin(dim=1)
        return predictions, -distances
