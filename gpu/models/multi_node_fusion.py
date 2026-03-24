"""Attention-based multi-node fusion for CSI sensing."""

import torch
import torch.nn as nn
from typing import List


class MultiNodeFusion(nn.Module):
    """
    Attention-based fusion for multiple RX nodes.
    Per-node linear encoders, cross-node multi-head attention, classifier.

    Input: list of [B, T, 108] tensors, one per node
    Output: [B, n_classes] logits
    """

    def __init__(
        self,
        feature_dim: int = 256,
        n_nodes: int = 3,
        n_classes: int = 7,
        n_heads: int = 4,
        input_dim: int = 108,
    ) -> None:
        super().__init__()
        self.n_nodes = n_nodes

        self.node_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, feature_dim),
            )
            for _ in range(n_nodes)
        ])

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * n_nodes, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, node_features: List[torch.Tensor]) -> torch.Tensor:
        encoded = []
        for i, feat in enumerate(node_features):
            feat_avg = feat.mean(dim=1)  # [B, 108]
            enc = self.node_encoders[i](feat_avg)  # [B, feature_dim]
            encoded.append(enc)

        # Stack for attention: [B, n_nodes, feature_dim]
        stacked = torch.stack(encoded, dim=1)

        # Cross-node self-attention
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        # Flatten and classify
        fused = attended.reshape(attended.size(0), -1)
        return self.classifier(fused)
