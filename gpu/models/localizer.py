"""CSI-based position regression and zone classification."""

import torch
import torch.nn as nn


class CSILocalizer(nn.Module):
    """
    Regress (x, y) room position or classify zone from multi-node CSI.

    Input: [B, n_nodes * n_subcarriers] concatenated amplitude features (default 324)
    Output: [B, 2] for regression or [B, n_zones] for zone classification
    """

    def __init__(
        self,
        n_nodes: int = 3,
        n_subcarriers: int = 108,
        hidden_dim: int = 256,
        n_zones: int = 16,
    ) -> None:
        super().__init__()

        input_dim = n_nodes * n_subcarriers  # 324

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
        )

        self.position_head = nn.Linear(128, 2)
        self.zone_head = nn.Linear(128, n_zones)

    def forward(self, x: torch.Tensor, mode: str = "regression") -> torch.Tensor:
        features = self.encoder(x)

        if mode == "regression":
            return self.position_head(features)  # [B, 2]
        else:
            return self.zone_head(features)  # [B, n_zones]
