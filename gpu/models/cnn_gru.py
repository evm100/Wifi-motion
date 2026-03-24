"""CNN-GRU hybrid for temporal CSI sequence classification (Widar3.0 style)."""

import torch
import torch.nn as nn


class CNNGRU(nn.Module):
    """
    Per-frame CNN spatial feature extractor followed by bidirectional GRU.

    Input: [B, T, C, H, W] — sequence of T spectrogram frames
    Output: [B, n_classes] logits
    """

    def __init__(
        self,
        n_input_channels: int = 3,
        n_classes: int = 7,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.gru = nn.GRU(
            input_size=128 * 4 * 4,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape

        # Apply CNN to each time step
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)
        features = features.view(B, T, -1)  # [B, T, 128*4*4]

        # GRU over temporal sequence
        gru_out, _ = self.gru(features)  # [B, T, hidden*2]

        # Use last time step output
        out = gru_out[:, -1, :]
        return self.fc(out)
