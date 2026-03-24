"""Transformer encoder for CSI activity recognition on PCA time series."""

import torch
import torch.nn as nn


class CSITransformer(nn.Module):
    """
    Transformer encoder for CSI activity recognition.
    Processes PCA-reduced time series directly.

    Input: [B, T, D] where D = n_nodes * n_pcs
    Output: [B, n_classes] logits
    """

    def __init__(
        self,
        input_dim: int = 60,
        n_classes: int = 7,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        # Project input and add positional encoding
        x = self.input_proj(x)  # [B, T, d_model]
        x = x + self.pos_embedding[:, :T, :]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, T+1, d_model]

        # Transformer encoding
        x = self.transformer(x)

        # CLS token output for classification
        cls_out = x[:, 0, :]
        return self.fc(cls_out)
