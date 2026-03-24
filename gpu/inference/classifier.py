"""Real-time CSI classifier wrapping model loading, device placement, and prediction."""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np
import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import CSIResNet


CLASS_NAMES: List[str] = [
    "empty", "walking", "sitting", "standing",
    "falling", "gesture", "breathing",
]


class RealtimeCSIClassifier:
    """
    Wraps model loading, GPU placement, and inference.

    Returns (activity_name, confidence, latency_ms) per prediction.
    """

    def __init__(
        self,
        model_path: str | None = None,
        n_input_channels: int = 30,
        n_classes: int = 7,
        device: str = "cuda",
        class_names: List[str] | None = None,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.class_names = class_names or CLASS_NAMES[:n_classes]

        self.model = CSIResNet(
            n_input_channels=n_input_channels,
            n_classes=n_classes,
            pretrained=False,
        )

        if model_path is not None:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )

        self.model.to(self.device)
        self.model.eval()

        # Warm up GPU with dummy forward pass
        dummy = torch.randn(1, n_input_channels, 65, 32).to(self.device)
        with torch.no_grad():
            self.model(dummy)

    @torch.no_grad()
    def predict(self, spectrogram: np.ndarray) -> Tuple[str, float, float]:
        """
        Classify a single spectrogram.

        Args:
            spectrogram: [C, H, W] numpy array

        Returns:
            activity: predicted activity name
            confidence: softmax probability of prediction
            latency_ms: inference time in milliseconds
        """
        t0 = time.perf_counter()

        x = torch.from_numpy(spectrogram).float().unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

        latency_ms = (time.perf_counter() - t0) * 1000

        return self.class_names[pred_idx], confidence, latency_ms
