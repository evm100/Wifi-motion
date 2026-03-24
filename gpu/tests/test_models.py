"""Tests for all GPU model architectures — verify forward pass shapes."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import (
    CSIResNet,
    CNNGRU,
    CSITransformer,
    MultiNodeFusion,
    CSILocalizer,
    DomainAdaptiveCSINet,
)

N_CLASSES = 7


class TestCSIResNet:
    def _make_model(self):
        return CSIResNet(n_input_channels=30, n_classes=N_CLASSES, pretrained=False)

    def test_output_shape_batch1(self):
        model = self._make_model()
        x = torch.randn(1, 30, 65, 32)
        out = model(x)
        assert out.shape == (1, N_CLASSES)

    def test_output_shape_batch8(self):
        model = self._make_model()
        x = torch.randn(8, 30, 65, 32)
        out = model(x)
        assert out.shape == (8, N_CLASSES)


class TestCNNGRU:
    def _make_model(self):
        return CNNGRU(n_input_channels=3, n_classes=N_CLASSES, hidden_dim=64)

    def test_output_shape_batch1(self):
        model = self._make_model()
        x = torch.randn(1, 10, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, N_CLASSES)

    def test_output_shape_batch8(self):
        model = self._make_model()
        x = torch.randn(8, 10, 3, 32, 32)
        out = model(x)
        assert out.shape == (8, N_CLASSES)


class TestCSITransformer:
    def _make_model(self):
        return CSITransformer(
            input_dim=60, n_classes=N_CLASSES, d_model=64, nhead=4, num_layers=2
        )

    def test_output_shape_batch1(self):
        model = self._make_model()
        x = torch.randn(1, 50, 60)
        out = model(x)
        assert out.shape == (1, N_CLASSES)

    def test_output_shape_batch8(self):
        model = self._make_model()
        x = torch.randn(8, 50, 60)
        out = model(x)
        assert out.shape == (8, N_CLASSES)


class TestMultiNodeFusion:
    def _make_model(self):
        return MultiNodeFusion(
            feature_dim=128, n_nodes=3, n_classes=N_CLASSES, n_heads=4
        )

    def test_output_shape_batch1(self):
        model = self._make_model()
        inputs = [torch.randn(1, 20, 108) for _ in range(3)]
        out = model(inputs)
        assert out.shape == (1, N_CLASSES)

    def test_output_shape_batch8(self):
        model = self._make_model()
        inputs = [torch.randn(8, 20, 108) for _ in range(3)]
        out = model(inputs)
        assert out.shape == (8, N_CLASSES)


class TestCSILocalizer:
    def _make_model(self):
        return CSILocalizer(n_nodes=3, n_subcarriers=108, n_zones=16)

    def test_regression_shape_batch1(self):
        model = self._make_model()
        model.eval()  # BatchNorm1d requires batch>1 in train mode
        x = torch.randn(1, 324)
        out = model(x, mode="regression")
        assert out.shape == (1, 2)

    def test_zone_shape_batch8(self):
        model = self._make_model()
        x = torch.randn(8, 324)
        out = model(x, mode="zone")
        assert out.shape == (8, 16)


class TestDomainAdaptiveCSINet:
    def _make_model(self):
        return DomainAdaptiveCSINet(
            n_input_channels=30, n_classes=N_CLASSES, n_domains=3, lambda_domain=1.0
        )

    def test_output_shape_batch1(self):
        model = self._make_model()
        x = torch.randn(1, 30, 65, 32)
        activity, domain = model(x)
        assert activity.shape == (1, N_CLASSES)
        assert domain.shape == (1, 3)

    def test_output_shape_batch8(self):
        model = self._make_model()
        x = torch.randn(8, 30, 65, 32)
        activity, domain = model(x)
        assert activity.shape == (8, N_CLASSES)
        assert domain.shape == (8, 3)
