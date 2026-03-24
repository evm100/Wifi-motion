"""Tests for RealtimeCSIClassifier inference wrapper."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.classifier import RealtimeCSIClassifier, CLASS_NAMES


class TestRealtimeCSIClassifier:
    def _make_classifier(self):
        return RealtimeCSIClassifier(
            model_path=None,
            n_input_channels=30,
            n_classes=7,
            device="cpu",
        )

    def test_predict_returns_valid_class(self):
        clf = self._make_classifier()
        spec = np.random.randn(30, 65, 32).astype(np.float32)
        activity, confidence, latency_ms = clf.predict(spec)
        assert activity in CLASS_NAMES

    def test_confidence_range(self):
        clf = self._make_classifier()
        spec = np.random.randn(30, 65, 32).astype(np.float32)
        _, confidence, _ = clf.predict(spec)
        assert 0.0 <= confidence <= 1.0

    def test_positive_latency(self):
        clf = self._make_classifier()
        spec = np.random.randn(30, 65, 32).astype(np.float32)
        _, _, latency_ms = clf.predict(spec)
        assert latency_ms > 0.0

    def test_multiple_predictions_consistent_format(self):
        clf = self._make_classifier()
        for _ in range(5):
            spec = np.random.randn(30, 65, 32).astype(np.float32)
            activity, confidence, latency_ms = clf.predict(spec)
            assert isinstance(activity, str)
            assert isinstance(confidence, float)
            assert isinstance(latency_ms, float)
