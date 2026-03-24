"""
test_filters.py — Tests for DSP filters: Hampel, Butterworth, and baseline.

Uses synthetic signals with known properties to verify filter behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edge.dsp.amplitude_filter import HampelFilter, ButterFilter
from edge.dsp.baseline import AdaptiveBaseline


class TestHampelFilter:
    def test_replaces_known_outlier(self):
        """Single large spike should be detected and replaced by median."""
        data = np.ones(50)
        data[25] = 100.0  # Obvious outlier

        hf = HampelFilter(window_size=5, n_sigma=3.0)
        filtered, mask = hf.filter_1d(data)

        assert mask[25], "Outlier at index 25 should be detected"
        assert filtered[25] == pytest.approx(1.0), "Should be replaced by median"

    def test_no_false_positives_on_clean_data(self):
        """Clean constant data should have zero outliers."""
        data = np.ones(100) * 5.0
        hf = HampelFilter(window_size=5, n_sigma=3.0)
        filtered, mask = hf.filter_1d(data)

        assert not mask.any(), "No outliers in clean data"
        np.testing.assert_array_equal(filtered, data)

    def test_2d_applies_per_subcarrier(self):
        """Each column should be filtered independently."""
        rng = np.random.RandomState(42)
        data = rng.randn(50, 10) * 0.1 + 5.0
        # Inject outlier in subcarrier 3, sample 20
        data[20, 3] = 500.0

        hf = HampelFilter(window_size=5, n_sigma=3.0)
        filtered, total = hf.filter(data)

        assert total >= 1, "Should detect at least the injected outlier"
        assert filtered[20, 3] != pytest.approx(500.0), "Outlier should be replaced"
        # Other subcarriers should be mostly unchanged (allow a few Hampel corrections)
        diff_count = np.sum(~np.isclose(filtered[:, 0], data[:, 0], atol=0.01))
        assert diff_count < 5, f"Too many changes in clean subcarrier: {diff_count}"

    def test_multiple_outliers(self):
        data = np.ones(100) * 10.0
        data[10] = 1000.0
        data[50] = -500.0
        data[90] = 999.0

        hf = HampelFilter(window_size=5, n_sigma=3.0)
        filtered, mask = hf.filter_1d(data)

        assert mask[10] and mask[50] and mask[90]
        assert filtered[10] == pytest.approx(10.0)


class TestButterFilter:
    def test_attenuates_above_cutoff(self):
        """A 30 Hz sine should be attenuated by a 10 Hz lowpass at 100 Hz fs."""
        fs = 100.0
        t = np.arange(0, 2.0, 1.0 / fs)
        # 3 Hz signal + 30 Hz noise
        signal_low = np.sin(2 * np.pi * 3 * t)
        noise_high = 0.5 * np.sin(2 * np.pi * 30 * t)
        mixed = signal_low + noise_high

        bf = ButterFilter(cutoff_hz=10.0, fs=fs, order=4)
        filtered = bf.apply_offline(mixed)

        # After filtering, the high-freq component should be greatly reduced
        # Check middle portion to avoid edge effects
        mid = len(t) // 4
        end = 3 * len(t) // 4
        residual_power = np.mean((filtered[mid:end] - signal_low[mid:end]) ** 2)
        original_noise_power = np.mean(noise_high[mid:end] ** 2)

        assert residual_power < 0.01 * original_noise_power, (
            "30 Hz noise should be attenuated by >20 dB"
        )

    def test_preserves_below_cutoff(self):
        """A 2 Hz sine should pass through a 10 Hz lowpass nearly unchanged."""
        fs = 100.0
        t = np.arange(0, 3.0, 1.0 / fs)
        signal = np.sin(2 * np.pi * 2 * t)

        bf = ButterFilter(cutoff_hz=10.0, fs=fs, order=4)
        filtered = bf.apply_offline(signal)

        mid = len(t) // 4
        end = 3 * len(t) // 4
        correlation = np.corrcoef(signal[mid:end], filtered[mid:end])[0, 1]
        assert correlation > 0.99, "Signal below cutoff should be preserved"

    def test_2d_input(self):
        """Should filter [n_packets, n_subcarriers] along axis=0."""
        fs = 100.0
        n_samples = 200
        n_sc = 5
        t = np.arange(n_samples) / fs
        data = np.column_stack([np.sin(2 * np.pi * 2 * t)] * n_sc)

        bf = ButterFilter(cutoff_hz=10.0, fs=fs)
        filtered = bf.apply_offline(data)

        assert filtered.shape == (n_samples, n_sc)

    def test_realtime_produces_output(self):
        """Real-time mode should return filtered data and state."""
        bf = ButterFilter(cutoff_hz=10.0, fs=100.0, order=4)
        data = np.random.randn(50, 3).astype(np.float64)

        filtered, zi = bf.apply_realtime(data)
        assert filtered.shape == data.shape
        assert zi is not None

        # Second call with state
        data2 = np.random.randn(50, 3).astype(np.float64)
        filtered2, zi2 = bf.apply_realtime(data2, zi=zi)
        assert filtered2.shape == data2.shape


class TestAdaptiveBaseline:
    def test_calibration_completes(self):
        """After n_required samples, calibrated should be True."""
        bl = AdaptiveBaseline(fast_alpha=0.1, slow_alpha=0.001)
        amp = np.ones(10) * 5.0

        for i in range(299):
            done = bl.calibrate(amp, n_required=300)
            assert not done

        done = bl.calibrate(amp, n_required=300)
        assert done
        assert bl.calibrated

    def test_baseline_converges(self):
        """Baseline should converge to the constant input during calibration."""
        bl = AdaptiveBaseline(fast_alpha=0.1, slow_alpha=0.001)
        amp = np.ones(10) * 42.0

        for _ in range(300):
            bl.calibrate(amp, n_required=300)

        np.testing.assert_allclose(bl.baseline, 42.0, atol=0.5)

    def test_remove_static_isolates_dynamic(self):
        """Dynamic signal should be amplitude minus baseline."""
        bl = AdaptiveBaseline()
        static = np.ones(10) * 10.0

        # Calibrate with static
        for _ in range(300):
            bl.calibrate(static, n_required=300)

        # Add dynamic component
        dynamic_input = static + 5.0  # 15.0 total
        result = bl.remove_static(dynamic_input)

        # Result should be approximately 5.0 (the dynamic part)
        np.testing.assert_allclose(result, 5.0, atol=0.5)

    def test_slow_adaptation(self):
        """Baseline should slowly adapt when signal energy is low."""
        bl = AdaptiveBaseline(fast_alpha=0.1, slow_alpha=0.01)  # faster slow for test
        static = np.ones(10) * 10.0

        for _ in range(300):
            bl.calibrate(static, n_required=300)

        initial_baseline = bl.baseline.copy()

        # Feed slightly different static (drift)
        new_static = np.ones(10) * 10.05
        for _ in range(100):
            bl.remove_static(new_static)

        # Baseline should have drifted slightly toward 10.05
        assert np.mean(bl.baseline) > np.mean(initial_baseline)
