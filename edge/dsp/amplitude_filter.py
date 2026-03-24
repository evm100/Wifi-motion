"""
amplitude_filter.py — Amplitude denoising filters for CSI data.

HampelFilter: outlier removal using median/MAD (robust to contamination).
ButterFilter: Butterworth lowpass for high-frequency noise removal.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt


class HampelFilter:
    """
    Hampel filter for outlier detection and replacement.

    Uses median absolute deviation (MAD) instead of mean/std, making it
    robust to up to 50% contamination. The factor 1.4826 converts MAD to
    a consistent estimator of standard deviation for Gaussian data.
    """

    MAD_SCALE = 1.4826

    def __init__(self, window_size: int = 5, n_sigma: float = 3.0) -> None:
        """
        Args:
            window_size: Half-window size (total window = 2*window_size + 1).
            n_sigma: Threshold in MAD-scaled sigma units.
        """
        self.window_size = window_size
        self.n_sigma = n_sigma

    def filter_1d(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply Hampel filter to a 1D time series.

        Args:
            data: 1D array (e.g., amplitude for one subcarrier over time).

        Returns:
            filtered: Cleaned array with outliers replaced by local median.
            outlier_mask: Boolean array marking detected outliers.
        """
        n = len(data)
        filtered = data.copy()
        outlier_mask = np.zeros(n, dtype=bool)
        k = self.MAD_SCALE

        for i in range(n):
            lo = max(0, i - self.window_size)
            hi = min(n, i + self.window_size + 1)
            window = data[lo:hi]

            median = np.median(window)
            mad = np.median(np.abs(window - median))

            deviation = np.abs(data[i] - median)
            if mad > 0:
                if deviation > self.n_sigma * k * mad:
                    filtered[i] = median
                    outlier_mask[i] = True
            elif deviation > 0:
                # MAD is 0 (constant window) but sample differs — it's an outlier
                filtered[i] = median
                outlier_mask[i] = True

        return filtered, outlier_mask

    def filter(self, data_2d: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Apply Hampel filter to [n_packets, n_subcarriers] amplitude matrix.

        Each subcarrier's time series is filtered independently.

        Args:
            data_2d: Amplitude matrix [n_packets, n_subcarriers].

        Returns:
            filtered: Cleaned amplitude matrix.
            total_outliers: Total number of outliers replaced.
        """
        if data_2d.ndim == 1:
            filtered, mask = self.filter_1d(data_2d)
            return filtered, int(mask.sum())

        n_packets, n_sc = data_2d.shape
        filtered = data_2d.copy()
        total_outliers = 0

        for sc in range(n_sc):
            filtered[:, sc], mask = self.filter_1d(data_2d[:, sc])
            total_outliers += int(mask.sum())

        return filtered, total_outliers


class ButterFilter:
    """
    Butterworth low-pass filter for CSI amplitude denoising.

    Uses second-order sections (SOS) form for numerical stability.
    Provides both offline (zero-phase) and real-time (causal with state) modes.
    """

    def __init__(
        self,
        cutoff_hz: float = 10.0,
        fs: float = 100.0,
        order: int = 4,
    ) -> None:
        """
        Args:
            cutoff_hz: Low-pass cutoff frequency in Hz.
            fs: CSI sampling rate in Hz.
            order: Filter order (4 is standard).
        """
        self.cutoff = cutoff_hz
        self.fs = fs
        self.order = order
        nyquist = fs / 2.0
        normalized_cutoff = cutoff_hz / nyquist
        self.sos = butter(order, normalized_cutoff, btype="low", output="sos")
        self._zi_template = sosfilt_zi(self.sos)

    def apply_offline(self, data: np.ndarray) -> np.ndarray:
        """
        Zero-phase filtering (no time delay) for batch/offline processing.

        Args:
            data: [n_packets, n_subcarriers] amplitude matrix, or 1D.

        Returns:
            Filtered data, same shape as input.
        """
        return sosfiltfilt(self.sos, data, axis=0)

    def apply_realtime(
        self,
        data: np.ndarray,
        zi: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Causal filtering with state for real-time streaming.

        Args:
            data: [n_new_packets, n_subcarriers] new amplitude data.
            zi: Filter state from previous call (None for first call).

        Returns:
            filtered: Filtered data.
            zf: Updated filter state (pass to next call).
        """
        if zi is None:
            n_sc = data.shape[1] if data.ndim > 1 else 1
            zi = np.repeat(self._zi_template[:, :, np.newaxis], n_sc, axis=2)
            zi *= data[0] if data.ndim == 1 else data[0, :]

        filtered, zf = sosfilt(self.sos, data, axis=0, zi=zi)
        return filtered, zf
