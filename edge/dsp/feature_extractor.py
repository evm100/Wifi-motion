"""
feature_extractor.py — Statistical and spectral feature extraction from CSI.

All methods operate on numpy arrays and are vectorized where possible.
Features feed into the GPU deep learning models via ZMQ.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft, welch
from scipy.stats import entropy as scipy_entropy


class CSIFeatureExtractor:
    """
    Extract time-domain, frequency-domain, and spatial features from
    preprocessed CSI amplitude/phase data on sliding windows.
    """

    def __init__(
        self,
        fs: float = 100.0,
        window_size: int = 256,
        hop_size: int = 50,
    ) -> None:
        """
        Args:
            fs: CSI sampling rate (Hz).
            window_size: Feature window in samples (256 = 2.56 sec @ 100 Hz).
            hop_size: Stride between windows (50 = 0.5 sec).
        """
        self.fs = fs
        self.window_size = window_size
        self.hop_size = hop_size

    # --- Time-domain features ---

    def amplitude_variance(self, amplitude_window: np.ndarray) -> np.ndarray:
        """
        Per-subcarrier variance over the time window.

        Args:
            amplitude_window: [window_size, n_subcarriers].

        Returns:
            variance: [n_subcarriers].
        """
        return np.var(amplitude_window, axis=0)

    def amplitude_range(self, amplitude_window: np.ndarray) -> np.ndarray:
        """
        Peak-to-peak amplitude range per subcarrier.

        Args:
            amplitude_window: [window_size, n_subcarriers].

        Returns:
            range: [n_subcarriers].
        """
        return np.ptp(amplitude_window, axis=0)

    def temporal_correlation(
        self, amplitude_window: np.ndarray, lag: int = 1
    ) -> np.ndarray:
        """
        Autocorrelation at specified lag per subcarrier.

        High correlation = periodic motion (walking cadence).
        Low correlation = random or static.

        Args:
            amplitude_window: [window_size, n_subcarriers].
            lag: Lag in samples.

        Returns:
            corr: [n_subcarriers] normalized autocorrelation.
        """
        x = amplitude_window[:-lag]
        y = amplitude_window[lag:]
        corr = np.mean(x * y, axis=0) / (
            np.std(x, axis=0) * np.std(y, axis=0) + 1e-10
        )
        return corr

    def signal_entropy(
        self, amplitude_window: np.ndarray, n_bins: int = 20
    ) -> np.ndarray:
        """
        Shannon entropy of amplitude distribution per subcarrier.

        High entropy = motion (spread distribution).
        Low entropy = static (concentrated).

        Args:
            amplitude_window: [window_size, n_subcarriers].
            n_bins: Histogram bins for distribution estimation.

        Returns:
            entropies: [n_subcarriers].
        """
        n_sc = amplitude_window.shape[1]
        entropies = np.zeros(n_sc)
        for sc in range(n_sc):
            hist, _ = np.histogram(amplitude_window[:, sc], bins=n_bins, density=True)
            hist = hist[hist > 0]
            entropies[sc] = scipy_entropy(hist)
        return entropies

    # --- Frequency-domain features ---

    def doppler_spectrogram(
        self,
        csi_timeseries: np.ndarray,
        nperseg: int = 128,
        noverlap: int = 120,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Doppler spectrogram from complex CSI time series.

        Primary input for CNN-based activity classification.

        Args:
            csi_timeseries: Complex CSI [n_packets, n_subcarriers].
            nperseg: STFT window size.
            noverlap: Overlap samples.

        Returns:
            spectrograms: [n_subcarriers, n_freq_bins, n_time_bins].
            freqs: Frequency (Doppler) axis in Hz.
            times: Time axis in seconds.
        """
        n_packets, n_sc = csi_timeseries.shape
        specs = []

        for sc in range(n_sc):
            f, t, Zxx = stft(
                csi_timeseries[:, sc],
                fs=self.fs,
                nperseg=nperseg,
                noverlap=noverlap,
            )
            specs.append(np.abs(Zxx) ** 2)

        return np.array(specs), f, t

    def power_spectral_density(
        self,
        amplitude_window: np.ndarray,
        nperseg: int = 128,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Power spectral density per subcarrier via Welch's method.

        Args:
            amplitude_window: [window_size, n_subcarriers].
            nperseg: Segment length for Welch.

        Returns:
            psds: [n_subcarriers, n_freq_bins].
            freqs: Frequency axis in Hz.
        """
        n_t, n_sc = amplitude_window.shape
        psds = []
        for sc in range(n_sc):
            f, psd = welch(amplitude_window[:, sc], fs=self.fs, nperseg=nperseg)
            psds.append(psd)
        return np.array(psds), f

    # --- Multi-node spatial features ---

    def spatial_variance(self, node_amplitudes: np.ndarray) -> float:
        """
        Variance of amplitude across subcarriers at a single time instant.

        Args:
            node_amplitudes: [n_subcarriers] from one node.

        Returns:
            Spatial variance (scalar).
        """
        return float(np.var(node_amplitudes))

    def cross_node_correlation(
        self,
        amp_node_a: np.ndarray,
        amp_node_b: np.ndarray,
    ) -> float:
        """
        Pearson correlation between two nodes' mean amplitude time series.

        Args:
            amp_node_a: [n_packets, n_subcarriers] from node A.
            amp_node_b: [n_packets, n_subcarriers] from node B.

        Returns:
            Correlation coefficient (scalar).
        """
        corr_matrix = np.corrcoef(
            np.mean(amp_node_a, axis=1),
            np.mean(amp_node_b, axis=1),
        )
        return float(corr_matrix[0, 1])
