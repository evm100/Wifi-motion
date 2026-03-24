"""
phase_sanitizer.py — Phase cleaning algorithms for CSI data.

Three methods:
1. Linear regression (SpotFi-style) — removes CFO/SFO/PDD per packet.
2. Conjugate multiplication — cross-node TX error cancellation.
3. TSFR — time smoothing + frequency rebuild for best phase quality.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def sanitize_phase_linear(
    csi_complex: np.ndarray,
    subcarrier_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove CFO (constant offset) and SFO/PDD (linear slope) from CSI phase
    using least-squares linear regression across subcarriers.

    Args:
        csi_complex: Complex CSI array [n_subcarriers] for one packet.
        subcarrier_indices: Physical subcarrier index array (default: 0..N-1).

    Returns:
        sanitized_complex: Complex CSI with cleaned phase, original amplitude.
        sanitized_phase: Cleaned phase array [n_subcarriers].
    """
    n_sc = len(csi_complex)
    if subcarrier_indices is None:
        subcarrier_indices = np.arange(n_sc)

    raw_phase = np.angle(csi_complex)
    unwrapped = np.unwrap(raw_phase)

    # Least-squares: phase = slope * k + intercept + residual
    k = subcarrier_indices.astype(np.float64)
    A = np.vstack([k, np.ones(n_sc)]).T
    slope, intercept = np.linalg.lstsq(A, unwrapped, rcond=None)[0]

    sanitized_phase = unwrapped - (slope * k + intercept)

    amplitude = np.abs(csi_complex)
    sanitized_complex = amplitude * np.exp(1j * sanitized_phase)

    return sanitized_complex, sanitized_phase


def conjugate_multiply(
    csi_node_a: np.ndarray,
    csi_node_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Conjugate multiplication between two RX nodes' CSI.

    Cancels TX-side phase errors (CFO, SFO, PDD) and preserves the
    differential phase caused by different propagation paths.

    With 3 RX nodes, compute pairs (1,2), (1,3), (2,3) for full
    spatial coverage.

    Args:
        csi_node_a: Complex CSI [n_subcarriers] from RX node A.
        csi_node_b: Complex CSI [n_subcarriers] from RX node B.

    Returns:
        cross_csi: Complex conjugate product H_A * conj(H_B).
        diff_phase: Phase difference (spatial information).
        product_amplitude: |H_A| * |H_B|.
    """
    cross_csi = csi_node_a * np.conj(csi_node_b)
    diff_phase = np.angle(cross_csi)
    product_amplitude = np.abs(cross_csi)

    return cross_csi, diff_phase, product_amplitude


def sanitize_phase_tsfr(
    phase_timeseries: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    """
    TSFR: Time Smoothing and Frequency Rebuild phase sanitization.

    Applies Savitzky-Golay smoothing in time per subcarrier, then re-applies
    linear regression per packet to remove distortions introduced by the
    time-domain filtering.

    Input should already be linear-regression sanitized.

    Args:
        phase_timeseries: Unwrapped phase [n_packets, n_subcarriers].
        window_length: Savitzky-Golay window (odd integer, in packets).
        polyorder: Polynomial order for Savitzky-Golay.

    Returns:
        tsfr_phase: Fully sanitized phase [n_packets, n_subcarriers].
    """
    n_packets, n_sc = phase_timeseries.shape

    # Clamp window_length to data size and ensure odd
    wl = min(window_length, n_packets)
    if wl % 2 == 0:
        wl -= 1
    if wl < 1:
        return phase_timeseries.copy()
    po = min(polyorder, wl - 1)

    # Step 1: time-domain smoothing per subcarrier
    time_smoothed = np.zeros_like(phase_timeseries)
    for sc in range(n_sc):
        time_smoothed[:, sc] = savgol_filter(
            phase_timeseries[:, sc],
            window_length=wl,
            polyorder=po,
        )

    # Step 2: frequency rebuild — linear regression per packet
    tsfr_phase = np.zeros_like(time_smoothed)
    k = np.arange(n_sc, dtype=np.float64)
    A = np.vstack([k, np.ones(n_sc)]).T

    for t in range(n_packets):
        slope, intercept = np.linalg.lstsq(A, time_smoothed[t], rcond=None)[0]
        tsfr_phase[t] = time_smoothed[t] - (slope * k + intercept)

    return tsfr_phase
