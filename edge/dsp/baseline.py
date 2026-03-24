"""
baseline.py — Adaptive baseline subtraction for static component removal.

Uses exponential moving average (EMA) with two alpha rates:
- fast_alpha (0.1) during calibration to quickly converge on the static channel.
- slow_alpha (0.001) during operation to track environmental drift.
"""

from __future__ import annotations

import numpy as np


class AdaptiveBaseline:
    """
    EMA-based baseline for isolating the dynamic (human) component of CSI.

    Calibration phase: feed empty-room CSI until n_required samples collected.
    Operation phase: subtract baseline, slowly adapt when signal energy is low.
    """

    def __init__(
        self,
        fast_alpha: float = 0.1,
        slow_alpha: float = 0.001,
    ) -> None:
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.alpha = fast_alpha
        self.baseline: np.ndarray | None = None
        self.calibrated: bool = False
        self.calibration_count: int = 0

    def calibrate(self, amplitude: np.ndarray, n_required: int = 300) -> bool:
        """
        Feed one empty-room amplitude sample during calibration.

        Call repeatedly until returns True.

        Args:
            amplitude: [n_subcarriers] amplitude vector for one packet.
            n_required: Number of samples needed to complete calibration.

        Returns:
            True when calibration is complete.
        """
        if self.baseline is None:
            self.baseline = amplitude.copy().astype(np.float64)
        else:
            self.baseline = (1 - self.alpha) * self.baseline + self.alpha * amplitude

        self.calibration_count += 1
        if self.calibration_count >= n_required:
            self.alpha = self.slow_alpha
            self.calibrated = True
            return True
        return False

    def remove_static(self, amplitude: np.ndarray) -> np.ndarray:
        """
        Remove static component and return dynamic (human) signal.

        Also slowly adapts the baseline when signal energy is low
        (indicating a mostly-static scene), to track environmental drift
        like furniture movement or temperature changes.

        Args:
            amplitude: [n_subcarriers] amplitude vector.

        Returns:
            dynamic: amplitude - baseline (the human-induced component).
        """
        if self.baseline is None:
            self.baseline = amplitude.copy().astype(np.float64)
            return np.zeros_like(amplitude)

        dynamic = amplitude - self.baseline

        # Slow update when signal is small (likely static environment)
        signal_energy = np.mean(np.abs(dynamic))
        baseline_mean = np.mean(self.baseline)
        if baseline_mean > 0 and signal_energy < baseline_mean * 0.1:
            self.baseline = (
                (1 - self.slow_alpha) * self.baseline + self.slow_alpha * amplitude
            )

        return dynamic
