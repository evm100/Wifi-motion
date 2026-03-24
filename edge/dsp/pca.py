"""
pca.py — Streaming PCA for multi-node CSI dimensionality reduction.

Concatenates all nodes' amplitudes into a single vector per frame (n_nodes × 108),
standardizes, then reduces to top-K principal components that capture the most
human-motion variance.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import IncrementalPCA


class StreamingCSIPCA:
    """
    Real-time PCA for multi-node CSI dimensionality reduction.

    Calibration: collect samples, compute mean/std, fit IncrementalPCA.
    Operation: standardize + transform each frame to PC scores.
    """

    def __init__(
        self,
        n_nodes: int = 3,
        n_subcarriers: int = 108,
        n_components: int = 20,
        calibration_size: int = 500,
    ) -> None:
        self.n_nodes = n_nodes
        self.n_sc = n_subcarriers
        self.n_components = n_components
        self.input_dim = n_nodes * n_subcarriers

        self.pca = IncrementalPCA(n_components=n_components)
        self.calibration_buffer: list[np.ndarray] = []
        self.calibration_size = calibration_size
        self.calibrated: bool = False

        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.explained_variance_ratio: np.ndarray | None = None

    def _concat_nodes(self, node_amplitudes: dict[int, np.ndarray]) -> np.ndarray:
        """Concatenate per-node amplitudes into single feature vector."""
        return np.concatenate([
            node_amplitudes[i] for i in sorted(node_amplitudes.keys())
        ])

    def add_calibration_frame(
        self, node_amplitudes: dict[int, np.ndarray]
    ) -> bool:
        """
        Add one frame during calibration phase.

        Args:
            node_amplitudes: dict {node_id: amplitude_array[n_sc]}.

        Returns:
            True when calibration_size samples reached and PCA is fitted.
        """
        combined = self._concat_nodes(node_amplitudes)
        self.calibration_buffer.append(combined)

        if len(self.calibration_buffer) >= self.calibration_size:
            data = np.array(self.calibration_buffer)

            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            self.std[self.std < 1e-6] = 1.0  # prevent division by zero

            standardized = (data - self.mean) / self.std
            self.pca.fit(standardized)
            self.explained_variance_ratio = self.pca.explained_variance_ratio_

            self.calibrated = True
            self.calibration_buffer = []
            return True

        return False

    def transform(self, node_amplitudes: dict[int, np.ndarray]) -> np.ndarray | None:
        """
        Transform a single frame to PCA space.

        Args:
            node_amplitudes: dict {node_id: amplitude_array[n_sc]}.

        Returns:
            PC scores array [n_components], or None if not yet calibrated.
        """
        if not self.calibrated:
            return None

        combined = self._concat_nodes(node_amplitudes)
        standardized = (combined - self.mean) / self.std
        return self.pca.transform(standardized.reshape(1, -1))[0]

    def get_top_subcarriers(self, n_top: int = 20) -> list[tuple[int, int, float]]:
        """
        Identify physical subcarriers contributing most to top PCs.

        Returns:
            List of (node_id, subcarrier_index, loading) tuples sorted by
            importance (highest loading first).
        """
        if not self.calibrated:
            return []

        # Sum absolute loadings across top 5 PCs
        n_pcs = min(5, self.pca.components_.shape[0])
        loadings = np.sum(np.abs(self.pca.components_[:n_pcs, :]), axis=0)

        results = []
        for idx in np.argsort(loadings)[::-1][:n_top]:
            node = idx // self.n_sc
            sc = idx % self.n_sc
            results.append((node + 1, sc, float(loadings[idx])))

        return results
