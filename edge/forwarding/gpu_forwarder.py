"""
gpu_forwarder.py — Stream processed CSI tensors to RTX 4080 via ZMQ PUB.

The Pi binds as PUB; the GPU server connects as SUB. This decouples the
two so the Pi can run without the GPU being online.
"""

from __future__ import annotations

import logging
import pickle
import time

import numpy as np
import zmq

logger = logging.getLogger(__name__)


class GPUForwarder:
    """
    ZMQ PUB socket for streaming CSI feature tensors to the GPU server.

    Sends pickled dicts with 'tensor' (numpy array) and 'metadata' (dict)
    keys. Non-blocking sends with configurable high-water mark.
    """

    def __init__(
        self,
        gpu_address: str = "tcp://*:5556",
        send_hwm: int = 100,
    ) -> None:
        """
        Args:
            gpu_address: ZMQ bind address (e.g., "tcp://*:5556").
            send_hwm: Send high-water mark — max queued messages before drop.
        """
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, send_hwm)
        self.socket.bind(gpu_address)
        self._tensors_sent: int = 0
        logger.info("GPUForwarder bound to %s (HWM=%d)", gpu_address, send_hwm)

    def send_tensor(
        self,
        tensor: np.ndarray,
        metadata: dict | None = None,
    ) -> bool:
        """
        Send a processed tensor to the GPU server.

        Args:
            tensor: Numpy array (GPU-ready feature tensor).
            metadata: Dict with frame count, motion energy, timestamp, etc.

        Returns:
            True if sent, False if dropped (queue full).
        """
        payload = {
            "tensor": tensor,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }
        try:
            self.socket.send(pickle.dumps(payload), zmq.NOBLOCK)
            self._tensors_sent += 1
            return True
        except zmq.Again:
            logger.warning("ZMQ send queue full — tensor dropped")
            return False

    @property
    def tensors_sent(self) -> int:
        return self._tensors_sent

    def close(self) -> None:
        """Clean shutdown of ZMQ socket and context."""
        self.socket.close()
        self.ctx.term()
        logger.info("GPUForwarder closed (%d tensors sent)", self._tensors_sent)
