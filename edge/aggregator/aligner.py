"""
aligner.py — Align CSI packets from multiple RX nodes by TX sequence number.

Buffers packets and emits an aligned group when all n_nodes have reported
for the same seq_num. Garbage-collects stale entries to bound memory.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

from .packet_parser import CSIPacket

logger = logging.getLogger(__name__)


@dataclass
class AlignmentStats:
    """Running alignment statistics."""

    groups_emitted: int = 0
    packets_received: int = 0
    packets_dropped: int = 0
    stale_entries_cleaned: int = 0
    last_log_time: float = field(default_factory=time.monotonic)

    @property
    def alignment_rate(self) -> float:
        if self.packets_received == 0:
            return 0.0
        return (self.groups_emitted * self.n_nodes) / self.packets_received

    def __post_init__(self) -> None:
        self.n_nodes = 3  # set by Aligner


class Aligner:
    """
    Buffer CSI packets by TX seq_num and emit aligned groups.

    An aligned group is a dict {node_id: CSIPacket} containing exactly
    one packet from each of the expected n_nodes.
    """

    def __init__(
        self,
        n_nodes: int = 3,
        stale_threshold: int = 50,
        on_aligned: Callable[[dict[int, CSIPacket]], None] | None = None,
        log_interval: float = 5.0,
    ) -> None:
        """
        Args:
            n_nodes: Number of RX nodes expected per group.
            stale_threshold: Sequence numbers older than (max_seq - threshold)
                             are garbage-collected.
            on_aligned: Callback invoked with aligned group dict.
            log_interval: Seconds between stats log messages.
        """
        self.n_nodes = n_nodes
        self.stale_threshold = stale_threshold
        self.on_aligned = on_aligned
        self.log_interval = log_interval

        # OrderedDict preserves insertion order for efficient cleanup
        self._pending: OrderedDict[int, dict[int, CSIPacket]] = OrderedDict()
        self._max_seq: int = 0

        self.stats = AlignmentStats()
        self.stats.n_nodes = n_nodes

    def add_packet(self, packet: CSIPacket) -> dict[int, CSIPacket] | None:
        """
        Add a parsed packet. Returns the aligned group if complete, else None.

        Args:
            packet: Parsed CSI packet with node_id and seq_num.

        Returns:
            dict {node_id: CSIPacket} when all n_nodes arrive, else None.
        """
        seq = packet.seq_num
        node_id = packet.node_id
        self.stats.packets_received += 1

        # Track max sequence number for stale detection
        if seq > self._max_seq:
            self._max_seq = seq

        # Insert into pending buffer
        if seq not in self._pending:
            self._pending[seq] = {}

        group = self._pending[seq]

        if node_id in group:
            # Duplicate packet for this node/seq — keep first
            return None

        group[node_id] = packet

        # Check if group is complete
        result = None
        if len(group) == self.n_nodes:
            result = self._pending.pop(seq)
            self.stats.groups_emitted += 1

            if self.on_aligned is not None:
                self.on_aligned(result)

        # Periodic garbage collection and logging
        self._gc_stale()
        self._maybe_log()

        return result

    def _gc_stale(self) -> None:
        """Remove entries older than stale_threshold behind max_seq."""
        cutoff = self._max_seq - self.stale_threshold
        if cutoff <= 0:
            return

        stale_keys = [seq for seq in self._pending if seq < cutoff]
        for seq in stale_keys:
            dropped_group = self._pending.pop(seq)
            n_dropped = len(dropped_group)
            self.stats.packets_dropped += n_dropped
            self.stats.stale_entries_cleaned += 1

    def _maybe_log(self) -> None:
        """Log alignment stats periodically."""
        now = time.monotonic()
        if now - self.stats.last_log_time >= self.log_interval:
            s = self.stats
            total = s.packets_received
            aligned_pct = (s.groups_emitted * self.n_nodes / total * 100) if total else 0
            dropped_pct = (s.packets_dropped / total * 100) if total else 0
            logger.info(
                "Aligner: %d groups emitted, %d pkts received, "
                "%.1f%% aligned, %.1f%% dropped, %d stale cleaned",
                s.groups_emitted,
                total,
                aligned_pct,
                dropped_pct,
                s.stale_entries_cleaned,
            )
            s.last_log_time = now
