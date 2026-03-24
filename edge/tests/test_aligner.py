"""
test_aligner.py — Tests for CSI packet alignment by TX sequence number.

Verifies group emission, stale entry cleanup, duplicate handling, and stats.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edge.aggregator.aligner import Aligner
from edge.aggregator.packet_parser import CSIPacket


def make_packet(node_id: int, seq_num: int) -> CSIPacket:
    """Create a minimal CSIPacket for testing."""
    n_sc = 108
    csi = np.ones(n_sc, dtype=np.complex64)
    return CSIPacket(
        node_id=node_id,
        seq_num=seq_num,
        timestamp_us=seq_num * 10000,
        rssi=-45,
        noise_floor=-90,
        channel=6,
        flags=0x01,
        n_subcarriers_raw=128,
        csi_complex=csi,
        amplitude=np.abs(csi),
        phase=np.angle(csi),
    )


class TestBasicAlignment:
    def test_emits_group_when_all_nodes_arrive(self):
        """Group should be emitted when all 3 nodes report for same seq_num."""
        aligner = Aligner(n_nodes=3)

        result = aligner.add_packet(make_packet(1, seq_num=100))
        assert result is None

        result = aligner.add_packet(make_packet(2, seq_num=100))
        assert result is None

        result = aligner.add_packet(make_packet(3, seq_num=100))
        assert result is not None
        assert set(result.keys()) == {1, 2, 3}
        assert all(pkt.seq_num == 100 for pkt in result.values())

    def test_two_node_mode(self):
        """Should work with configurable n_nodes."""
        aligner = Aligner(n_nodes=2)

        aligner.add_packet(make_packet(1, seq_num=50))
        result = aligner.add_packet(make_packet(2, seq_num=50))
        assert result is not None
        assert len(result) == 2

    def test_out_of_order_arrival(self):
        """Packets arriving in any order should still align correctly."""
        aligner = Aligner(n_nodes=3)

        aligner.add_packet(make_packet(3, seq_num=200))
        aligner.add_packet(make_packet(1, seq_num=200))
        result = aligner.add_packet(make_packet(2, seq_num=200))

        assert result is not None
        assert set(result.keys()) == {1, 2, 3}

    def test_multiple_groups(self):
        """Multiple seq_nums should each emit independently."""
        aligner = Aligner(n_nodes=3)
        emitted = []

        for seq in [10, 11, 12]:
            for node in [1, 2, 3]:
                result = aligner.add_packet(make_packet(node, seq))
                if result is not None:
                    emitted.append(result)

        assert len(emitted) == 3
        seqs = [list(g.values())[0].seq_num for g in emitted]
        assert sorted(seqs) == [10, 11, 12]


class TestIncompleteGroups:
    def test_missing_node_does_not_emit(self):
        """If one node never arrives, the group should not emit."""
        aligner = Aligner(n_nodes=3)

        aligner.add_packet(make_packet(1, seq_num=500))
        result = aligner.add_packet(make_packet(2, seq_num=500))
        assert result is None

    def test_duplicate_node_ignored(self):
        """A second packet from the same node/seq should be ignored."""
        aligner = Aligner(n_nodes=3)

        aligner.add_packet(make_packet(1, seq_num=300))
        aligner.add_packet(make_packet(1, seq_num=300))  # duplicate
        aligner.add_packet(make_packet(2, seq_num=300))
        result = aligner.add_packet(make_packet(3, seq_num=300))

        assert result is not None
        assert result[1].node_id == 1  # first packet kept


class TestStaleCleanup:
    def test_stale_entries_are_removed(self):
        """Entries older than stale_threshold behind max_seq should be cleaned."""
        aligner = Aligner(n_nodes=3, stale_threshold=10)

        # Create an incomplete group at seq=1
        aligner.add_packet(make_packet(1, seq_num=1))
        assert 1 in aligner._pending

        # Push max_seq far ahead
        for node in [1, 2, 3]:
            aligner.add_packet(make_packet(node, seq_num=100))

        # Seq 1 should have been cleaned (100 - 1 > 10)
        assert 1 not in aligner._pending

    def test_recent_entries_preserved(self):
        """Entries within stale_threshold should not be cleaned."""
        aligner = Aligner(n_nodes=3, stale_threshold=50)

        aligner.add_packet(make_packet(1, seq_num=80))
        aligner.add_packet(make_packet(1, seq_num=100))

        # 100 - 80 = 20 < 50, so seq=80 should still be pending
        assert 80 in aligner._pending


class TestStats:
    def test_stats_counting(self):
        aligner = Aligner(n_nodes=3)

        for node in [1, 2, 3]:
            aligner.add_packet(make_packet(node, seq_num=1))

        assert aligner.stats.packets_received == 3
        assert aligner.stats.groups_emitted == 1

    def test_callback_invoked(self):
        """on_aligned callback should fire with the complete group."""
        received = []
        aligner = Aligner(n_nodes=3, on_aligned=received.append)

        for node in [1, 2, 3]:
            aligner.add_packet(make_packet(node, seq_num=42))

        assert len(received) == 1
        assert set(received[0].keys()) == {1, 2, 3}

    def test_drop_counting(self):
        """Stale cleanup should increment packets_dropped."""
        aligner = Aligner(n_nodes=3, stale_threshold=5)

        # Incomplete group at seq=1
        aligner.add_packet(make_packet(1, seq_num=1))
        aligner.add_packet(make_packet(2, seq_num=1))

        # Jump ahead to trigger cleanup
        for node in [1, 2, 3]:
            aligner.add_packet(make_packet(node, seq_num=100))

        assert aligner.stats.packets_dropped == 2  # 2 packets in stale seq=1
        assert aligner.stats.stale_entries_cleaned == 1
