"""
test_parser.py — Tests for CSI packet parsing.

Constructs valid binary packets by hand and verifies the parser extracts
correct fields, validates magic, and converts I/Q to complex correctly.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from proto.csi_packet import HEADER_FMT, HEADER_SIZE, CSI_MAGIC, CSI_PROTOCOL_VER
from proto.constants import ALL_VALID, N_VALID_SUBCARRIERS
from edge.aggregator.packet_parser import (
    parse_packet,
    parse_payload_to_complex,
    extract_valid_subcarriers,
    CSIPacket,
)


def build_raw_packet(
    node_id: int = 1,
    n_subcarriers: int = 128,
    rssi: int = -45,
    noise_floor: int = -90,
    channel: int = 6,
    flags: int = 0x01,
    seq_num: int = 1000,
    timestamp_us: int = 5_000_000,
    iq_pairs: list[tuple[int, int]] | None = None,
) -> bytes:
    """Build a valid raw packet from scratch."""
    header = struct.pack(
        HEADER_FMT,
        CSI_MAGIC,
        CSI_PROTOCOL_VER,
        node_id,
        n_subcarriers,
        rssi,
        noise_floor,
        channel,
        flags,
        seq_num,
        timestamp_us,
    )
    if iq_pairs is not None:
        payload = b""
        for imag, real in iq_pairs:
            payload += struct.pack("bb", imag, real)
    else:
        # Generate deterministic I/Q data: imag=i%127, real=(i*3)%127
        payload = b""
        for i in range(n_subcarriers):
            imag = (i % 127) if i % 2 == 0 else -(i % 127)
            real = ((i * 3) % 127) if i % 3 == 0 else -((i * 3) % 127)
            payload += struct.pack("bb", imag, real)

    return header + payload


class TestParseValidPacket:
    def test_basic_fields(self):
        raw = build_raw_packet(
            node_id=2, seq_num=42, rssi=-50, timestamp_us=123456
        )
        pkt = parse_packet(raw)
        assert pkt.node_id == 2
        assert pkt.seq_num == 42
        assert pkt.rssi == -50
        assert pkt.timestamp_us == 123456
        assert pkt.channel == 6
        assert pkt.flags == 0x01
        assert pkt.n_subcarriers_raw == 128

    def test_csi_complex_shape(self):
        raw = build_raw_packet()
        pkt = parse_packet(raw)
        assert pkt.csi_complex.shape == (N_VALID_SUBCARRIERS,)
        assert pkt.csi_complex.dtype == np.complex64

    def test_amplitude_shape(self):
        raw = build_raw_packet()
        pkt = parse_packet(raw)
        assert pkt.amplitude.shape == (N_VALID_SUBCARRIERS,)
        assert np.all(pkt.amplitude >= 0)

    def test_phase_shape(self):
        raw = build_raw_packet()
        pkt = parse_packet(raw)
        assert pkt.phase.shape == (N_VALID_SUBCARRIERS,)
        assert np.all(pkt.phase >= -np.pi)
        assert np.all(pkt.phase <= np.pi)

    def test_all_node_ids(self):
        for nid in [1, 2, 3]:
            raw = build_raw_packet(node_id=nid)
            pkt = parse_packet(raw)
            assert pkt.node_id == nid


class TestInvalidPackets:
    def test_bad_magic_raises(self):
        raw = bytearray(build_raw_packet())
        raw[0:4] = struct.pack("<I", 0xDEADBEEF)
        with pytest.raises(ValueError, match="Bad magic"):
            parse_packet(bytes(raw))

    def test_truncated_header_raises(self):
        with pytest.raises(ValueError, match="Need at least"):
            parse_packet(b"\x00" * 10)

    def test_truncated_payload_raises(self):
        header_only = build_raw_packet()[:HEADER_SIZE]
        with pytest.raises(ValueError, match="Payload too short"):
            parse_packet(header_only)


class TestIQConversion:
    def test_known_values(self):
        """Verify [imag, real] -> real + j*imag conversion."""
        # 4 subcarriers with known I/Q values
        iq_pairs = [
            (10, 20),   # imag=10, real=20 → 20 + 10j
            (-5, 15),   # -5+15j → 15 - 5j
            (0, 0),     # 0 + 0j
            (127, -128),  # edge values
        ]
        payload = b""
        for imag, real in iq_pairs:
            payload += struct.pack("bb", imag, real)

        csi = parse_payload_to_complex(payload, n_subcarriers=4)
        assert len(csi) == 4
        assert csi[0] == pytest.approx(20 + 10j)
        assert csi[1] == pytest.approx(15 - 5j)
        assert csi[2] == pytest.approx(0 + 0j)
        assert csi[3].real == pytest.approx(-128)
        assert csi[3].imag == pytest.approx(127)

    def test_amplitude_from_iq(self):
        """Amplitude should be sqrt(real^2 + imag^2)."""
        iq_pairs = [(3, 4)]  # |4 + 3j| = 5
        payload = struct.pack("bb", 3, 4)
        csi = parse_payload_to_complex(payload, n_subcarriers=1)
        assert np.abs(csi[0]) == pytest.approx(5.0)


class TestValidSubcarrierExtraction:
    def test_extracts_108(self):
        full = np.arange(128, dtype=np.complex64)
        valid = extract_valid_subcarriers(full)
        assert len(valid) == N_VALID_SUBCARRIERS
        # Check specific known indices
        assert valid[0] == full[ALL_VALID[0]]
        assert valid[-1] == full[ALL_VALID[-1]]
