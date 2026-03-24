"""
packet_parser.py — Parse raw UDP datagrams into CSIPacket dataclasses.

Validates magic number, extracts header fields, converts I/Q payload to
complex numpy array, and selects valid subcarriers per proto/constants.py.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add repo root to path so proto/ is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from proto.csi_packet import CSIPacketHeader, HEADER_SIZE, CSI_MAGIC
from proto.constants import ALL_VALID, N_VALID_SUBCARRIERS


@dataclass
class CSIPacket:
    """Parsed CSI packet with extracted complex data and derived arrays."""

    node_id: int
    seq_num: int
    timestamp_us: int
    rssi: int
    noise_floor: int
    channel: int
    flags: int
    n_subcarriers_raw: int
    csi_complex: np.ndarray    # [N_VALID_SUBCARRIERS] complex64
    amplitude: np.ndarray      # [N_VALID_SUBCARRIERS] float32
    phase: np.ndarray          # [N_VALID_SUBCARRIERS] float32


def parse_payload_to_complex(payload: bytes, n_subcarriers: int) -> np.ndarray:
    """
    Convert raw I/Q payload bytes to complex numpy array.

    Each subcarrier is 2 bytes: [imaginary, real] as int8.
    Returns complex64 array of length n_subcarriers.
    """
    expected = n_subcarriers * 2
    if len(payload) < expected:
        raise ValueError(
            f"Payload too short: need {expected} bytes for {n_subcarriers} subcarriers, "
            f"got {len(payload)}"
        )
    raw = np.frombuffer(payload[:expected], dtype=np.int8)
    # Layout: [imag0, real0, imag1, real1, ...]
    imag = raw[0::2].astype(np.float32)
    real = raw[1::2].astype(np.float32)
    return real + 1j * imag


def extract_valid_subcarriers(csi_all: np.ndarray) -> np.ndarray:
    """Select the 108 valid subcarriers from full 128-subcarrier CSI array."""
    return csi_all[ALL_VALID]


def parse_packet(data: bytes) -> CSIPacket:
    """
    Parse a raw UDP datagram into a CSIPacket.

    Validates magic number, extracts header, parses I/Q payload into
    complex array, and selects valid subcarriers.

    Args:
        data: Raw bytes from UDP socket (must be >= HEADER_SIZE bytes).

    Returns:
        CSIPacket with complex CSI, amplitude, and phase for valid subcarriers.

    Raises:
        ValueError: On bad magic, truncated data, or payload too short.
    """
    header = CSIPacketHeader.from_bytes(data)  # validates magic
    payload = data[HEADER_SIZE:]

    csi_all = parse_payload_to_complex(payload, header.n_subcarriers)
    csi_valid = extract_valid_subcarriers(csi_all)

    amplitude = np.abs(csi_valid)
    phase = np.angle(csi_valid)

    return CSIPacket(
        node_id=header.node_id,
        seq_num=header.seq_num,
        timestamp_us=header.timestamp_us,
        rssi=header.rssi,
        noise_floor=header.noise_floor,
        channel=header.channel,
        flags=header.flags,
        n_subcarriers_raw=header.n_subcarriers,
        csi_complex=csi_valid,
        amplitude=amplitude,
        phase=phase,
    )
