"""
test_protocol_compat.py — Verify C/Python protocol agreement.

Tests that csi_packet.py produces and parses headers identical to csi_packet.h.
"""

import struct
import pytest
from csi_packet import (
    CSIPacketHeader,
    HEADER_FMT,
    HEADER_SIZE,
    CSI_MAGIC,
    CSI_PROTOCOL_VER,
    CSI_FLAG_HAS_HTLTF,
    CSI_FLAG_FIRST_INVALID,
)
from constants import (
    N_VALID_SUBCARRIERS,
    N_LLTF,
    N_HTLTF,
    LLTF_VALID,
    HTLTF_VALID,
    ALL_VALID,
    CSI_HEADER_SIZE,
)


class TestHeaderSize:
    def test_header_size_is_20(self):
        assert HEADER_SIZE == 20

    def test_constant_matches_struct(self):
        assert CSI_HEADER_SIZE == HEADER_SIZE

    def test_to_bytes_length(self):
        hdr = CSIPacketHeader()
        assert len(hdr.to_bytes()) == 20


class TestFieldRoundtrip:
    def test_default_values(self):
        hdr = CSIPacketHeader()
        raw = hdr.to_bytes()
        restored = CSIPacketHeader.from_bytes(raw)
        assert restored.magic == CSI_MAGIC
        assert restored.version == CSI_PROTOCOL_VER
        assert restored.node_id == 0
        assert restored.n_subcarriers == 0
        assert restored.rssi == 0
        assert restored.noise_floor == 0
        assert restored.channel == 6
        assert restored.flags == 0
        assert restored.seq_num == 0
        assert restored.timestamp_us == 0

    def test_all_fields(self):
        hdr = CSIPacketHeader(
            magic=CSI_MAGIC,
            version=1,
            node_id=3,
            n_subcarriers=128,
            rssi=-45,
            noise_floor=-90,
            channel=6,
            flags=CSI_FLAG_HAS_HTLTF | CSI_FLAG_FIRST_INVALID,
            seq_num=42857,
            timestamp_us=0xDEADBEEF,
        )
        raw = hdr.to_bytes()
        restored = CSIPacketHeader.from_bytes(raw)
        assert restored.magic == CSI_MAGIC
        assert restored.version == 1
        assert restored.node_id == 3
        assert restored.n_subcarriers == 128
        assert restored.rssi == -45
        assert restored.noise_floor == -90
        assert restored.channel == 6
        assert restored.flags == 0x03
        assert restored.seq_num == 42857
        assert restored.timestamp_us == 0xDEADBEEF

    def test_negative_rssi_roundtrip(self):
        for rssi_val in [-1, -45, -80, -128, 0, 127]:
            hdr = CSIPacketHeader(rssi=rssi_val, noise_floor=rssi_val)
            restored = CSIPacketHeader.from_bytes(hdr.to_bytes())
            assert restored.rssi == rssi_val
            assert restored.noise_floor == rssi_val

    def test_max_seq_num(self):
        hdr = CSIPacketHeader(seq_num=0xFFFFFFFF)
        restored = CSIPacketHeader.from_bytes(hdr.to_bytes())
        assert restored.seq_num == 0xFFFFFFFF

    def test_max_timestamp(self):
        hdr = CSIPacketHeader(timestamp_us=0xFFFFFFFF)
        restored = CSIPacketHeader.from_bytes(hdr.to_bytes())
        assert restored.timestamp_us == 0xFFFFFFFF

    def test_node_ids(self):
        for nid in [1, 2, 3, 255]:
            hdr = CSIPacketHeader(node_id=nid)
            restored = CSIPacketHeader.from_bytes(hdr.to_bytes())
            assert restored.node_id == nid


class TestMagicValidation:
    def test_correct_magic(self):
        hdr = CSIPacketHeader()
        CSIPacketHeader.from_bytes(hdr.to_bytes())  # Should not raise

    def test_wrong_magic_raises(self):
        hdr = CSIPacketHeader()
        raw = bytearray(hdr.to_bytes())
        raw[0:4] = struct.pack("<I", 0xDEADBEEF)
        with pytest.raises(ValueError, match="Bad magic"):
            CSIPacketHeader.from_bytes(bytes(raw))

    def test_zero_magic_raises(self):
        raw = b"\x00" * 20
        with pytest.raises(ValueError, match="Bad magic"):
            CSIPacketHeader.from_bytes(raw)

    def test_truncated_data_raises(self):
        with pytest.raises(ValueError, match="Need at least"):
            CSIPacketHeader.from_bytes(b"\x00" * 10)


class TestSubcarrierConstants:
    def test_lltf_count(self):
        assert N_LLTF == 52

    def test_htltf_count(self):
        assert N_HTLTF == 56

    def test_total_valid(self):
        assert N_VALID_SUBCARRIERS == 108

    def test_all_valid_is_lltf_plus_htltf(self):
        assert ALL_VALID == LLTF_VALID + HTLTF_VALID

    def test_no_overlap(self):
        assert len(set(LLTF_VALID) & set(HTLTF_VALID)) == 0

    def test_indices_in_range(self):
        for idx in ALL_VALID:
            assert 0 <= idx < 128


class TestFlagBits:
    def test_flag_values(self):
        assert CSI_FLAG_HAS_HTLTF == 0x01
        assert CSI_FLAG_FIRST_INVALID == 0x02

    def test_flags_independent(self):
        hdr = CSIPacketHeader(flags=CSI_FLAG_HAS_HTLTF)
        restored = CSIPacketHeader.from_bytes(hdr.to_bytes())
        assert restored.flags & CSI_FLAG_HAS_HTLTF
        assert not (restored.flags & CSI_FLAG_FIRST_INVALID)


class TestByteLayout:
    """Verify exact byte offsets match the C struct layout from CLAUDE.md."""

    def test_field_offsets(self):
        hdr = CSIPacketHeader(
            magic=CSI_MAGIC,
            version=1,
            node_id=2,
            n_subcarriers=128,
            rssi=-45,
            noise_floor=-90,
            channel=6,
            flags=0x03,
            seq_num=1000,
            timestamp_us=5000000,
        )
        raw = hdr.to_bytes()
        # Offset 0: magic (4 bytes LE)
        assert struct.unpack_from("<I", raw, 0)[0] == CSI_MAGIC
        # Offset 4: version (1 byte)
        assert raw[4] == 1
        # Offset 5: node_id (1 byte)
        assert raw[5] == 2
        # Offset 6: n_subcarriers (2 bytes LE)
        assert struct.unpack_from("<H", raw, 6)[0] == 128
        # Offset 8: rssi (1 byte signed)
        assert struct.unpack_from("<b", raw, 8)[0] == -45
        # Offset 9: noise_floor (1 byte signed)
        assert struct.unpack_from("<b", raw, 9)[0] == -90
        # Offset 10: channel (1 byte)
        assert raw[10] == 6
        # Offset 11: flags (1 byte)
        assert raw[11] == 0x03
        # Offset 12: seq_num (4 bytes LE)
        assert struct.unpack_from("<I", raw, 12)[0] == 1000
        # Offset 16: timestamp_us (4 bytes LE)
        assert struct.unpack_from("<I", raw, 16)[0] == 5000000
