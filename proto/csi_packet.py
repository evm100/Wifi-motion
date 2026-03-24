"""
csi_packet.py — Python mirror of csi_packet.h binary packet format.

Provides CSIPacketHeader dataclass with from_bytes() / to_bytes() for
serialization compatible with the C struct.
"""

import struct
from dataclasses import dataclass

# Little-endian: uint32 magic, uint8 version, uint8 node_id, uint16 n_subcarriers,
#                int8 rssi, int8 noise_floor, uint8 channel, uint8 flags,
#                uint32 seq_num, uint32 timestamp_us
HEADER_FMT = "<IBBHbbBBII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 20, f"Header size mismatch: {HEADER_SIZE} != 20"

CSI_MAGIC = 0xC5110001
CSI_PROTOCOL_VER = 1

CSI_FLAG_HAS_HTLTF = 1 << 0
CSI_FLAG_FIRST_INVALID = 1 << 1


@dataclass
class CSIPacketHeader:
    magic: int = CSI_MAGIC
    version: int = CSI_PROTOCOL_VER
    node_id: int = 0
    n_subcarriers: int = 0
    rssi: int = 0
    noise_floor: int = 0
    channel: int = 6
    flags: int = 0
    seq_num: int = 0
    timestamp_us: int = 0

    def to_bytes(self) -> bytes:
        return struct.pack(
            HEADER_FMT,
            self.magic,
            self.version,
            self.node_id,
            self.n_subcarriers,
            self.rssi,
            self.noise_floor,
            self.channel,
            self.flags,
            self.seq_num,
            self.timestamp_us,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "CSIPacketHeader":
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Need at least {HEADER_SIZE} bytes, got {len(data)}")
        fields = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
        magic = fields[0]
        if magic != CSI_MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X}, expected 0x{CSI_MAGIC:08X}")
        return cls(
            magic=fields[0],
            version=fields[1],
            node_id=fields[2],
            n_subcarriers=fields[3],
            rssi=fields[4],
            noise_floor=fields[5],
            channel=fields[6],
            flags=fields[7],
            seq_num=fields[8],
            timestamp_us=fields[9],
        )
