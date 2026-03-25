"""
monitor.py -- Background UDP listener that tracks per-node health stats.

Runs an asyncio UDP socket alongside the FastAPI app, parses CSI packet
headers, and maintains a dict of per-node stats (packet rate, RSSI,
last-seen time, connection status).
"""

from __future__ import annotations

import asyncio
import struct
import time
from dataclasses import dataclass, field

from proto.csi_packet import CSIPacketHeader, HEADER_SIZE, CSI_MAGIC
from proto.constants import UDP_PORT, N_RX_NODES

# A node is considered disconnected after this many seconds of silence.
NODE_TIMEOUT_S = 3.0

# How often to recompute packet rates.
STATS_INTERVAL_S = 1.0


@dataclass
class NodeStats:
    node_id: int
    last_seen: float = 0.0
    rssi: int = 0
    noise_floor: int = 0
    channel: int = 0
    seq_num: int = 0
    packets_total: int = 0
    # For rate calculation
    _packets_at_last_tick: int = 0
    packets_per_sec: float = 0.0

    @property
    def connected(self) -> bool:
        if self.last_seen == 0.0:
            return False
        return (time.monotonic() - self.last_seen) < NODE_TIMEOUT_S

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "connected": self.connected,
            "rssi": self.rssi,
            "noise_floor": self.noise_floor,
            "channel": self.channel,
            "seq_num": self.seq_num,
            "packets_total": self.packets_total,
            "packets_per_sec": round(self.packets_per_sec, 1),
            "last_seen": round(self.last_seen, 3) if self.last_seen else None,
        }


class NodeMonitor:
    """Tracks stats for all ESP32 RX nodes."""

    def __init__(self, expected_node_ids: list[int] | None = None):
        if expected_node_ids is None:
            expected_node_ids = list(range(1, N_RX_NODES + 1))
        self.nodes: dict[int, NodeStats] = {
            nid: NodeStats(node_id=nid) for nid in expected_node_ids
        }
        self._transport: asyncio.DatagramTransport | None = None
        self._stats_task: asyncio.Task | None = None
        # Subscribers notified on every stats tick (WebSocket push).
        self._subscribers: list[asyncio.Queue] = []

    def handle_packet(self, data: bytes) -> None:
        """Called for every raw UDP datagram. Parse header and update stats."""
        if len(data) < HEADER_SIZE:
            return
        # Quick magic check without full parsing overhead.
        magic = struct.unpack_from("<I", data, 0)[0]
        if magic != CSI_MAGIC:
            return
        try:
            hdr = CSIPacketHeader.from_bytes(data)
        except ValueError:
            return

        nid = hdr.node_id
        if nid not in self.nodes:
            self.nodes[nid] = NodeStats(node_id=nid)

        ns = self.nodes[nid]
        ns.last_seen = time.monotonic()
        ns.rssi = hdr.rssi
        ns.noise_floor = hdr.noise_floor
        ns.channel = hdr.channel
        ns.seq_num = hdr.seq_num
        ns.packets_total += 1

    def get_status(self) -> list[dict]:
        return [ns.to_dict() for ns in sorted(self.nodes.values(), key=lambda n: n.node_id)]

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=4)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def _stats_loop(self) -> None:
        """Periodically compute rates and push to WebSocket subscribers."""
        while True:
            await asyncio.sleep(STATS_INTERVAL_S)
            for ns in self.nodes.values():
                delta = ns.packets_total - ns._packets_at_last_tick
                ns.packets_per_sec = delta / STATS_INTERVAL_S
                ns._packets_at_last_tick = ns.packets_total

            status = self.get_status()
            for q in list(self._subscribers):
                try:
                    q.put_nowait(status)
                except asyncio.QueueFull:
                    # Drop stale update; subscriber is behind.
                    pass

    async def start(self, host: str = "0.0.0.0", port: int = UDP_PORT) -> None:
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _MonitorProtocol(self.handle_packet),
            local_addr=(host, port),
        )
        self._stats_task = asyncio.create_task(self._stats_loop())

    async def stop(self) -> None:
        if self._stats_task:
            self._stats_task.cancel()
        if self._transport:
            self._transport.close()


class _MonitorProtocol(asyncio.DatagramProtocol):
    def __init__(self, on_packet):
        self._on_packet = on_packet

    def datagram_received(self, data: bytes, addr) -> None:
        self._on_packet(data)

    def error_received(self, exc: Exception) -> None:
        pass
