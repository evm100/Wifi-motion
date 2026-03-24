"""
udp_receiver.py — Asyncio UDP receiver for CSI binary packets.

Implements asyncio.DatagramProtocol. Passes raw bytes to a user-supplied
callback (typically packet_parser.parse_packet → aligner).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class CSIDatagramProtocol(asyncio.DatagramProtocol):
    """Asyncio datagram protocol that forwards raw packets to a callback."""

    def __init__(self, on_packet: Callable[[bytes, tuple[str, int]], None]) -> None:
        self.on_packet = on_packet
        self.transport: asyncio.DatagramTransport | None = None
        self.packets_received: int = 0
        self.bytes_received: int = 0

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self.transport = transport
        logger.info("UDP receiver ready on %s", transport.get_extra_info("sockname"))

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self.packets_received += 1
        self.bytes_received += len(data)
        self.on_packet(data, addr)

    def error_received(self, exc: Exception) -> None:
        logger.error("UDP error: %s", exc)

    def connection_lost(self, exc: Exception | None) -> None:
        if exc:
            logger.error("UDP connection lost: %s", exc)
        else:
            logger.info("UDP receiver closed")


async def start_udp_receiver(
    on_packet: Callable[[bytes, tuple[str, int]], None],
    host: str = "0.0.0.0",
    port: int = 5005,
    loop: asyncio.AbstractEventLoop | None = None,
) -> tuple[asyncio.DatagramTransport, CSIDatagramProtocol]:
    """
    Create and start the UDP receiver.

    Args:
        on_packet: Callback invoked with (raw_bytes, (sender_ip, sender_port))
                   for each received datagram.
        host: Bind address (default all interfaces).
        port: UDP port (default 5005).
        loop: Event loop (default: running loop).

    Returns:
        (transport, protocol) tuple for lifecycle management.
    """
    if loop is None:
        loop = asyncio.get_running_loop()

    transport, protocol = await loop.create_datagram_endpoint(
        lambda: CSIDatagramProtocol(on_packet),
        local_addr=(host, port),
    )
    logger.info("Listening for CSI packets on %s:%d", host, port)
    return transport, protocol
