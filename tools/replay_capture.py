#!/usr/bin/env python3
"""
replay_capture.py — Replay recorded CSI capture files over UDP.

Reads a length-prefixed binary capture file (from collect_data.py),
extracts timestamps from CSI packet headers, and replays packets to
localhost UDP at the original inter-packet timing.

Usage:
    python tools/replay_capture.py recording.bin
    python tools/replay_capture.py recording.bin --host 127.0.0.1 --port 5005 --speed 2.0
"""

import argparse
import socket
import struct
import sys
import time
from pathlib import Path

# Add repo root to path so proto/ is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from proto.csi_packet import CSIPacketHeader, HEADER_SIZE
from proto.constants import UDP_PORT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a recorded CSI binary capture over UDP"
    )
    parser.add_argument(
        "capture_file", type=str,
        help="Path to the .bin capture file (length-prefixed format)"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Target host to send UDP packets (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=UDP_PORT,
        help=f"Target UDP port (default: {UDP_PORT})"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0, 2.0 = double speed)"
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop the capture file continuously"
    )
    return parser.parse_args()


def read_packets(capture_file: Path) -> list[tuple[int, bytes]]:
    """
    Read all packets from a length-prefixed binary file.

    Returns list of (timestamp_us, raw_bytes) tuples.
    """
    packets = []
    with open(capture_file, "rb") as f:
        while True:
            length_bytes = f.read(2)
            if len(length_bytes) < 2:
                break
            pkt_len = struct.unpack("<H", length_bytes)[0]
            data = f.read(pkt_len)
            if len(data) < pkt_len:
                break
            try:
                hdr = CSIPacketHeader.from_bytes(data)
                packets.append((hdr.timestamp_us, data))
            except ValueError:
                continue
    return packets


def replay(
    packets: list[tuple[int, bytes]],
    sock: socket.socket,
    dest: tuple[str, int],
    speed: float,
) -> int:
    """
    Replay packets with original timing. Returns count of packets sent.

    Uses the microsecond timestamps from packet headers to determine
    inter-packet delays. Handles timestamp wraparound (32-bit us counter).
    """
    if not packets:
        return 0

    sent = 0
    prev_ts = packets[0][0]
    wall_start = time.monotonic()

    for ts_us, data in packets:
        # Compute delay from timestamp delta
        delta_us = (ts_us - prev_ts) & 0xFFFFFFFF  # handle 32-bit wrap
        if delta_us > 1_000_000:
            # Cap at 1 second to handle gaps or wraps gracefully
            delta_us = 10_000  # assume 100 Hz nominal

        delay_s = (delta_us / 1_000_000.0) / speed

        # Busy-wait for precise timing
        target_time = wall_start + delay_s * (sent + 1) if sent > 0 else wall_start
        if sent > 0:
            # Calculate cumulative target to avoid drift
            elapsed_target = delay_s
            now = time.monotonic()
            sleep_time = target_time - now
            if sleep_time > 0.001:
                time.sleep(sleep_time - 0.001)
            while time.monotonic() < target_time:
                pass

        sock.sendto(data, dest)
        sent += 1
        prev_ts = ts_us

        if sent % 100 == 0:
            elapsed = time.monotonic() - wall_start
            rate = sent / elapsed if elapsed > 0 else 0
            print(f"  Sent {sent}/{len(packets)} packets ({rate:.1f} pkt/s)",
                  end="\r", flush=True)

    return sent


def main() -> None:
    args = parse_args()
    capture_file = Path(args.capture_file)

    if not capture_file.exists():
        print(f"Error: capture file not found: {capture_file}")
        sys.exit(1)

    print(f"Loading capture: {capture_file}")
    packets = read_packets(capture_file)
    if not packets:
        print("Error: no valid packets found in capture file")
        sys.exit(1)

    # Compute capture duration from timestamps
    first_ts = packets[0][0]
    last_ts = packets[-1][0]
    duration_us = (last_ts - first_ts) & 0xFFFFFFFF
    duration_s = duration_us / 1_000_000.0

    print(f"Loaded {len(packets)} packets ({duration_s:.2f}s capture)")
    print(f"Replaying to {args.host}:{args.port} at {args.speed}x speed")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.host, args.port)

    try:
        iteration = 0
        while True:
            iteration += 1
            if args.loop:
                print(f"\n--- Loop iteration {iteration} ---")

            sent = replay(packets, sock, dest, args.speed)
            elapsed = duration_s / args.speed
            print(f"\n  Replay complete: {sent} packets in ~{elapsed:.1f}s")

            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nReplay stopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
