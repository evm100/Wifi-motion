#!/usr/bin/env python3
"""
visualize_csi.py — Real-time CSI amplitude heatmap visualization.

Listens on UDP port 5005 for CSI packets from ESP32 RX nodes,
parses them using proto/, and plots a live amplitude heatmap
(subcarriers x time) with one subplot per node.

Usage:
    python tools/visualize_csi.py
    python tools/visualize_csi.py --port 5005 --duration 60 --save capture.png
"""

import argparse
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add repo root to path so proto/ is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from proto.csi_packet import CSIPacketHeader, HEADER_SIZE
from proto.constants import (
    ALL_VALID,
    N_VALID_SUBCARRIERS,
    UDP_PORT,
    N_RX_NODES,
    CSI_PACKET_SIZE,
)

# Display window: how many time steps to show
DISPLAY_WINDOW = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time CSI amplitude heatmap visualization"
    )
    parser.add_argument(
        "--port", type=int, default=UDP_PORT,
        help=f"UDP port to listen on (default: {UDP_PORT})"
    )
    parser.add_argument(
        "--duration", type=float, default=0,
        help="Duration in seconds (0 = indefinite, default: 0)"
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Save final figure to this path (e.g., capture.png)"
    )
    return parser.parse_args()


def parse_csi_amplitude(data: bytes) -> tuple[int, np.ndarray]:
    """Parse a raw UDP packet, return (node_id, amplitude_array)."""
    header = CSIPacketHeader.from_bytes(data)
    payload = data[HEADER_SIZE:]
    expected = header.n_subcarriers * 2
    if len(payload) < expected:
        raise ValueError("Payload too short")
    raw = np.frombuffer(payload[:expected], dtype=np.int8)
    imag = raw[0::2].astype(np.float32)
    real = raw[1::2].astype(np.float32)
    csi_all = np.sqrt(real ** 2 + imag ** 2)
    return header.node_id, csi_all[ALL_VALID]


def main() -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    args = parse_args()

    # Set up UDP socket (non-blocking for animation loop)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    sock.setblocking(False)
    print(f"Listening for CSI packets on UDP :{args.port}")

    # Per-node amplitude history buffers
    node_ids = list(range(1, N_RX_NODES + 1))
    buffers: dict[int, np.ndarray] = {
        nid: np.zeros((N_VALID_SUBCARRIERS, DISPLAY_WINDOW), dtype=np.float32)
        for nid in node_ids
    }
    pkt_counts: dict[int, int] = defaultdict(int)

    # Set up figure with one subplot per node
    fig, axes = plt.subplots(
        N_RX_NODES, 1, figsize=(12, 3 * N_RX_NODES), sharex=True
    )
    if N_RX_NODES == 1:
        axes = [axes]

    images = {}
    for ax, nid in zip(axes, node_ids):
        im = ax.imshow(
            buffers[nid],
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=30,
            interpolation="nearest",
        )
        ax.set_ylabel(f"Node {nid}\nSubcarrier")
        fig.colorbar(im, ax=ax, label="Amplitude")
        images[nid] = im

    axes[-1].set_xlabel("Time (packets)")
    fig.suptitle("Real-Time CSI Amplitude Heatmap")
    fig.tight_layout()

    start_time = time.monotonic()

    def update(frame: int) -> list:
        # Drain all available packets from the socket
        while True:
            try:
                data, _ = sock.recvfrom(CSI_PACKET_SIZE + 64)
            except BlockingIOError:
                break
            try:
                node_id, amplitude = parse_csi_amplitude(data)
            except (ValueError, IndexError):
                continue
            if node_id not in buffers:
                continue
            # Shift buffer left and append new column
            buffers[node_id][:, :-1] = buffers[node_id][:, 1:]
            buffers[node_id][:, -1] = amplitude
            pkt_counts[node_id] += 1

        # Update images
        updated = []
        for nid in node_ids:
            images[nid].set_data(buffers[nid])
            updated.append(images[nid])

        # Check duration
        if args.duration > 0:
            elapsed = time.monotonic() - start_time
            if elapsed >= args.duration:
                ani.event_source.stop()
                if args.save:
                    fig.savefig(args.save, dpi=150, bbox_inches="tight")
                    print(f"Saved figure to {args.save}")
                total = sum(pkt_counts.values())
                print(f"Received {total} packets in {elapsed:.1f}s")
                for nid in node_ids:
                    print(f"  Node {nid}: {pkt_counts[nid]} packets")
                plt.close(fig)

        return updated

    ani = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        if args.save and not (args.duration > 0):
            fig.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {args.save}")
        total = sum(pkt_counts.values())
        elapsed = time.monotonic() - start_time
        print(f"Received {total} packets in {elapsed:.1f}s")
        for nid in node_ids:
            print(f"  Node {nid}: {pkt_counts[nid]} packets")


if __name__ == "__main__":
    main()
