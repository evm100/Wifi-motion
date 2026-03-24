#!/usr/bin/env python3
"""
collect_data.py — Orchestrate labeled CSI data collection sessions.

Prompts the operator for activity names, provides countdown timers,
records raw CSI packets from UDP to timestamped binary files, and saves
metadata (activity, timestamps, node count) as JSON sidecars.

Usage:
    python tools/collect_data.py --output-dir data/sessions
    python tools/collect_data.py --activities walking,sitting,standing --duration-per-activity 30 --repetitions 3
    python tools/collect_data.py --config collection_config.yaml
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to path so proto/ is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from proto.csi_packet import CSIPacketHeader, HEADER_SIZE
from proto.constants import UDP_PORT, CSI_PACKET_SIZE

DEFAULT_ACTIVITIES = ["empty", "walking", "sitting", "standing", "falling", "gesture", "breathing"]
DEFAULT_DURATION = 30  # seconds per activity
DEFAULT_REPETITIONS = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate labeled CSI data collection sessions"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML config file for collection parameters"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/sessions",
        help="Output directory for captures (default: data/sessions)"
    )
    parser.add_argument(
        "--activities", type=str, default=None,
        help="Comma-separated list of activity names (default: all 7 classes)"
    )
    parser.add_argument(
        "--duration-per-activity", type=float, default=DEFAULT_DURATION,
        help=f"Seconds to record per activity (default: {DEFAULT_DURATION})"
    )
    parser.add_argument(
        "--repetitions", type=int, default=DEFAULT_REPETITIONS,
        help=f"Repetitions per activity (default: {DEFAULT_REPETITIONS})"
    )
    parser.add_argument(
        "--port", type=int, default=UDP_PORT,
        help=f"UDP port to listen on (default: {UDP_PORT})"
    )
    parser.add_argument(
        "--countdown", type=int, default=5,
        help="Countdown seconds before each recording (default: 5)"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load collection config from a YAML file."""
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def countdown_timer(seconds: int, label: str) -> None:
    """Display a countdown in the terminal."""
    print(f"\n  Prepare for: {label}")
    for i in range(seconds, 0, -1):
        print(f"  Starting in {i}...", end="\r", flush=True)
        time.sleep(1)
    print(f"  RECORDING: {label}       ")


def record_activity(
    sock: socket.socket,
    duration: float,
    activity: str,
    output_dir: Path,
    session_id: str,
    rep: int,
) -> dict:
    """
    Record raw CSI packets for a given duration.

    Returns metadata dict describing the capture.
    """
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_name = f"{session_id}_{activity}_rep{rep:02d}_{timestamp_str}"
    bin_path = output_dir / f"{base_name}.bin"
    meta_path = output_dir / f"{base_name}.json"

    node_ids_seen: set[int] = set()
    packet_count = 0
    start_time = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()

    with open(bin_path, "wb") as f:
        while True:
            elapsed = time.monotonic() - start_time
            if elapsed >= duration:
                break

            remaining = duration - elapsed
            # Use short timeout so we can check elapsed
            sock.settimeout(min(0.1, remaining))
            try:
                data, _ = sock.recvfrom(CSI_PACKET_SIZE + 64)
            except socket.timeout:
                continue

            # Validate packet before writing
            try:
                hdr = CSIPacketHeader.from_bytes(data)
                node_ids_seen.add(hdr.node_id)
            except ValueError:
                continue

            # Write length-prefixed raw packet: [uint16_le length][raw bytes]
            length = len(data)
            f.write(length.to_bytes(2, "little"))
            f.write(data)
            packet_count += 1

            # Progress indicator every 100 packets
            if packet_count % 100 == 0:
                print(f"  {packet_count} packets ({elapsed:.1f}s)", end="\r", flush=True)

    end_utc = datetime.now(timezone.utc).isoformat()
    actual_duration = time.monotonic() - start_time

    metadata = {
        "session_id": session_id,
        "activity": activity,
        "repetition": rep,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "duration_seconds": round(actual_duration, 3),
        "packet_count": packet_count,
        "node_ids": sorted(node_ids_seen),
        "node_count": len(node_ids_seen),
        "binary_file": bin_path.name,
        "binary_format": "length_prefixed",
        "binary_format_detail": "uint16_le packet_length followed by raw UDP payload",
        "udp_port": sock.getsockname()[1],
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Recorded {packet_count} packets in {actual_duration:.1f}s "
          f"from {len(node_ids_seen)} node(s)")
    print(f"  -> {bin_path.name}")

    return metadata


def main() -> None:
    args = parse_args()

    # Load activities from config, CLI args, or defaults
    activities = DEFAULT_ACTIVITIES
    duration = args.duration_per_activity
    repetitions = args.repetitions

    if args.config:
        cfg = load_config(args.config)
        activities = cfg.get("activities", activities)
        duration = cfg.get("duration_per_activity", duration)
        repetitions = cfg.get("repetitions", repetitions)

    if args.activities:
        activities = [a.strip() for a in args.activities.split(",")]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Session ID from timestamp
    session_id = datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")

    # Set up UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", args.port))
    print(f"Data collection session: {session_id}")
    print(f"Listening on UDP :{args.port}")
    print(f"Activities: {activities}")
    print(f"Duration per activity: {duration}s")
    print(f"Repetitions: {repetitions}")
    print(f"Output: {output_dir}")

    all_metadata = []

    try:
        for activity in activities:
            for rep in range(1, repetitions + 1):
                label = f"{activity} (rep {rep}/{repetitions})"

                # Interactive prompt
                input(f"\nPress Enter when ready for: {label}")
                countdown_timer(args.countdown, label)

                meta = record_activity(
                    sock, duration, activity, output_dir, session_id, rep
                )
                all_metadata.append(meta)

    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user.")
    finally:
        sock.close()

    # Write session summary
    summary_path = output_dir / f"{session_id}_summary.json"
    summary = {
        "session_id": session_id,
        "total_activities_recorded": len(all_metadata),
        "total_packets": sum(m["packet_count"] for m in all_metadata),
        "captures": all_metadata,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSession complete. Summary: {summary_path}")
    print(f"Total captures: {len(all_metadata)}, "
          f"Total packets: {summary['total_packets']}")


if __name__ == "__main__":
    main()
