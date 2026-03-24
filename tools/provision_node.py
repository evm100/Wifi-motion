#!/usr/bin/env python3
"""
provision_node.py — Write NVS configuration to an ESP32-S3 node via serial.

Generates an NVS partition binary with node configuration (node_id, WiFi
credentials, target IP, etc.) and flashes it to the connected ESP32.

Usage:
    python tools/provision_node.py --port /dev/ttyUSB0 --node-id 1
    python tools/provision_node.py --port /dev/ttyUSB1 --node-id 2 --ssid MyAP --password secret
    python tools/provision_node.py --port /dev/ttyUSB0 --node-id 0 --role tx
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Default NVS values matching Kconfig defaults
DEFAULTS = {
    "node_id": 1,
    "target_ip": "192.168.4.1",
    "target_port": 5005,
    "wifi_ssid": "CSI_AP",
    "wifi_password": "csi12345",
    "wifi_channel": 6,
    "tx_rate_hz": 100,
    "tx_mac": "FF:FF:FF:FF:FF:FF",
}

# NVS partition offset (must match partitions.csv)
NVS_OFFSET = "0x9000"
NVS_SIZE = "0x6000"  # 24 KB


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Provision ESP32-S3 node with NVS configuration"
    )
    parser.add_argument(
        "--port", type=str, required=True,
        help="Serial port (e.g., /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--role", type=str, choices=["tx", "rx"], default="rx",
        help="Node role: tx or rx (default: rx)"
    )
    parser.add_argument(
        "--node-id", type=int, default=DEFAULTS["node_id"],
        help=f"Node ID (1-255, default: {DEFAULTS['node_id']})"
    )
    parser.add_argument(
        "--target-ip", type=str, default=DEFAULTS["target_ip"],
        help=f"Pi IP for UDP streaming (default: {DEFAULTS['target_ip']})"
    )
    parser.add_argument(
        "--target-port", type=int, default=DEFAULTS["target_port"],
        help=f"UDP port (default: {DEFAULTS['target_port']})"
    )
    parser.add_argument(
        "--ssid", type=str, default=DEFAULTS["wifi_ssid"],
        help=f"WiFi SSID (default: {DEFAULTS['wifi_ssid']})"
    )
    parser.add_argument(
        "--password", type=str, default=DEFAULTS["wifi_password"],
        help=f"WiFi password (default: {DEFAULTS['wifi_password']})"
    )
    parser.add_argument(
        "--channel", type=int, default=DEFAULTS["wifi_channel"],
        help=f"WiFi channel (default: {DEFAULTS['wifi_channel']})"
    )
    parser.add_argument(
        "--tx-rate", type=int, default=DEFAULTS["tx_rate_hz"],
        help=f"TX rate in Hz, TX node only (default: {DEFAULTS['tx_rate_hz']})"
    )
    parser.add_argument(
        "--tx-mac", type=str, default=DEFAULTS["tx_mac"],
        help=f"TX MAC address filter, RX node only (default: {DEFAULTS['tx_mac']})"
    )
    parser.add_argument(
        "--baud", type=int, default=460800,
        help="Serial baud rate (default: 460800)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate NVS CSV and binary but don't flash"
    )
    return parser.parse_args()


def generate_nvs_csv(args: argparse.Namespace) -> str:
    """Generate NVS partition CSV content from args."""
    rows = [
        ["key", "type", "encoding", "value"],
        ["csi_config", "namespace", "", ""],
        ["node_id", "data", "u8", str(args.node_id)],
        ["role", "data", "string", args.role],
        ["target_ip", "data", "string", args.target_ip],
        ["target_port", "data", "u16", str(args.target_port)],
        ["wifi_ssid", "data", "string", args.ssid],
        ["wifi_pass", "data", "string", args.password],
        ["wifi_channel", "data", "u8", str(args.channel)],
        ["tx_rate_hz", "data", "u16", str(args.tx_rate)],
        ["tx_mac", "data", "string", args.tx_mac],
    ]

    lines = []
    for row in rows:
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def find_nvs_partition_gen() -> str:
    """Find the nvs_partition_gen.py tool from ESP-IDF."""
    # Check IDF_PATH environment variable
    idf_path = os.environ.get("IDF_PATH", "")
    if idf_path:
        tool = Path(idf_path) / "components" / "nvs_flash" / "nvs_partition_generator" / "nvs_partition_gen.py"
        if tool.exists():
            return str(tool)

    # Try common paths
    for base in [Path.home() / "esp" / "esp-idf", Path("/opt/esp-idf")]:
        tool = base / "components" / "nvs_flash" / "nvs_partition_generator" / "nvs_partition_gen.py"
        if tool.exists():
            return str(tool)

    # Fall back to PATH
    return "nvs_partition_gen.py"


def main() -> None:
    args = parse_args()

    print(f"Provisioning {args.role.upper()} node (ID={args.node_id}) on {args.port}")
    print(f"  WiFi: {args.ssid} (channel {args.channel})")
    print(f"  Target: {args.target_ip}:{args.target_port}")
    if args.role == "tx":
        print(f"  TX rate: {args.tx_rate} Hz")
    else:
        print(f"  TX MAC filter: {args.tx_mac}")

    # Generate NVS CSV
    csv_content = generate_nvs_csv(args)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "nvs_config.csv"
        bin_path = Path(tmpdir) / "nvs_config.bin"

        csv_path.write_text(csv_content)
        print(f"\nNVS CSV generated:")
        print(csv_content)

        if args.dry_run:
            print("Dry run — not flashing.")
            return

        # Generate NVS binary using nvs_partition_gen.py
        nvs_gen = find_nvs_partition_gen()
        print(f"Generating NVS binary with: {nvs_gen}")

        gen_cmd = [
            sys.executable, nvs_gen, "generate",
            str(csv_path), str(bin_path), NVS_SIZE,
        ]

        result = subprocess.run(gen_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error generating NVS binary:\n{result.stderr}")
            sys.exit(1)
        print(f"NVS binary: {bin_path} ({bin_path.stat().st_size} bytes)")

        # Flash NVS partition using esptool
        print(f"\nFlashing NVS partition to {args.port} at offset {NVS_OFFSET}...")
        flash_cmd = [
            sys.executable, "-m", "esptool",
            "--chip", "esp32s3",
            "--port", args.port,
            "--baud", str(args.baud),
            "write_flash", NVS_OFFSET, str(bin_path),
        ]

        result = subprocess.run(flash_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error flashing NVS:\n{result.stderr}")
            sys.exit(1)

        print(result.stdout.split("\n")[-3] if result.stdout else "")
        print(f"\nNode provisioned successfully: {args.role.upper()} "
              f"(ID={args.node_id}) on {args.port}")


if __name__ == "__main__":
    main()
