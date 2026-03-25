#!/usr/bin/env bash
# setup.sh — Idempotent setup for Raspberry Pi CSI edge pipeline.
#
# The Pi connects to an iPhone hotspot as a WiFi client (STA).
# Installs system dependencies, creates Python venv, installs packages,
# and installs the systemd unit.
#
# Usage: sudo bash edge/setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EDGE_DIR="${REPO_ROOT}/edge"
VENV_DIR="${REPO_ROOT}/.venv"

echo "=== CSI Edge Pipeline Setup ==="
echo "Repo root: ${REPO_ROOT}"

# --- System dependencies ---
echo "--- Installing system packages ---"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3-venv \
    python3-dev \
    libopenblas-dev

# --- Python virtual environment ---
echo "--- Setting up Python venv ---"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv "${VENV_DIR}"
    echo "Created venv at ${VENV_DIR}"
else
    echo "Venv already exists at ${VENV_DIR}"
fi

"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${EDGE_DIR}/requirements.txt"

# --- Symlink proto/ into edge for imports ---
echo "--- Checking proto/ accessibility ---"
if [ ! -d "${REPO_ROOT}/proto" ]; then
    echo "WARNING: proto/ directory not found at ${REPO_ROOT}/proto"
else
    echo "proto/ found — accessible via PYTHONPATH=${REPO_ROOT}"
fi

# --- systemd unit ---
echo "--- Installing systemd service ---"
cp "${EDGE_DIR}/systemd/csi-aggregator.service" /etc/systemd/system/csi-aggregator.service
systemctl daemon-reload
systemctl enable csi-aggregator.service
echo "Enabled csi-aggregator.service (start with: systemctl start csi-aggregator)"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Connect the Pi to your iPhone hotspot WiFi network"
echo "  2. Note the Pi's IP address (ip addr show wlan0) — ESP32 nodes need it"
echo "  3. Start the pipeline: sudo systemctl start csi-aggregator"
