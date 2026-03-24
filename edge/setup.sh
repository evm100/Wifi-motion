#!/usr/bin/env bash
# setup.sh — Idempotent setup for Raspberry Pi CSI edge pipeline.
#
# Installs system dependencies, creates Python venv, installs packages,
# symlinks proto/, copies network configs, and installs systemd unit.
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
    hostapd \
    dnsmasq \
    libatlas-base-dev \
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

# --- Network configs ---
echo "--- Configuring network services ---"

# hostapd
if [ ! -f /etc/hostapd/hostapd.conf ]; then
    cp "${EDGE_DIR}/config/hostapd.conf.example" /etc/hostapd/hostapd.conf
    echo "Copied hostapd.conf.example → /etc/hostapd/hostapd.conf"
    echo ">>> EDIT /etc/hostapd/hostapd.conf to set SSID and password <<<"
else
    echo "hostapd.conf already exists — skipping (check manually)"
fi

# dnsmasq
if [ ! -f /etc/dnsmasq.d/csi-sensing.conf ]; then
    cp "${EDGE_DIR}/config/dnsmasq.conf" /etc/dnsmasq.d/csi-sensing.conf
    echo "Installed dnsmasq config to /etc/dnsmasq.d/csi-sensing.conf"
else
    echo "dnsmasq config already exists — skipping"
fi

# --- systemd unit ---
echo "--- Installing systemd service ---"
cp "${EDGE_DIR}/systemd/csi-aggregator.service" /etc/systemd/system/csi-aggregator.service
systemctl daemon-reload
systemctl enable csi-aggregator.service
echo "Enabled csi-aggregator.service (start with: systemctl start csi-aggregator)"

# Enable hostapd and dnsmasq
systemctl unmask hostapd 2>/dev/null || true
systemctl enable hostapd
systemctl enable dnsmasq

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Edit /etc/hostapd/hostapd.conf with your SSID and password"
echo "  2. Configure static IP on wlan0 (192.168.4.1/24)"
echo "  3. Reboot or start services:"
echo "     sudo systemctl start hostapd dnsmasq csi-aggregator"
