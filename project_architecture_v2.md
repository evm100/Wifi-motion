# WiFi CSI Sensing Pipeline — Repository Architecture

## Design Principles

- **ESP32-S3 firmware:** Native ESP-IDF (developer's `esp` alias activates the environment)
- **Raspberry Pi:** Native Python in a virtualenv, managed by systemd
- **RTX 4080 server:** Docker with NVIDIA runtime (only tier that benefits from containerization)
- **Shared protocol:** `proto/` directory is the single source of truth for the binary packet contract

## Directory Structure

```
wifi-csi-sensing/
│
├── README.md
├── LICENSE
├── .gitignore
├── Makefile
│
├── docs/
│   ├── module1_rf_physics.md
│   ├── module2_edge_hardware.md
│   ├── module3_dsp_pipeline.md
│   ├── module4_deep_learning.md
│   └── deployment.md
│
│
│  ── SHARED PROTOCOL ──────────────────────────────────────────
│
├── proto/
│   ├── csi_packet.h                  # C header (firmware includes this)
│   ├── csi_packet.py                 # Python dataclass mirror
│   ├── constants.py                  # Channel, subcarrier maps, valid indices
│   └── test_protocol_compat.py       # CI: verifies C and Python agree
│
│
│  ── TIER 1: ESP32-S3 FIRMWARE ────────────────────────────────
│  Build: `esp` then `idf.py build`
│  Flash: `idf.py -p /dev/ttyUSB0 flash monitor`
│
├── firmware/
│   ├── tx-node/
│   │   ├── CMakeLists.txt
│   │   ├── sdkconfig.defaults
│   │   ├── sdkconfig.defaults.esp32s3
│   │   ├── partitions.csv
│   │   ├── main/
│   │   │   ├── CMakeLists.txt
│   │   │   └── main.c
│   │   └── components/
│   │       ├── wifi_manager/
│   │       │   ├── CMakeLists.txt
│   │       │   ├── Kconfig
│   │       │   ├── include/wifi_manager.h
│   │       │   └── wifi_manager.c
│   │       ├── csi_tx/
│   │       │   ├── CMakeLists.txt
│   │       │   ├── Kconfig
│   │       │   ├── include/csi_tx.h
│   │       │   └── csi_tx.c
│   │       ├── time_sync/
│   │       │   ├── CMakeLists.txt
│   │       │   ├── include/time_sync.h
│   │       │   └── time_sync.c
│   │       └── health_check/
│   │           ├── CMakeLists.txt
│   │           ├── include/health_check.h
│   │           └── health_check.c
│   │
│   ├── rx-node/
│   │   ├── CMakeLists.txt
│   │   ├── sdkconfig.defaults
│   │   ├── sdkconfig.defaults.esp32s3
│   │   ├── partitions.csv
│   │   ├── main/
│   │   │   ├── CMakeLists.txt
│   │   │   └── main.c
│   │   └── components/
│   │       ├── wifi_manager/
│   │       │   ├── CMakeLists.txt
│   │       │   ├── Kconfig
│   │       │   ├── include/wifi_manager.h
│   │       │   └── wifi_manager.c
│   │       ├── csi_stream/
│   │       │   ├── CMakeLists.txt
│   │       │   ├── Kconfig
│   │       │   ├── include/csi_stream.h
│   │       │   └── csi_stream.c
│   │       ├── csi_config/
│   │       │   ├── CMakeLists.txt
│   │       │   ├── include/csi_config.h
│   │       │   └── csi_config.c
│   │       ├── time_sync/
│   │       │   ├── CMakeLists.txt
│   │       │   ├── include/time_sync.h
│   │       │   └── time_sync.c
│   │       └── health_check/
│   │           ├── CMakeLists.txt
│   │           ├── include/health_check.h
│   │           └── health_check.c
│   │
│   └── shared/
│       └── include/
│           └── csi_protocol.h        # Symlink → ../../proto/csi_packet.h
│
│
│  ── TIER 2: RASPBERRY PI (native Python + systemd) ──────────
│  Setup: `./edge/setup.sh`
│  Run: `sudo systemctl start csi-aggregator`
│
├── edge/
│   ├── setup.sh                      # Creates venv, installs deps, installs systemd unit
│   ├── requirements.txt              # numpy, scipy, scikit-learn, pyzmq
│   │
│   ├── aggregator/
│   │   ├── __init__.py
│   │   ├── __main__.py               # Entry point: `python -m aggregator`
│   │   ├── udp_receiver.py
│   │   ├── packet_parser.py
│   │   └── aligner.py
│   │
│   ├── dsp/
│   │   ├── __init__.py
│   │   ├── phase_sanitizer.py
│   │   ├── amplitude_filter.py
│   │   ├── baseline.py
│   │   ├── pca.py
│   │   └── feature_extractor.py
│   │
│   ├── forwarding/
│   │   ├── __init__.py
│   │   └── gpu_forwarder.py
│   │
│   ├── config/
│   │   ├── pipeline.yaml             # DSP tuning: cutoffs, PCA components, etc.
│   │   ├── network.yaml              # IPs, ports, node count
│   │   ├── hostapd.conf              # WiFi AP for ESP32 mesh
│   │   └── dnsmasq.conf              # DHCP for ESP32 nodes
│   │
│   ├── systemd/
│   │   ├── csi-aggregator.service    # Main pipeline service
│   │   └── csi-hostapd.service       # WiFi AP service (if not using system hostapd)
│   │
│   └── tests/
│       ├── test_parser.py
│       ├── test_filters.py
│       ├── test_aligner.py
│       └── fixtures/
│           └── sample_capture.bin
│
│
│  ── TIER 3: GPU SERVER (Docker + NVIDIA runtime) ────────────
│  Train: `make gpu-train`
│  Infer: `make gpu-infer`
│
├── gpu/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet_csi.py
│   │   ├── cnn_gru.py
│   │   ├── transformer.py
│   │   ├── multi_node_fusion.py
│   │   ├── localizer.py
│   │   └── domain_adaptation.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── dataset.py
│   │   ├── contrastive_pretrain.py
│   │   └── few_shot.py
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   └── classifier.py
│   │
│   ├── configs/
│   │   ├── train_resnet.yaml
│   │   ├── train_transformer.yaml
│   │   └── domain_adapt.yaml
│   │
│   ├── data/                         # gitignored (large)
│   │   ├── raw/
│   │   ├── processed/
│   │   └── README.md
│   │
│   ├── checkpoints/                  # gitignored
│   │   └── .gitkeep
│   │
│   └── tests/
│       ├── test_models.py
│       └── test_inference.py
│
│
│  ── TOOLS ────────────────────────────────────────────────────
│
├── tools/
│   ├── collect_data.py
│   ├── visualize_csi.py
│   ├── replay_capture.py
│   ├── flash_all.sh                  # Flash all 4 ESP32s in sequence
│   └── provision_node.py             # Write NVS config per node
│
│
│  ── CI ───────────────────────────────────────────────────────
│
└── .github/
    └── workflows/
        ├── edge-test.yml
        ├── gpu-test.yml
        └── protocol-compat.yml
```


## Raspberry Pi Setup (No Docker)

### setup.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_DIR/.venv-edge"
EDGE_DIR="$REPO_DIR/edge"

echo "=== CSI Aggregator Setup ==="

# ── System deps ──────────────────────────────────────────────
echo "[1/5] Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    python3 python3-venv python3-dev \
    hostapd dnsmasq \
    libatlas-base-dev libopenblas-dev    # numpy/scipy native acceleration

# ── Python venv ──────────────────────────────────────────────
echo "[2/5] Creating virtualenv at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel
pip install -r "$EDGE_DIR/requirements.txt"

# ── Symlink proto into Python path ───────────────────────────
echo "[3/5] Linking shared protocol..."
ln -sf "$REPO_DIR/proto" "$EDGE_DIR/aggregator/proto"
ln -sf "$REPO_DIR/proto" "$EDGE_DIR/dsp/proto"

# ── Network config (WiFi AP for ESP32 mesh) ──────────────────
echo "[4/5] Configuring WiFi AP..."
sudo cp "$EDGE_DIR/config/hostapd.conf" /etc/hostapd/hostapd.conf
sudo cp "$EDGE_DIR/config/dnsmasq.conf" /etc/dnsmasq.d/csi-mesh.conf

# Enable IP forwarding (so ESP32s can reach the GPU server via Pi)
echo "net.ipv4.ip_forward=1" | sudo tee /etc/sysctl.d/99-csi-forward.conf
sudo sysctl -p /etc/sysctl.d/99-csi-forward.conf

# Static IP on wlan0 (AP interface)
sudo tee /etc/network/interfaces.d/csi-wlan0 > /dev/null <<EOF
auto wlan0
iface wlan0 inet static
    address 192.168.4.1
    netmask 255.255.255.0
EOF

# ── Systemd services ─────────────────────────────────────────
echo "[5/5] Installing systemd services..."

sudo tee /etc/systemd/system/csi-aggregator.service > /dev/null <<EOF
[Unit]
Description=CSI Sensing Aggregator
After=network.target hostapd.service
Wants=hostapd.service

[Service]
Type=simple
User=pi
WorkingDirectory=$REPO_DIR
ExecStart=$VENV_DIR/bin/python -m edge.aggregator --config edge/config/pipeline.yaml
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Performance tuning
Nice=-5
CPUAffinity=2 3
LimitNOFILE=65536

# Environment
Environment="PYTHONPATH=$REPO_DIR"
Environment="GPU_ADDRESS=tcp://192.168.1.100:5556"

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable csi-aggregator.service
sudo systemctl enable hostapd
sudo systemctl enable dnsmasq

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit edge/config/network.yaml with your GPU server IP"
echo "  2. Edit edge/config/hostapd.conf with your WiFi password"
echo "  3. Reboot, or run:"
echo "     sudo systemctl restart hostapd dnsmasq"
echo "     sudo systemctl start csi-aggregator"
echo "  4. Check status:"
echo "     sudo systemctl status csi-aggregator"
echo "     journalctl -u csi-aggregator -f"
```

### requirements.txt

```
numpy>=1.24,<2.0
scipy>=1.11
scikit-learn>=1.3
pyzmq>=25.0
PyYAML>=6.0
```

### systemd service details

The systemd unit gives you everything Docker would, without the overhead:

- `Restart=on-failure` — auto-restarts on crash (like `--restart unless-stopped`)
- `CPUAffinity=2 3` — pins to cores 2-3, leaving cores 0-1 for the OS and WiFi stack
- `Nice=-5` — slight scheduling priority boost for the real-time pipeline
- `journalctl -u csi-aggregator -f` — structured logging (like `docker logs -f`)
- `sudo systemctl stop/start/restart` — service lifecycle management

### Updating the Pi after a code change

```bash
# On your dev machine
git push

# On the Pi (SSH)
cd ~/wifi-csi-sensing
git pull
sudo systemctl restart csi-aggregator

# Or if deps changed
source .venv-edge/bin/activate
pip install -r edge/requirements.txt
sudo systemctl restart csi-aggregator
```


## Firmware Workflow (Native ESP-IDF)

```bash
# Activate your IDF environment
esp

# Build TX node
cd firmware/tx-node
idf.py set-target esp32s3
idf.py build

# Flash TX node (plug in via USB)
idf.py -p /dev/ttyUSB0 flash monitor

# Build RX node (different terminal or after TX is running)
cd ../rx-node
idf.py set-target esp32s3

# Per-node config: set node ID before building
# Option A: menuconfig
idf.py menuconfig    # → CSI Stream Configuration → Node ID = 1

# Option B: sdkconfig.defaults override per node
echo 'CONFIG_CSI_NODE_ID=1' >> sdkconfig.defaults
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor

# For nodes 2 and 3, change NODE_ID and reflash
```

### Linking the shared protocol header

In `firmware/rx-node/components/csi_stream/CMakeLists.txt`:

```cmake
idf_component_register(
    SRCS "csi_stream.c"
    INCLUDE_DIRS "include" "../../../shared/include"
    REQUIRES esp_wifi lwip esp_timer nvs_flash
)
```

The `shared/include/csi_protocol.h` is a symlink to `proto/csi_packet.h`,
so firmware and Python always reference the same definition.


## GPU Server (Docker — the one tier that earns it)

Docker on the GPU server solves real problems: CUDA version pinning,
PyTorch build compatibility, and isolation from whatever else runs
on the server.

### docker-compose.yml

```yaml
services:
  training:
    build:
      context: ..
      dockerfile: gpu/Dockerfile
    runtime: nvidia
    profiles: ["training"]
    volumes:
      - ./data:/app/data
      - ./checkpoints:/app/checkpoints
      - ./configs:/app/configs:ro
      - ../proto:/app/proto:ro
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
    command: python training/train.py --config configs/train_resnet.yaml

  inference:
    build:
      context: ..
      dockerfile: gpu/Dockerfile
    runtime: nvidia
    restart: unless-stopped
    network_mode: host            # Direct ZMQ access
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ../proto:/app/proto:ro
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - MODEL_PATH=/app/checkpoints/best_model.pt
      - ZMQ_ADDRESS=tcp://0.0.0.0:5556
    command: python inference/server.py
```

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY gpu/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY proto/ /app/proto/
COPY gpu/models/ /app/models/
COPY gpu/training/ /app/training/
COPY gpu/inference/ /app/inference/

ENV PYTHONPATH=/app
CMD ["python3", "inference/server.py"]
```


## Top-Level Makefile

```makefile
.PHONY: help build-tx build-rx flash-tx flash-rx \
        pi-setup pi-start pi-stop pi-logs pi-restart \
        gpu-train gpu-infer gpu-logs gpu-down \
        test collect visualize

help:
	@echo "── Firmware (run 'esp' first) ─────────────────────"
	@echo "  make build-tx          Build TX firmware"
	@echo "  make build-rx          Build RX firmware"
	@echo "  make flash-tx          Flash TX (set PORT=/dev/ttyUSBx)"
	@echo "  make flash-rx          Flash RX (set PORT=/dev/ttyUSBx)"
	@echo ""
	@echo "── Raspberry Pi (SSH into Pi) ─────────────────────"
	@echo "  make pi-setup          First-time Pi setup"
	@echo "  make pi-start          Start aggregator service"
	@echo "  make pi-stop           Stop aggregator service"
	@echo "  make pi-restart        Restart after code change"
	@echo "  make pi-logs           Tail aggregator logs"
	@echo ""
	@echo "── GPU Server ─────────────────────────────────────"
	@echo "  make gpu-train         Run training container"
	@echo "  make gpu-infer         Start inference daemon"
	@echo "  make gpu-logs          Tail inference logs"
	@echo "  make gpu-down          Stop all GPU containers"
	@echo ""
	@echo "── Dev tools ──────────────────────────────────────"
	@echo "  make test              Run all tests"
	@echo "  make collect           Start data collection session"
	@echo "  make visualize         Real-time CSI visualizer"

# ── Firmware ─────────────────────────────────────────────────
PORT ?= /dev/ttyUSB0

build-tx:
	cd firmware/tx-node && idf.py set-target esp32s3 && idf.py build

build-rx:
	cd firmware/rx-node && idf.py set-target esp32s3 && idf.py build

flash-tx:
	cd firmware/tx-node && idf.py -p $(PORT) flash monitor

flash-rx:
	cd firmware/rx-node && idf.py -p $(PORT) flash monitor

# ── Raspberry Pi ─────────────────────────────────────────────
pi-setup:
	bash edge/setup.sh

pi-start:
	sudo systemctl start csi-aggregator

pi-stop:
	sudo systemctl stop csi-aggregator

pi-restart:
	sudo systemctl restart csi-aggregator

pi-logs:
	journalctl -u csi-aggregator -f --no-pager

# ── GPU Server ───────────────────────────────────────────────
gpu-train:
	cd gpu && docker compose --profile training up --build

gpu-infer:
	cd gpu && docker compose up inference -d --build

gpu-logs:
	cd gpu && docker compose logs -f inference

gpu-down:
	cd gpu && docker compose down

# ── Testing ──────────────────────────────────────────────────
test:
	python -m pytest proto/ edge/tests/ gpu/tests/ -v

test-proto:
	python -m pytest proto/test_protocol_compat.py -v

test-edge:
	cd edge && python -m pytest tests/ -v

test-gpu:
	cd gpu && docker compose run --rm training python -m pytest tests/ -v

# ── Tools ────────────────────────────────────────────────────
collect:
	python tools/collect_data.py --config edge/config/pipeline.yaml

visualize:
	python tools/visualize_csi.py --port 5005
```


## .gitignore

```gitignore
# ESP-IDF build artifacts
firmware/*/build/
firmware/*/sdkconfig
firmware/*/sdkconfig.old

# Python
__pycache__/
*.pyc
*.pyo
.venv*/
*.egg-info/

# Data and models (large files)
gpu/data/raw/
gpu/data/processed/
gpu/checkpoints/*.pt
gpu/checkpoints/*.pth

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Secrets
*.env
edge/config/hostapd.conf   # contains WiFi password — use .example
```
