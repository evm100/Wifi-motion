# WiFi CSI Motion Sensing

Multi-node WiFi CSI (Channel State Information) sensing system for device-free human motion and presence detection. The system captures 802.11 CSI using ESP32-S3 microcontrollers, preprocesses signals on a Raspberry Pi edge node, and classifies activities with deep learning on an RTX 4080 GPU server.

## Architecture

```
[TX ESP32-S3] ──ESP-NOW broadcast @ 100Hz──> air
                                              |
                    +--------------------------+
                    |           |              |
              [RX Node 1] [RX Node 2] [RX Node 3]
              (promiscuous CSI capture, UDP to Pi)
                    |           |              |
                    +-----+----+--------------+
                          | UDP :5005 (binary packets)
                          v
                    [Raspberry Pi 5]
                    - WiFi AP (hostapd, 192.168.4.1, ch 6)
                    - UDP ingestion + packet alignment
                    - DSP: phase sanitization, Hampel, Butterworth, PCA
                    - Feature extraction: Doppler spectrograms
                    - ZMQ PUB to GPU server
                          | ZMQ tcp://*:5556
                          v
                    [RTX 4080 Server]
                    - Docker + NVIDIA runtime
                    - PyTorch: ResNet18, CNN-GRU, Transformer
                    - Real-time classification + localization
```

## Quick Start

### Firmware (ESP32-S3)

Requires ESP-IDF 5.5+. The developer alias `esp` activates the IDF environment.

```bash
# Build and flash TX node
cd firmware/tx-node
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor

# Build and flash RX nodes
cd firmware/rx-node
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSB1 flash monitor

# Or flash all at once
./tools/flash_all.sh /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyUSB2 /dev/ttyUSB3

# Provision node-specific config via NVS
python tools/provision_node.py --port /dev/ttyUSB1 --node-id 1 --ssid CSI_AP --password csi12345
```

### Raspberry Pi (Edge)

```bash
# Set up Python environment
cd edge
./setup.sh              # creates venv, installs deps, configures hostapd
source .venv/bin/activate

# Start the aggregation pipeline
python -m aggregator     # UDP receiver + DSP + ZMQ forwarding

# Or use systemd
sudo systemctl start csi-aggregator
```

### GPU Server

```bash
# Training
cd gpu
docker compose --profile training up

# Real-time inference
docker compose up -d
docker compose logs -f
```

### Developer Tools

```bash
# Visualize live CSI amplitude heatmap
python tools/visualize_csi.py --port 5005

# Collect labeled training data
python tools/collect_data.py --activities walking,sitting,standing --duration-per-activity 30

# Replay a capture for offline testing
python tools/replay_capture.py data/sessions/capture.bin --speed 2.0
```

## Project Structure

```
wifi-csi-sensing/
├── proto/           # Shared binary protocol (C struct + Python dataclass)
├── firmware/
│   ├── tx-node/     # ESP-IDF: ESP-NOW broadcast injector
│   └── rx-node/     # ESP-IDF: promiscuous CSI capture + UDP stream
├── edge/            # Raspberry Pi: DSP pipeline + ZMQ forwarding
│   ├── aggregator/  # UDP receiver, parser, packet alignment
│   ├── dsp/         # Phase sanitization, filters, PCA, features
│   └── forwarding/  # ZMQ publisher to GPU
├── gpu/             # RTX 4080: PyTorch models + inference server
│   ├── models/      # ResNet18, CNN-GRU, Transformer, fusion, DA
│   ├── training/    # Train loops, dataset, contrastive, few-shot
│   └── inference/   # ZMQ subscriber + real-time classifier
├── tools/           # Developer utilities
└── .github/         # CI workflows
```

See `docs/` for detailed documentation.
