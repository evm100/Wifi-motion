# CLAUDE.md — WiFi CSI Sensing Pipeline Project Brief

## Project Overview

Multi-node WiFi CSI sensing system for device-free human motion and presence detection. The system uses 802.11 Channel State Information captured by ESP32-S3 microcontrollers, preprocessed on a Raspberry Pi, and classified by deep learning on an RTX 4080 GPU server.

## Reference Documents

Detailed implementation code, algorithms, equations, and research citations are in these
files at the repo root. **Read the relevant module before implementing each phase.**
They contain working, tested code to use directly — not pseudocode.

- `module1_rf_physics_csi.md` — RF propagation math, CSI channel model, Fresnel zones, subcarrier layout, Doppler shift calculations, ESP32-S3 CSI callback code, Python receiver code
- `module2_edge_hardware_topology.md` — ESP32-S3 N16R8 hardware constraints, CFO/SFO/PDD explanation, network topology, SNTP time sync, binary packet format, PSRAM ring buffer, full TX and RX firmware code, Pi UDP receiver, Kconfig definitions, sdkconfig.defaults
- `module3_dsp_feature_extraction.md` — Phase sanitization (linear regression, conjugate multiplication, TSFR), Hampel filter, Butterworth lowpass, adaptive baseline removal, streaming PCA, Doppler spectrogram, feature extraction, ZMQ forwarding, complete integrated pipeline class
- `module4_deep_learning_domain_adaptation.md` — CSI spectrogram construction, BVP estimation, ResNet18 for CSI, CNN-GRU hybrid, CSI Transformer, multi-node attention fusion, localization regression, adversarial domain adaptation with gradient reversal, Prototypical Networks for few-shot, contrastive self-supervised pretraining, training loop, real-time inference server

## Hardware

- **4× ESP32-S3-WROOM-1-N16R8** (16 MB flash, 8 MB octal PSRAM, dual-core LX7 @ 240 MHz)
- Each board has a **2 dBi external stick dipole antenna** (IPEX/U.FL connector)
- **1 board is the TX node** (ESP-NOW broadcast frame injector)
- **3 boards are RX nodes** (promiscuous-mode CSI capture + UDP streaming)
- **1× Raspberry Pi 5** as edge aggregator (WiFi AP for ESP32 mesh + Ethernet to GPU)
- **1× RTX 4080 server** for training and real-time inference

## Architecture

```
[TX ESP32-S3] ──ESP-NOW broadcast @ 100Hz──▶ air
                                              │
                    ┌─────────────────────────┤
                    │           │             │
              [RX Node 1] [RX Node 2] [RX Node 3]
              (promiscuous CSI capture, UDP to Pi)
                    │           │             │
                    └─────────┬───────────────┘
                              │ UDP :5005 (binary packets)
                              ▼
                    [Raspberry Pi]
                    - WiFi AP (hostapd, 192.168.4.1, channel 6)
                    - UDP ingestion + packet alignment by TX seq num
                    - DSP: phase sanitization, Hampel, Butterworth, PCA
                    - Feature extraction: Doppler spectrograms
                    - ZMQ PUB to GPU server
                              │ ZMQ tcp://*:5556 (over Ethernet)
                              ▼
                    [RTX 4080 Server]
                    - Docker container with NVIDIA runtime
                    - ZMQ SUB receives tensors
                    - PyTorch models: ResNet18, CNN-GRU, Transformer
                    - Real-time classification + localization
```

## Repository Structure

Monorepo with three tiers. No Docker on ESP32 or Pi. Docker only on GPU server.

```
wifi-csi-sensing/
├── CLAUDE.md                         # This file
├── README.md
├── LICENSE
├── Makefile
├── .gitignore
│
├── proto/                            # Shared binary protocol (THE contract)
│   ├── csi_packet.h                  # C struct for firmware
│   ├── csi_packet.py                 # Python dataclass mirror
│   ├── constants.py                  # Shared constants
│   └── test_protocol_compat.py       # CI test: C/Python agreement
│
├── firmware/
│   ├── tx-node/                      # ESP-IDF project for TX
│   │   ├── CMakeLists.txt
│   │   ├── sdkconfig.defaults
│   │   ├── sdkconfig.defaults.esp32s3
│   │   ├── partitions.csv
│   │   ├── main/
│   │   │   ├── CMakeLists.txt
│   │   │   └── main.c
│   │   └── components/
│   │       ├── wifi_manager/
│   │       ├── csi_tx/
│   │       ├── time_sync/
│   │       └── health_check/
│   ├── rx-node/                      # ESP-IDF project for RX
│   │   ├── CMakeLists.txt
│   │   ├── sdkconfig.defaults
│   │   ├── sdkconfig.defaults.esp32s3
│   │   ├── partitions.csv
│   │   ├── main/
│   │   │   ├── CMakeLists.txt
│   │   │   └── main.c
│   │   └── components/
│   │       ├── wifi_manager/
│   │       ├── csi_stream/
│   │       ├── csi_config/
│   │       ├── time_sync/
│   │       └── health_check/
│   └── shared/
│       └── include/
│           └── csi_protocol.h        # Symlink → ../../proto/csi_packet.h
│
├── edge/                             # Raspberry Pi (native Python + systemd)
│   ├── setup.sh
│   ├── requirements.txt
│   ├── aggregator/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── udp_receiver.py
│   │   ├── packet_parser.py
│   │   └── aligner.py
│   ├── dsp/
│   │   ├── __init__.py
│   │   ├── phase_sanitizer.py
│   │   ├── amplitude_filter.py
│   │   ├── baseline.py
│   │   ├── pca.py
│   │   └── feature_extractor.py
│   ├── forwarding/
│   │   ├── __init__.py
│   │   └── gpu_forwarder.py
│   ├── config/
│   │   ├── pipeline.yaml
│   │   ├── network.yaml
│   │   ├── hostapd.conf.example
│   │   └── dnsmasq.conf
│   ├── systemd/
│   │   └── csi-aggregator.service
│   └── tests/
│       ├── test_parser.py
│       ├── test_filters.py
│       ├── test_aligner.py
│       └── fixtures/
│
├── gpu/                              # RTX 4080 (Docker + NVIDIA)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet_csi.py
│   │   ├── cnn_gru.py
│   │   ├── transformer.py
│   │   ├── multi_node_fusion.py
│   │   ├── localizer.py
│   │   └── domain_adaptation.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── dataset.py
│   │   ├── contrastive_pretrain.py
│   │   └── few_shot.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── server.py
│   │   └── classifier.py
│   ├── configs/
│   │   ├── train_resnet.yaml
│   │   ├── train_transformer.yaml
│   │   └── domain_adapt.yaml
│   ├── data/
│   │   └── README.md
│   ├── checkpoints/
│   │   └── .gitkeep
│   └── tests/
│       ├── test_models.py
│       └── test_inference.py
│
├── tools/
│   ├── collect_data.py
│   ├── visualize_csi.py
│   ├── replay_capture.py
│   ├── flash_all.sh
│   └── provision_node.py
│
├── docs/
│
└── .github/
    └── workflows/
        ├── edge-test.yml
        ├── gpu-test.yml
        └── protocol-compat.yml
```

## Shared Protocol (proto/)

The binary packet format is the critical contract between firmware and Python.

### Packet structure (20-byte header + variable payload)

```
Offset  Size  Field            Type       Notes
0       4     magic            uint32_le  Always 0xC5110001
4       1     version          uint8      Protocol version (currently 1)
5       1     node_id          uint8      RX node identifier (1, 2, or 3)
6       2     n_subcarriers    uint16_le  Number of I/Q pairs in payload
8       1     rssi             int8       Received signal strength (dBm)
9       1     noise_floor      int8       RF noise floor (dBm)
10      1     channel          uint8      WiFi channel number
11      1     flags            uint8      Bit flags (see below)
12      4     seq_num          uint32_le  TX sequence number
16      4     timestamp_us     uint32_le  Local microsecond timestamp (low 32 bits)
20      N*2   payload          int8[]     I/Q pairs: [imag0, real0, imag1, real1, ...]
```

Flag bits:
- Bit 0: `CSI_FLAG_HAS_HTLTF` — HT-LTF data present in payload
- Bit 1: `CSI_FLAG_FIRST_INVALID` — first 4 bytes of CSI payload are invalid

Total packet size for HT20 with LLTF+HT-LTF: 20 + 256 = 276 bytes.

## ESP32-S3 Firmware Conventions (ESP-IDF 5.5+)

### Build system
- Developer has an alias `esp` that activates the IDF environment
- Build: `cd firmware/tx-node && idf.py set-target esp32s3 && idf.py build`
- Flash: `idf.py -p /dev/ttyUSB0 flash monitor`
- Target: ESP32-S3 only

### Component architecture
- Lean `app_main()`: only init NVS, default event loop, then start components
- Each subsystem is its own IDF component under `components/`
- Components communicate via `esp_event`, not direct function calls
- Every component exposes a health status function
- Every public function returns `esp_err_t`
- Logging: `static const char *TAG = "component_name";` and `ESP_LOGx()` macros

### Task pinning
- Core 0: WiFi stack, CSI callback, networking (keep WiFi here)
- Core 1: CSI processing, UDP sending, health check

### Memory
- Use `heap_caps_malloc(size, MALLOC_CAP_SPIRAM)` for buffers >4 KB
- DMA-capable allocations in internal RAM: `MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL`
- CSI ring buffer: allocated in PSRAM, 32 KB, RINGBUF_TYPE_NOSPLIT

### PSRAM configuration (N16R8 specific)
- Octal SPI mode — GPIO35, 36, 37 are reserved for PSRAM bus
- sdkconfig: `CONFIG_SPIRAM=y`, `CONFIG_SPIRAM_MODE_OCT=y`, `CONFIG_SPIRAM_SPEED_80M=y`

### WiFi & CSI configuration
- WiFi channel: 6 (configurable via Kconfig)
- All nodes connect as STA to Pi's AP (SSID/password via Kconfig or NVS)
- TX node: ESP-NOW broadcast at configurable rate (default 100 Hz)
- RX nodes: promiscuous mode, CSI callback filters by TX MAC address
- CSI config: `lltf_en=true`, `htltf_en=true`, `stbc_htltf2_en=false`
- CSI callback must be IRAM_ATTR, non-blocking: copy to ring buffer only

### Kconfig parameters (rx-node)
```
CONFIG_CSI_NODE_ID       (uint8, 1-255, default 1)
CONFIG_CSI_TARGET_IP     (string, default "192.168.4.1")
CONFIG_CSI_TARGET_PORT   (uint16, default 5005)
CONFIG_CSI_TX_RATE_HZ    (uint16, default 100)     # TX node only
CONFIG_CSI_WIFI_SSID     (string)
CONFIG_CSI_WIFI_PASSWORD (string)
CONFIG_CSI_WIFI_CHANNEL  (uint8, default 6)
CONFIG_CSI_TX_MAC        (string, "AA:BB:CC:DD:EE:FF")  # RX nodes filter by this
```

### TX node behavior
1. Connect to Pi AP as STA
2. Init ESP-NOW, add broadcast peer (FF:FF:FF:FF:FF:FF)
3. SNTP sync to Pi
4. Loop: send ESP-NOW broadcast with incrementing seq_num in payload at fixed interval
5. Precise timing via `vTaskDelayUntil`, not `vTaskDelay`

### RX node behavior
1. Connect to Pi AP as STA
2. Enable promiscuous mode
3. Configure CSI (LLTF + HT-LTF)
4. Register CSI callback (IRAM_ATTR)
5. CSI callback: check TX MAC filter → copy header+CSI to PSRAM ring buffer
6. Sender task on Core 1: pull from ring buffer → UDP send to Pi
7. Stats task: log packet rate, RSSI, heap usage every 5 seconds

## Raspberry Pi Edge Code

### Technology stack
- Python 3.11+ in a virtualenv
- numpy, scipy, scikit-learn, pyzmq, PyYAML
- systemd for service management
- hostapd + dnsmasq for WiFi AP

### Pipeline flow (per aligned frame group at ~100 Hz)
1. **UDP receive** — async UDP socket on :5005, parse binary packets
2. **Align** — match packets from 3 RX nodes by TX seq_num
3. **Phase sanitize** — linear regression across subcarriers per packet
4. **Hampel filter** — outlier removal on amplitude (window=5, n_sigma=3)
5. **Butterworth lowpass** — 4th order, 10 Hz cutoff, SOS form
6. **Baseline removal** — adaptive EMA, fast alpha during calibration
7. **PCA** — IncrementalPCA, 20 components from 324 features (3×108)
8. **Feature extraction** — Doppler spectrogram via STFT every 50 frames
9. **ZMQ publish** — send tensor + metadata to GPU server

### Subcarrier extraction (HT20, LLTF + HT-LTF, 108 valid subcarriers)
```python
LLTF_VALID = list(range(6, 32)) + list(range(33, 59))      # 52
HTLTF_VALID = list(range(66, 94)) + list(range(100, 128))   # 56
ALL_VALID = LLTF_VALID + HTLTF_VALID                        # 108
```
Each subcarrier is 2 bytes in the payload: [imaginary, real] as int8.
Complex CSI: `real + j * imaginary`.

### Configuration files
- `pipeline.yaml`: DSP parameters (cutoff frequencies, PCA components, window sizes)
- `network.yaml`: IPs, ports, node count, GPU server address
- `hostapd.conf.example`: template (real one is gitignored — contains password)

## GPU Server Code

### Technology stack
- Docker with `nvidia/cuda:12.4.1-devel-ubuntu22.04` base
- PyTorch 2.x, torchvision
- pyzmq for tensor reception
- Hydra or argparse for training configs

### Models to implement
1. **CSIResNet** — ResNet18 modified for multi-channel CSI spectrograms [C, H, W] where C = n_nodes × n_pcs. Replace first conv layer to accept C input channels. Pretrained ImageNet weights (repeat across new channels).
2. **CNNGRU** — CNN spatial feature extractor + bidirectional GRU temporal model. Input: [B, T, C, H, W] sequence of spectrogram frames.
3. **CSITransformer** — Transformer encoder on PCA time series [B, T, D]. Learnable positional encoding + CLS token.
4. **MultiNodeFusion** — Per-node encoders + cross-node multi-head attention + classifier.
5. **CSILocalizer** — Position regression [B, 2] or zone classification [B, n_zones].
6. **DomainAdaptiveCSINet** — Shared feature extractor + activity classifier + gradient-reversal domain discriminator.

### Inference server
- ZMQ SUB subscribes to Pi's PUB socket
- Receives pickled dicts with 'tensor' and 'metadata' keys
- Runs model.forward() on GPU, returns predictions
- Logs classification results and latency

### Docker compose profiles
- `training`: interactive, mounts data/ and checkpoints/, runs train.py
- `inference` (default): daemon, read-only checkpoints, auto-restart, network_mode host

## Key Technical Constants

```
WiFi:
  wavelength: 0.125 m (2.4 GHz)
  channel: 6 (center freq 2437 MHz)
  bandwidth: 20 MHz (HT20)
  subcarrier_spacing: 312.5 kHz
  n_valid_subcarriers: 108 (52 LLTF + 56 HT-LTF)

CSI:
  sampling_rate: 100 Hz (TX injection rate)
  packet_size: 276 bytes (20 header + 256 payload)
  per_node_bandwidth: ~27.6 KB/s
  total_bandwidth: ~82.8 KB/s (3 nodes)

DSP:
  hampel_window: 5
  hampel_n_sigma: 3.0
  butterworth_order: 4
  butterworth_cutoff: 10.0 Hz
  pca_n_components: 20
  spectrogram_nperseg: 128
  spectrogram_noverlap: 120
  feature_window: 256 samples (2.56 sec)
  feature_hop: 50 samples (0.5 sec)

ML:
  default_model: ResNet18
  default_n_classes: 7 (empty, walking, sitting, standing, falling, gesture, breathing)
  default_batch_size: 32
  default_lr: 1e-3
  default_epochs: 100
```

## Implementation Priority

### Phase 1 — Scaffold and end-to-end data flow
1. `proto/` — packet header C and Python definitions + test
2. `firmware/tx-node/` — minimal ESP-NOW broadcaster
3. `firmware/rx-node/` — CSI capture + UDP stream
4. `edge/aggregator/` — UDP receiver + parser + alignment
5. Verify: Pi receives and parses CSI from all 3 nodes

### Phase 2 — DSP pipeline
6. `edge/dsp/` — phase sanitization, Hampel, Butterworth, baseline
7. `edge/dsp/pca.py` — streaming PCA with calibration
8. `edge/dsp/feature_extractor.py` — Doppler spectrogram
9. `edge/forwarding/` — ZMQ to GPU

### Phase 3 — ML models
10. `gpu/models/resnet_csi.py` — first working classifier
11. `gpu/training/dataset.py` — spectrogram dataset
12. `gpu/training/train.py` — training loop
13. `gpu/inference/server.py` — ZMQ receiver + real-time prediction

### Phase 4 — Advanced
14. Multi-node fusion model
15. Localization model
16. Domain adaptation
17. Few-shot meta-learning
