# Claude Code Prompts

## Phase 1 Prompt — Protocol + Firmware

```
Read module2_edge_hardware_topology.md fully before starting. It contains the
reference implementations for all firmware components.

Implement Phase 1 from CLAUDE.md: the shared protocol and both ESP-IDF firmware
projects.

Start with proto/:
- csi_packet.h — the C struct exactly as specified in CLAUDE.md (20-byte packed
  header with magic, version, node_id, n_subcarriers, rssi, noise_floor, channel,
  flags, seq_num, timestamp_us). Include the flag bit defines and the _Static_assert.
- csi_packet.py — Python dataclass mirror with from_bytes() and to_bytes() using
  struct.pack/unpack. Assert HEADER_SIZE == 20.
- constants.py — WiFi constants (channel, wavelength, subcarrier spacing) and the
  LLTF_VALID/HTLTF_VALID/ALL_VALID subcarrier index lists from module2.
- test_protocol_compat.py — pytest tests: header size check, field roundtrip for
  all fields including negative rssi/noise_floor, and magic value validation.

Then scaffold firmware/tx-node/ as a complete ESP-IDF 5.5 project for ESP32-S3:
- CMakeLists.txt, partitions.csv, sdkconfig.defaults, sdkconfig.defaults.esp32s3
  with the PSRAM/WiFi/CSI settings from CLAUDE.md.
- main/main.c — lean main: init NVS, default event loop, start wifi_manager,
  time_sync, csi_tx, health_check in order.
- components/wifi_manager/ — STA connection to configurable AP (SSID/password via
  Kconfig). Post WIFI_EVENT and IP_EVENT events. Reconnect on disconnect.
- components/csi_tx/ — ESP-NOW broadcast sender. Kconfig for TX_RATE_HZ. Use
  vTaskDelayUntil for precise timing. Embed incrementing seq_num in payload.
  Pin task to Core 0.
- components/time_sync/ — SNTP sync to Pi (192.168.4.1). get_timestamp_us()
  function using gettimeofday.
- components/health_check/ — periodic task logging free heap, PSRAM free, uptime.
  Feed task watchdog.

Then scaffold firmware/rx-node/ as a complete ESP-IDF 5.5 project:
- Same structure as tx-node but with csi_stream and csi_config instead of csi_tx.
- components/csi_stream/ — the core component. IRAM_ATTR CSI callback that filters
  by TX MAC, builds the binary packet header, copies header+CSI to a PSRAM ring
  buffer (xRingbufferCreateWithCaps, MALLOC_CAP_SPIRAM, 32KB). Separate UDP sender
  task on Core 1 pulls from ring buffer and sends via UDP. Stats task logs rates
  every 5 seconds. Use the csi_protocol.h from shared/include/.
- components/csi_config/ — NVS-based config loader with Kconfig fallback for
  node_id, target_ip, target_port, tx_mac, wifi_ssid, wifi_password.
- Kconfig menus for all parameters listed in CLAUDE.md.

Create firmware/shared/include/csi_protocol.h as a symlink to ../../proto/csi_packet.h.

Every component must follow the ESP-IDF conventions: public header in include/,
esp_err_t return types, TAG-based logging, error propagation not ESP_ERROR_CHECK
(except in main). Include all CMakeLists.txt files with correct REQUIRES dependencies.

Write real, complete, compilable code — not stubs or TODOs.
```

---

## Phase 2 Prompt — Pi Edge DSP Pipeline

```
Read module3_dsp_feature_extraction.md fully before starting. It contains working
Python implementations for every DSP algorithm.

Implement Phase 2 from CLAUDE.md: the complete Raspberry Pi edge pipeline.

Start with edge/aggregator/:
- __main__.py — entry point that parses --config flag, loads pipeline.yaml, starts
  the async UDP receiver, wires it to the aligner, DSP pipeline, and GPU forwarder.
  Use asyncio.
- udp_receiver.py — asyncio DatagramProtocol on configurable port (default 5005).
  Passes raw bytes to packet_parser.
- packet_parser.py — import and use proto/csi_packet.py. Validate magic, extract
  header, parse I/Q payload into complex numpy array. Extract valid subcarriers
  using proto/constants.py indices. Return a CSIPacket dataclass with node_id,
  seq_num, timestamp, rssi, csi_complex, amplitude, phase.
- aligner.py — buffer packets by seq_num. When all n_nodes (configurable, default 3)
  arrive for a given seq, emit the aligned group. Garbage-collect stale entries
  older than 50 sequence numbers. Track and log alignment rate and drop rate.

Then edge/dsp/:
- phase_sanitizer.py — three methods from module3:
  (1) sanitize_phase_linear(): unwrap, least-squares linear fit across subcarrier
      indices, subtract slope and intercept.
  (2) conjugate_multiply(): cross-node phase cleaning, returns cross_csi, diff_phase,
      product_amplitude.
  (3) sanitize_phase_tsfr(): linear sanitize then Savitzky-Golay time smoothing
      then frequency rebuild. Takes [n_packets, n_subcarriers] as input.
- amplitude_filter.py — two classes:
  (1) HampelFilter: per-subcarrier sliding window (default window=5, n_sigma=3).
      Uses median/MAD. Vectorized for [n_packets, n_subcarriers] input.
  (2) ButterFilter: Butterworth lowpass using scipy.signal.butter with SOS output.
      Configurable cutoff (default 10 Hz), order (default 4), fs (default 100).
      Both offline (sosfiltfilt) and realtime (sosfilt with state) methods.
- baseline.py — AdaptiveBaseline class: EMA with fast_alpha (0.1, calibration) and
  slow_alpha (0.001, operation). calibrate() method returns True when n_required
  samples reached. remove_static() subtracts baseline and slowly adapts when
  signal energy is low.
- pca.py — StreamingCSIPCA: concatenates all nodes' amplitudes (n_nodes × n_sc),
  standardizes, uses IncrementalPCA. Calibration phase collects samples, then
  fit(). transform() returns PC scores. get_top_subcarriers() maps PCA loadings
  back to (node, subcarrier) pairs.
- feature_extractor.py — CSIFeatureExtractor with methods: amplitude_variance,
  amplitude_range, temporal_correlation, signal_entropy, doppler_spectrogram
  (using scipy.signal.stft), power_spectral_density, spatial_variance,
  cross_node_correlation. All vectorized numpy.

Then edge/forwarding/:
- gpu_forwarder.py — GPUForwarder class using ZMQ PUB socket. send_tensor()
  pickles a dict with 'tensor' (numpy array) and 'metadata' (dict with frame
  count, motion energy, timestamp). Configurable address and send HWM.

Then edge/config/:
- pipeline.yaml — all DSP parameters with comments explaining each.
- network.yaml — UDP port, GPU address, node count, node IDs.
- hostapd.conf.example — channel 6, WPA2, placeholder SSID/password.
- dnsmasq.conf — interface wlan0, DHCP range 192.168.4.10-50.

Then edge/systemd/:
- csi-aggregator.service — systemd unit as specified in the architecture doc.
  After=network.target hostapd.service. CPUAffinity=2 3. Nice=-5.
  ExecStart using the venv python path. PYTHONPATH set to repo root.

Then edge/setup.sh — complete setup script: install system deps (python3-venv,
hostapd, dnsmasq, libatlas-base-dev, libopenblas-dev), create venv, pip install
requirements, symlink proto, copy network configs, install systemd unit, enable
services. Make it idempotent.

Then edge/requirements.txt with pinned ranges: numpy, scipy, scikit-learn, pyzmq,
PyYAML.

Then edge/tests/ — write real pytest tests:
- test_parser.py: construct a valid binary packet by hand, verify parser extracts
  correct fields. Test invalid magic rejection. Test I/Q to complex conversion.
- test_filters.py: test Hampel replaces known outlier. Test Butterworth attenuates
  above cutoff. Test baseline subtraction isolates dynamic component.
- test_aligner.py: feed packets with matching and mismatched seq_nums, verify
  groups emitted correctly and stale entries cleaned.

All code must be complete and runnable. Use type hints throughout. No stubs or TODOs.
```

---

## Phase 3 Prompt — GPU Models + Training + Inference

```
Read module4_deep_learning_domain_adaptation.md fully before starting. It contains
working PyTorch implementations for all models and training logic.

Implement Phase 3 from CLAUDE.md: GPU models, training pipeline, and inference server.

Start with gpu/models/:
- resnet_csi.py — CSIResNet class: ResNet18 backbone, replace first conv layer to
  accept n_input_channels (default 30 = 3 nodes × 10 PCs). Initialize new conv1
  by repeating pretrained 3-channel weights. Classification head with dropout.
  Forward returns [B, n_classes] logits.
- cnn_gru.py — CNNGRU class: per-frame CNN (3 conv+bn+relu+pool layers →
  AdaptiveAvgPool2d(4,4)), flatten to GRU input, 2-layer bidirectional GRU,
  last hidden state → classifier. Input: [B, T, C, H, W]. Output: [B, n_classes].
- transformer.py — CSITransformer: linear input projection, learnable positional
  embedding, CLS token prepended, TransformerEncoder (4 layers, 8 heads, d_model=128),
  CLS output → LayerNorm → linear classifier. Input: [B, T, D]. Output: [B, n_classes].
- multi_node_fusion.py — MultiNodeFusion: per-node linear encoders, stack outputs,
  MultiheadAttention across nodes, flatten attended features, classifier.
  Input: list of [B, T, 108] per node. Output: [B, n_classes].
- localizer.py — CSILocalizer: shared encoder (3 linear+relu+bn+dropout layers),
  position_head for [B, 2] regression, zone_head for [B, n_zones] classification.
  Forward takes mode='regression' or 'zone'.
- domain_adaptation.py — two classes:
  (1) GradientReversalLayer: autograd.Function that negates gradients in backward.
  (2) DomainAdaptiveCSINet: shared ResNet feature extractor, activity_classifier
      head, domain_discriminator head (with GRL). Forward returns both predictions.

Then gpu/training/:
- dataset.py — CSISpectrogramDataset(Dataset): loads numpy spectrograms and labels.
  __getitem__ returns (tensor, label). Include a from_directory() classmethod
  that loads from processed/ folder structure.
- train.py — main training script. AdamW optimizer, CosineAnnealingLR scheduler,
  CrossEntropyLoss with label_smoothing=0.1. Training loop with gradient clipping.
  Validation loop. Save best model by val accuracy. Log epoch metrics to stdout.
  Accept --config YAML path for all hyperparameters. Support selecting model type
  via config (resnet/cnn_gru/transformer).
- contrastive_pretrain.py — ContrastiveCSIPretrainer with SimCLR NT-Xent loss.
  Takes pairs (node_a_spec, node_b_spec) as positive pairs. Shared encoder +
  projection head. Training loop saves encoder weights for downstream fine-tuning.
- few_shot.py — PrototypicalCSINet: encoder that maps spectrograms to embedding
  space. classify_few_shot() computes class prototypes from support set, classifies
  queries by nearest prototype distance.

Then gpu/inference/:
- server.py — ZMQ SUB socket connecting to Pi. Main loop: recv → unpickle →
  classify → log result. Load model from MODEL_PATH env var. Warm up GPU with
  dummy forward pass on startup. Log prediction, confidence, and latency_ms.
- classifier.py — RealtimeCSIClassifier: wraps model loading, device placement,
  torch.no_grad() prediction. Returns (activity_name, confidence, latency_ms).

Then gpu/configs/:
- train_resnet.yaml — model: resnet, n_channels: 30, n_classes: 7, lr: 1e-3,
  epochs: 100, batch_size: 32, label_smoothing: 0.1, weight_decay: 1e-4.
- train_transformer.yaml — model: transformer, d_model: 128, nhead: 8,
  num_layers: 4, same training params.
- domain_adapt.yaml — model: domain_adaptive, n_domains: 3, lambda_domain: 0.5,
  source and target data paths.

Then gpu/Dockerfile:
- Base: nvidia/cuda:12.4.1-devel-ubuntu22.04
- Install python3, pip. Copy and install requirements.txt.
- Copy proto/, models/, training/, inference/. Set PYTHONPATH=/app.
- CMD: python3 inference/server.py

Then gpu/docker-compose.yml:
- training service: profile "training", runtime nvidia, mount data/ checkpoints/
  configs/ proto/. Command: python training/train.py.
- inference service: default, runtime nvidia, network_mode host, restart unless-stopped,
  mount checkpoints/ and proto/ read-only. Env: NVIDIA_VISIBLE_DEVICES, MODEL_PATH,
  ZMQ_ADDRESS.

Then gpu/requirements.txt: torch, torchvision, pyzmq, PyYAML, numpy, scipy, wandb.

Then gpu/tests/:
- test_models.py — for each model: instantiate, create random input tensor of
  correct shape, forward pass, verify output shape is [B, n_classes]. Test with
  batch size 1 and batch size 8.
- test_inference.py — instantiate RealtimeCSIClassifier with a freshly created
  model (random weights), run predict(), verify it returns a valid class name,
  confidence in [0,1], and positive latency.

All code must be complete, type-hinted, and runnable. No stubs or TODOs.
```

---

## Phase 4 Prompt — Tools, CI, and Polish

```
Implement Phase 4: developer tools, CI workflows, and repo polish.

tools/:
- visualize_csi.py — listens on UDP port 5005, parses CSI packets using proto/,
  plots real-time amplitude heatmap (subcarriers × time) using matplotlib with
  animation. One subplot per node. Command-line args: --port, --duration, --save.
- collect_data.py — orchestrates a labeled data collection session. Prompts the
  operator for activity name, countdown timer, records raw CSI packets from UDP
  to timestamped binary files. Saves metadata (activity, start/end timestamps,
  node count) as JSON sidecar. Args: --config, --output-dir, --activities,
  --duration-per-activity, --repetitions.
- replay_capture.py — reads a recorded binary capture file, replays packets to
  localhost UDP at original timing (using timestamps from packets). Useful for
  testing the edge pipeline offline without ESP32 hardware.
- flash_all.sh — takes a list of serial ports as args, flashes tx-node to the
  first port and rx-node to the remaining ports. Uses idf.py. Prints summary.
- provision_node.py — writes NVS config (node_id, target_ip, wifi_ssid, etc.)
  to a connected ESP32 via serial. Uses esptool or nvs_partition_gen.

.github/workflows/:
- protocol-compat.yml — on push to proto/: run test_protocol_compat.py.
- edge-test.yml — on push to edge/ or proto/: install deps in venv, run
  edge/tests/ with pytest.
- gpu-test.yml — on push to gpu/ or proto/: build Docker image, run gpu/tests/
  with pytest inside container (no GPU needed for shape tests).

Root files:
- README.md — project title, one-paragraph description, architecture diagram
  (ASCII), quick start for each tier (firmware, Pi, GPU), link to docs/.
- .gitignore — firmware build artifacts, sdkconfig, __pycache__, .venv*,
  gpu/data/raw/, gpu/data/processed/, gpu/checkpoints/*.pt, *.env, .DS_Store,
  edge/config/hostapd.conf (not .example).
- Makefile — all targets from the architecture doc (help, build-tx, build-rx,
  flash-tx, flash-rx, pi-setup, pi-start, pi-stop, pi-restart, pi-logs,
  gpu-train, gpu-infer, gpu-logs, gpu-down, test, test-proto, test-edge,
  test-gpu, collect, visualize).

All code complete and runnable. No stubs or TODOs.
```
