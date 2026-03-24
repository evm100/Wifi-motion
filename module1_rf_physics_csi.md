# Module 1: The Physics of CSI and Human Interaction
## ESP32-S3 N16R8 WiFi Sensing — RF Physics, CSI Mathematics & Implementation Guide

---

## 1. How 2.4 GHz Radio Waves Interact with the Human Body

### 1.1 The Physical Mechanisms

At 2.4 GHz, the wavelength λ ≈ 12.5 cm. This is comparable to the scale of human body parts (torso ~40 cm, limbs ~8-12 cm width), which makes the human body a strong scatterer of WiFi signals. Three primary interaction mechanisms govern what your ESP32-S3 receivers will observe:

**Reflection** occurs when a signal hits a surface larger than the wavelength. The human torso, being several wavelengths wide, acts as a strong reflector. The reflected power depends on the dielectric properties of human tissue — at 2.4 GHz, the skin's relative permittivity (εr ≈ 38–42) creates a significant impedance mismatch with free space, reflecting roughly 50-60% of incident power at perpendicular incidence. This is your strongest sensing signal.

**Absorption** accounts for the energy that penetrates the body. Human tissue is ~60% water, and 2.4 GHz is close to water's microwave absorption band. The specific absorption rate means that signals passing *through* a person are attenuated by approximately 30–40 dB, depending on body composition and path length. This creates strong "shadow" effects behind a person relative to the TX.

**Scattering/Diffraction** occurs at edges and surfaces comparable to or smaller than the wavelength — arms, hands, fingers. This produces weaker but more directionally diverse signal components. For gesture recognition and fine-grained motion, scattered signals from limbs create the micro-Doppler signatures you'll classify on the GPU.

### 1.2 Practical Implication for Your 1TX-3RX Topology

With your 2dBi external stick antennas mounted on the ESP32-S3 boards, each TX-RX pair creates an ellipsoidal **Fresnel zone** — a 3D volume where signal propagation is most sensitive to disturbance. Any person crossing a Fresnel zone boundary causes constructive/destructive interference changes measurable in CSI.

The first Fresnel zone radius at the midpoint between TX and RX is:

```
r₁ = √(λ · d₁ · d₂ / (d₁ + d₂))
```

Where d₁ and d₂ are distances from the midpoint to TX and RX. For a 4m TX-RX separation at 2.4 GHz:

```
r₁ = √(0.125 × 2 × 2 / 4) ≈ 0.35 m
```

This ~35 cm radius means the first Fresnel zone is roughly body-width — ideal for whole-body presence detection. Your 3-RX topology creates three overlapping Fresnel zones, ensuring spatial coverage even when a person stands in a null zone of one link.

**Key hardware note:** External 2dBi dipole antennas on the ESP32-S3 are critical. PCB trace antennas have directional nulls and create massive near-field reflections off nearby surfaces (desk, wall mount). Mount your antennas vertically on tripods or wall brackets at ~1.2-1.5m height, keeping them at least 30 cm from any surface.

---

## 2. The CSI Channel Model — From Physics to Math

### 2.1 The OFDM Channel Response

In 802.11n, the channel is divided into OFDM subcarriers. Each subcarrier independently samples the channel's frequency response. The CSI for subcarrier k at time t is a complex value:

```
H(k, t) = |H(k, t)| · e^(j·∠H(k,t))
```

where `|H(k,t)|` is the **amplitude** and `∠H(k,t)` is the **phase**.

The underlying physics model decomposes this into multipath components:

```
H(k, t) = Σₙ αₙ(t) · e^(-j·2π·fₖ·τₙ(t))
```

Where:
- n indexes each propagation path (direct, wall reflection, human reflection, etc.)
- αₙ(t) is the complex attenuation of path n at time t
- fₖ is the frequency of subcarrier k
- τₙ(t) is the propagation delay (time-of-flight) of path n

This is the core equation connecting physical reality to your CSI data. When a person moves, they change τₙ(t) for the human-reflected paths, causing the complex sum to shift.

### 2.2 Static vs. Dynamic Component Decomposition

The standard decomposition for sensing separates CSI into:

```
H(k, t) = Hₛ(k) + Hd(k, t)
```

**Static component Hₛ(k)**: The sum of all paths from walls, floor, furniture, direct line-of-sight. This is your "baseline" — it doesn't change unless you move furniture.

**Dynamic component Hd(k, t)**: Paths involving human reflection/scattering. This is your "signal of interest."

In practice, your Raspberry Pi will estimate Hₛ by averaging CSI over a calibration window when the room is empty, then subtract it from live CSI to isolate Hd.

### 2.3 Your ESP32-S3 Subcarrier Layout

On the ESP32-S3 in HT20 (20 MHz) mode with both LLTF and HT-LTF enabled, you get up to **256 bytes** of CSI data per packet:

| Field | Subcarrier Indices | Valid Subcarriers | Bytes |
|-------|-------------------|-------------------|-------|
| LLTF  | 0–63              | 52 (excl. nulls/DC/guard) | 128 |
| HT-LTF | 64–127           | 56 (excl. nulls/DC) | 128 |
| **Total** |                | **108** usable | **256** |

Each subcarrier is stored as 2 bytes: `[imaginary, real]` (int8, int8). To reconstruct complex CSI:

```c
// In your ESP32-S3 CSI callback
complex_csi[k] = buf[2*k+1] + j * buf[2*k];  // real + j*imaginary
amplitude[k] = sqrt(buf[2*k+1]² + buf[2*k]²);
phase[k] = atan2(buf[2*k], buf[2*k+1]);  // atan2(imag, real)
```

**Valid subcarrier extraction for HT20 (no STBC):**

```python
# Python parsing on the Pi side
def extract_valid_subcarriers(raw_csi_bytes):
    """Extract 108 valid subcarriers from 256-byte ESP32-S3 CSI buffer."""
    complex_csi = []
    for i in range(0, len(raw_csi_bytes), 2):
        imag = int.from_bytes([raw_csi_bytes[i]], 'big', signed=True)
        real = int.from_bytes([raw_csi_bytes[i+1]], 'big', signed=True)
        complex_csi.append(complex(real, imag))

    valid_indices = []
    # LLTF: 52 valid subcarriers (skip nulls, guard, DC)
    valid_indices += list(range(6, 32))    # 26 subcarriers (negative freq)
    valid_indices += list(range(33, 59))   # 26 subcarriers (positive freq)
    # HT-LTF: 56 valid subcarriers
    valid_indices += list(range(66, 94))   # 28 subcarriers
    valid_indices += list(range(100, 128)) # 28 subcarriers

    return [complex_csi[i] for i in valid_indices]
```

The subcarrier frequency spacing Δf in 20 MHz mode is:

```
Δf = 20 MHz / 64 = 312.5 kHz
```

For channel 6 (center frequency 2437 MHz), subcarrier k maps to:

```
fₖ = 2437 MHz + k × 312.5 kHz    (k from -26 to +26 for LLTF)
```

This gives your system a frequency-domain resolution of 312.5 kHz across a 16.25 MHz effective bandwidth (52 × 312.5 kHz), which corresponds to a multipath time resolution of approximately 1/16.25 MHz ≈ 62 ns, or about 18.5 m in path-length difference. The HT-LTF extends this to 56 subcarriers with slightly different pilot placement.

---

## 3. Micro-Doppler Shifts in WiFi CSI

### 3.1 How Human Motion Creates Doppler

When a person moves with velocity v, the reflected signal experiences a Doppler frequency shift:

```
fD = (2v / λ) · cos(θ)
```

Where θ is the angle between the motion direction and the TX-RX bisector line. At 2.4 GHz (λ = 12.5 cm):

| Activity | Velocity | Max Doppler Shift |
|----------|----------|-------------------|
| Walking (normal) | ~1.2 m/s | ~19.2 Hz |
| Walking (fast) | ~1.8 m/s | ~28.8 Hz |
| Hand gesture | ~0.5 m/s | ~8.0 Hz |
| Breathing (chest displacement) | ~0.005 m/s | ~0.08 Hz |
| Falling | ~3.0 m/s | ~48.0 Hz |
| Sitting still | ~0 m/s | ~0 Hz (static presence) |

### 3.2 Observing Doppler in CSI Time Series

The Doppler shift manifests as a **time-varying phase** on subcarrier k:

```
∠H_dynamic(k, t) ≈ 2π · fD · t + φ₀
```

This means if you sample CSI at rate Fs and plot the phase of a single subcarrier over time, a walking person creates a sinusoidal oscillation at ~10-30 Hz. Breathing creates a much slower oscillation at 0.1-0.5 Hz.

**Critical constraint:** Your CSI sampling rate must be at least 2× the maximum expected Doppler shift (Nyquist). For walking detection, you need Fs ≥ ~60 Hz. For gesture recognition, Fs ≥ ~20 Hz. The ESP32-S3 with frame injection can achieve 100-200 packets/second, which is sufficient.

### 3.3 The Doppler Spectrogram — Your Primary ML Feature

The Doppler spectrogram is created by applying a Short-Time Fourier Transform (STFT) to the CSI time series of each subcarrier:

```python
import numpy as np
from scipy.signal import stft

def csi_to_doppler_spectrogram(csi_timeseries, fs=100, nperseg=128, noverlap=120):
    """
    Convert CSI time series to Doppler spectrogram.

    Args:
        csi_timeseries: Complex CSI values for one subcarrier [T samples]
        fs: CSI sampling rate in Hz
        nperseg: STFT window length (longer = better freq resolution)
        noverlap: Overlap between windows (higher = smoother time axis)

    Returns:
        f: Frequency axis (Doppler frequencies in Hz)
        t: Time axis
        Sxx: Power spectral density [freq × time]
    """
    f, t, Zxx = stft(csi_timeseries, fs=fs, nperseg=nperseg, noverlap=noverlap)
    Sxx = np.abs(Zxx) ** 2  # Power spectral density
    return f, t, 10 * np.log10(Sxx + 1e-10)  # dB scale
```

The resulting 2D spectrogram shows:
- **X-axis**: Time
- **Y-axis**: Doppler frequency (proportional to velocity)
- **Intensity**: Signal power at that velocity and time

A walking person creates a characteristic "whale tail" pattern. Breathing creates a low-frequency horizontal band. A fall creates a brief high-frequency burst followed by silence. These visual signatures are what your CNN on the RTX 4080 will classify.

---

## 4. The Multipath Environment Model

### 4.1 Modeling the Static Channel

In a typical indoor room, the static channel (no humans) consists of:

```
Hₛ(k) = α_LOS · e^(-j2πfₖτ_LOS)           # Direct path
       + Σ_walls αᵢ · e^(-j2πfₖτᵢ)          # Wall reflections
       + Σ_furniture αⱼ · e^(-j2πfₖτⱼ)       # Furniture reflections
       + Σ_higher_order ...                    # Multi-bounce paths
```

This baseline creates a characteristic frequency-selective fading pattern — some subcarriers will have high amplitude (constructive multipath interference), others low (destructive). This is the "fingerprint" of your room.

### 4.2 Defining the Noise Floor vs. Human Signal

**Environmental noise floor** has several components:

1. **Thermal noise**: kTB ≈ -101 dBm for 20 MHz bandwidth (constant, unavoidable)
2. **RF interference**: Other WiFi devices, Bluetooth, microwaves at 2.4 GHz
3. **Hardware noise**: ESP32-S3 ADC quantization, AGC artifacts
4. **CSI estimation noise**: The channel estimator in the ESP32's PHY layer has finite precision

**Practical SNR for human detection:**

The CSI amplitude variance when a person is present vs. absent is your core detection metric. In typical conditions:

- Empty room CSI amplitude variance: σ²_noise ≈ 0.5–2.0 (normalized units)
- Person walking: σ²_signal ≈ 10–50
- Person sitting still: σ²_signal ≈ 2–5
- Person breathing only: σ²_signal ≈ 0.5–2.0 (very close to noise floor)

This is why static presence detection is the hardest problem — the signal is buried in noise. Your multi-node approach helps: by correlating CSI changes across 3 RX nodes, you can reject uncorrelated noise while preserving correlated human signals.

### 4.3 Practical Baseline Calibration Code

```python
import numpy as np

class EnvironmentBaseline:
    """Adaptive baseline estimator for separating static and dynamic CSI."""

    def __init__(self, n_subcarriers=108, alpha=0.01):
        self.alpha = alpha  # Exponential moving average rate
        self.baseline = None
        self.variance_baseline = None

    def update_baseline(self, csi_complex):
        """Update static baseline with exponential moving average.

        Call this during calibration (empty room) with alpha=0.1 (fast),
        then switch to alpha=0.001 during operation (slow drift tracking).
        """
        if self.baseline is None:
            self.baseline = csi_complex.copy()
            self.variance_baseline = np.zeros(len(csi_complex))
        else:
            self.baseline = (1 - self.alpha) * self.baseline + self.alpha * csi_complex

    def extract_dynamic(self, csi_complex):
        """Remove static component to isolate human-induced changes."""
        if self.baseline is None:
            return csi_complex
        return csi_complex - self.baseline

    def compute_motion_metric(self, csi_complex, window=50):
        """Compute motion indicator from dynamic component variance.

        Returns a scalar indicating motion level across all subcarriers.
        """
        dynamic = self.extract_dynamic(csi_complex)
        amplitude_change = np.abs(dynamic)
        return np.mean(amplitude_change)
```

---

## 5. Putting It Together: End-to-End CSI Sensing on ESP32-S3

### 5.1 ESP32-S3 Firmware Architecture (ESP-IDF)

Your system uses a dedicated frame-injecting transmitter and three promiscuous-mode receivers. The transmitter sends ESP-NOW broadcast packets at a fixed rate; the receivers capture CSI from these packets.

**Transmitter (TX node) — Core task:**

```c
// components/csi_tx/csi_tx.c
// Frame injection at fixed rate using ESP-NOW broadcasts

#include "esp_now.h"
#include "esp_wifi.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "csi_tx";
#define TX_RATE_HZ     100   // 100 packets/sec = 10ms interval
#define PAYLOAD_SIZE   32    // Minimal payload, we only need the CSI trigger

static uint8_t broadcast_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
static uint32_t seq_num = 0;

static void tx_task(void *pvParams) {
    uint8_t payload[PAYLOAD_SIZE];
    TickType_t last_wake = xTaskGetTickCount();
    const TickType_t interval = pdMS_TO_TICKS(1000 / TX_RATE_HZ);

    while (1) {
        // Embed sequence number for sync
        memcpy(payload, &seq_num, sizeof(seq_num));
        seq_num++;

        esp_err_t ret = esp_now_send(broadcast_mac, payload, PAYLOAD_SIZE);
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "Send failed: %s", esp_err_to_name(ret));
        }

        // Precise timing via vTaskDelayUntil, not vTaskDelay
        vTaskDelayUntil(&last_wake, interval);
    }
}

esp_err_t csi_tx_init(void) {
    ESP_LOGI(TAG, "Initializing CSI TX at %d Hz", TX_RATE_HZ);

    // Initialize ESP-NOW
    ESP_ERROR_CHECK(esp_now_init());

    // Add broadcast peer
    esp_now_peer_info_t peer = {
        .channel = 6,        // Fixed channel — must match all RX nodes
        .ifidx = WIFI_IF_STA,
        .encrypt = false,
    };
    memcpy(peer.peer_addr, broadcast_mac, 6);
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));

    // Pin to core 0 (WiFi core on S3)
    xTaskCreatePinnedToCore(tx_task, "csi_tx", 4096, NULL, 5, NULL, 0);
    return ESP_OK;
}
```

**Receiver (RX node) — CSI extraction and UDP streaming:**

```c
// components/csi_rx/csi_rx.c
// Promiscuous mode CSI capture with UDP forwarding to Pi

#include "esp_wifi.h"
#include "esp_log.h"
#include "lwip/sockets.h"
#include <string.h>

static const char *TAG = "csi_rx";

#define PI_IP           "192.168.4.1"  // Raspberry Pi IP
#define PI_PORT         5005
#define NODE_ID         1              // Unique per RX node (1, 2, or 3)
#define CSI_BUF_SIZE    256            // HT20 LLTF+HT-LTF

static int udp_sock = -1;
static struct sockaddr_in pi_addr;

// Binary packet header for efficient parsing on Pi
typedef struct __attribute__((packed)) {
    uint32_t magic;          // 0xCSI10001
    uint8_t  node_id;
    uint8_t  n_subcarriers;
    int8_t   rssi;
    int8_t   noise_floor;
    uint32_t timestamp_us;   // Local microsecond timestamp
    uint32_t seq_num;        // From TX payload
    uint16_t data_len;       // CSI data length in bytes
} csi_header_t;

static void csi_callback(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len == 0) return;

    // Filter: only process packets from our TX node's MAC
    // (Add MAC filter check here for your TX device)

    // Build binary packet: header + raw CSI I/Q data
    csi_header_t hdr = {
        .magic = 0xC5110001,
        .node_id = NODE_ID,
        .n_subcarriers = info->len / 2,
        .rssi = info->rx_ctrl.rssi,
        .noise_floor = info->rx_ctrl.noise_floor,
        .timestamp_us = (uint32_t)(esp_timer_get_time() & 0xFFFFFFFF),
        .seq_num = 0,  // Extract from payload if needed
        .data_len = info->len,
    };

    // Assemble packet in PSRAM for large buffers
    uint8_t *pkt = heap_caps_malloc(sizeof(hdr) + info->len, MALLOC_CAP_SPIRAM);
    if (!pkt) return;

    memcpy(pkt, &hdr, sizeof(hdr));
    memcpy(pkt + sizeof(hdr), info->buf, info->len);

    // Non-blocking UDP send
    sendto(udp_sock, pkt, sizeof(hdr) + info->len, MSG_DONTWAIT,
           (struct sockaddr *)&pi_addr, sizeof(pi_addr));

    heap_caps_free(pkt);
}

esp_err_t csi_rx_init(void) {
    ESP_LOGI(TAG, "Initializing CSI RX node %d", NODE_ID);

    // Configure CSI extraction
    wifi_csi_config_t csi_config = {
        .lltf_en = true,           // Enable LLTF (52 subcarriers)
        .htltf_en = true,          // Enable HT-LTF (56 subcarriers)
        .stbc_htltf2_en = false,   // Disable STBC for simplicity
        .ltf_merge_en = false,     // Keep LLTF and HT-LTF separate
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    // Enable promiscuous mode to capture broadcast frames
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    // Setup UDP socket
    udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    memset(&pi_addr, 0, sizeof(pi_addr));
    pi_addr.sin_family = AF_INET;
    pi_addr.sin_port = htons(PI_PORT);
    inet_pton(AF_INET, PI_IP, &pi_addr.sin_addr);

    ESP_LOGI(TAG, "CSI RX streaming to %s:%d", PI_IP, PI_PORT);
    return ESP_OK;
}
```

### 5.2 Raspberry Pi — CSI Receiver and Quick Visualization

```python
#!/usr/bin/env python3
"""
csi_receiver.py — Runs on Raspberry Pi
Receives UDP CSI packets from 3 ESP32-S3 nodes,
parses binary format, extracts valid subcarriers,
computes amplitude and phase.
"""

import socket
import struct
import numpy as np
import time
from collections import defaultdict

# Binary header format matching csi_header_t
HEADER_FORMAT = '<IBBBB I I H'  # little-endian
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Valid subcarrier indices for HT20, LLTF + HT-LTF (108 total)
LLTF_VALID = list(range(6, 32)) + list(range(33, 59))    # 52
HTLTF_VALID = list(range(66, 94)) + list(range(100, 128)) # 56
ALL_VALID = LLTF_VALID + HTLTF_VALID                      # 108

def parse_csi_packet(data):
    """Parse binary CSI packet from ESP32-S3."""
    if len(data) < HEADER_SIZE:
        return None

    fields = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    magic, node_id, n_sc, rssi, noise, timestamp, seq, data_len = fields

    if magic != 0xC5110001:
        return None

    # Extract I/Q pairs as signed int8
    raw = data[HEADER_SIZE:HEADER_SIZE + data_len]
    iq_pairs = np.frombuffer(raw, dtype=np.int8).reshape(-1, 2)

    # Convert to complex: real + j*imag (note: ESP32 stores [imag, real])
    csi_complex = iq_pairs[:, 1].astype(np.float32) + \
                  1j * iq_pairs[:, 0].astype(np.float32)

    # Extract valid subcarriers
    valid_csi = csi_complex[ALL_VALID] if len(csi_complex) >= 128 else csi_complex[LLTF_VALID]

    return {
        'node_id': node_id,
        'rssi': rssi,
        'noise_floor': noise,
        'timestamp_us': timestamp,
        'seq_num': seq,
        'csi_complex': valid_csi,
        'amplitude': np.abs(valid_csi),
        'phase': np.angle(valid_csi),
    }

def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 5005))
    sock.settimeout(1.0)

    print(f"Listening for CSI on UDP :5005 (header={HEADER_SIZE}B)")

    # Per-node packet counters
    counts = defaultdict(int)
    t_start = time.time()

    while True:
        try:
            data, addr = sock.recvfrom(4096)
            pkt = parse_csi_packet(data)
            if pkt is None:
                continue

            nid = pkt['node_id']
            counts[nid] += 1

            # Print stats every 5 seconds
            elapsed = time.time() - t_start
            if elapsed >= 5.0:
                for node, count in sorted(counts.items()):
                    rate = count / elapsed
                    print(f"  Node {node}: {rate:.1f} pkt/s, "
                          f"RSSI={pkt['rssi']} dBm, "
                          f"subcarriers={len(pkt['csi_complex'])}")
                counts.clear()
                t_start = time.time()

        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break

    sock.close()

if __name__ == '__main__':
    main()
```

---

## 6. Key Research Papers and Resources

### Foundational Physics & Sensing Models

1. **Fresnel Zone Model for WiFi Sensing**
   Wu, D., Zeng, Y., Zhang, F. et al. "WiFi CSI-based device-free sensing: from Fresnel zone model to CSI-ratio model." *CCF Trans. Pervasive Comp. Interact.* 4, 88–102 (2022).
   — Establishes the mathematical link between Fresnel zones and CSI fluctuation patterns. Essential for understanding why TX-RX placement matters.

2. **FullBreathe: Complementarity of Amplitude and Phase**
   Zeng, Y. et al. "FullBreathe: Full Human Respiration Detection Exploiting Complementarity of CSI Phase and Amplitude." *Proc. ACM IMWUT* 2(3), Article 148 (2018).
   — Proves that amplitude and phase sensing have complementary blind spots (π/2 offset). Critical for your system: when amplitude is insensitive at a location, phase compensates.

3. **Widar 3.0: Body-coordinate Velocity Profile (BVP)**
   Zheng, Y. et al. "Zero-Effort Cross-Domain Gesture Recognition with Wi-Fi." *MobiSys 2019.*
   — Introduces BVP as a domain-independent feature, which is foundational for your Module 4 domain adaptation problem.

### ESP32-Specific CSI Tools

4. **Wi-ESP: CSI Tool for Device-Free Sensing**
   "Wi-ESP—A tool for CSI-based Device-Free Wi-Fi Sensing (DFWS)." *Journal of Computational Design and Engineering* 7(5), 644 (2020).
   — Analyzes ESP32 CSI capabilities for LLTF and HT-LTF fields, evaluates amplitude and phase quality.

5. **Espressif Official ESP-CSI Repository**
   https://github.com/espressif/esp-csi
   — Contains `console_test` example with human activity detection, plus `csi_data_read_parse.py` for subcarrier extraction. Your starting point for firmware.

6. **ESP-IDF v5.5 WiFi CSI Documentation (ESP32-S3)**
   https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/wifi.html
   — Official subcarrier index tables, CSI configuration API, and `wifi_csi_info_t` structure reference.

### Phase Sanitization

7. **Hands-on Wireless Sensing with Wi-Fi: A Tutorial**
   Zhang, D. et al. (2022). arXiv:2206.09532.
   — Comprehensive tutorial with MATLAB code for CFO calibration, SFO/PDD removal via conjugate multiplication, and nonlinear error correction. The sanitization code section at https://tns.thss.tsinghua.edu.cn/wst/docs/sanitization/ is directly implementable.

8. **TSFR: Time Smoothing and Frequency Rebuild**
   "Channel Phase Processing in Wireless Networks for HAR." arXiv:2303.16873 (2023).
   — Proposes an improved phase sanitization using linear regression + Savitzky-Golay filtering. Tested on 5 datasets with 3 DL architectures.

### Deep Learning for WiFi Sensing

9. **Deep Learning and Its Applications to WiFi Human Sensing**
   arXiv:2207.07859 (2022).
   — Benchmarks CNNs, LSTMs, and Transformers on CSI data across Intel 5300 and Atheros platforms. Provides the learning-based vs. model-based taxonomy.

10. **Person-in-WiFi**
    Wang, F. et al. "Person-in-WiFi: Fine-grained Person Perception using WiFi." *ICCV 2019.*
    — Demonstrates body segmentation and pose estimation from CSI using U-Net architecture with 3×3 antenna pairs.

### Curated Resource Lists

11. **Awesome WiFi CSI Sensing**
    https://github.com/NTUMARS/Awesome-WiFi-CSI-Sensing
    — Maintained list of papers, datasets (MM-Fi, Widar 3.0, NTU-Fi), and tools. Categorized by application (HAR, gesture, localization, vital signs).

### High-Dimensional ESP32-S3 Dataset

12. **Multi-Transceiver CSI Dataset for HAR**
    ScienceDirect, Data in Brief 55 (2024).
    — Uses four ESP32-S3-DevKitC-1 devices capturing 166 subcarriers (HT-LTF). 6 participants, 1200 samples/activity. Directly relevant to your hardware.

---

## 7. Quick-Start Checklist

- [ ] Flash TX node with ESP-NOW broadcast at 100 Hz on channel 6
- [ ] Flash 3 RX nodes in promiscuous mode, UDP streaming to Pi
- [ ] Mount all 4 ESP32-S3 boards with 2dBi stick antennas at 1.2-1.5m height
- [ ] Place TX and RX nodes to maximize Fresnel zone overlap coverage of the room
- [ ] Run `csi_receiver.py` on Pi — verify 100 pkt/s from each node
- [ ] Collect 30-second empty-room baseline for Hₛ estimation
- [ ] Walk through room and verify amplitude variance spike on all 3 links
- [ ] Generate first Doppler spectrogram with STFT to confirm motion signatures

---

*Module 1 Complete — Next: Module 2 (Edge Hardware & Network Topology) or Module 3 (Signal Pre-processing & Feature Extraction)*
