# Module 2: Edge Hardware & Network Topology
## ESP32-S3 N16R8 Constraints, Synchronization, and Data Aggregation for CSI Sensing

---

## 1. ESP32-S3 N16R8 Hardware Deep Dive

### 1.1 Your Specific Hardware: What N16R8 Gives You

The ESP32-S3-WROOM-1-N16R8 module packs a dual-core Xtensa LX7 at 240 MHz, 16 MB Quad SPI flash, and 8 MB Octal SPI PSRAM. For CSI sensing, the key specs are:

| Resource | Value | CSI Relevance |
|----------|-------|---------------|
| CPU | Dual-core LX7 @ 240 MHz | Core 0: WiFi stack. Core 1: CSI processing & UDP TX |
| Internal SRAM | 512 KB | CSI callback buffers, FreeRTOS stacks |
| PSRAM | 8 MB Octal @ 80 MHz | CSI ring buffers, packet assembly, large allocations |
| Flash | 16 MB Quad SPI | Firmware + OTA slots + NVS config |
| WiFi | 802.11 b/g/n, 2.4 GHz only | CSI from LLTF + HT-LTF, promiscuous mode |
| Antenna | External via U.FL/IPEX | Your 2dBi stick dipoles connect here |

**Critical PSRAM note for N16R8:** The 8 MB Octal PSRAM uses GPIO35, GPIO36, and GPIO37 for its SPI bus. These pins are **not available** for general use. Configure PSRAM as Octal in `sdkconfig`:

```
CONFIG_SPIRAM=y
CONFIG_SPIRAM_MODE_OCT=y
CONFIG_SPIRAM_SPEED_80M=y
```

### 1.2 WiFi NIC Limitations for CSI Extraction

The ESP32-S3's WiFi peripheral has specific constraints you need to design around:

**Subcarrier coverage (from Module 1):** In HT20 mode, you get 108 valid subcarriers (52 LLTF + 56 HT-LTF). In HT40 mode, this extends to ~166 subcarriers but requires a clear 40 MHz channel — difficult at 2.4 GHz where only 3 non-overlapping 20 MHz channels exist (1, 6, 11). **Recommendation: stay with HT20 on channel 6** unless your RF environment is exceptionally clean.

**`first_word_invalid` flag:** On the original ESP32, the first 4 bytes of CSI data were invalid due to a hardware bug (`first_word_invalid = true`). On the ESP32-S3, this issue is resolved in most firmware builds — check the flag in your CSI callback and skip the first 2 subcarriers if it's set.

**CSI callback context:** The CSI callback (`esp_wifi_set_csi_rx_cb`) runs inside the WiFi task on Core 0. You **must not** do any blocking or lengthy operations here. Copy the data to a PSRAM ring buffer and signal a processing task on Core 1.

**Promiscuous mode interaction:** When the RX nodes are in promiscuous mode, they capture CSI from **all** received frames, not just your TX node's frames. You must filter by the TX node's MAC address in the CSI callback to avoid processing ambient WiFi traffic CSI. The `wifi_csi_info_t.mac` field gives you the source MAC.

**Maximum CSI callback rate:** In practice with ESP-NOW broadcast at HT20/MCS0, the ESP32-S3 can reliably process CSI callbacks at 80–150 Hz. Beyond ~200 Hz, you'll start seeing callback queue overflows if your processing can't keep up. The PSRAM buffer strategy handles this.

### 1.3 Carrier Frequency Offset (CFO) and Sampling Frequency Offset (SFO)

This is the most important hardware imperfection to understand. Your TX and RX ESP32-S3 boards each have their own crystal oscillator (typically 40 MHz), and these crystals are never perfectly matched.

**CFO** arises because the two oscillators run at slightly different frequencies. This causes a random phase offset that shifts the entire phase-frequency curve up or down on every packet. At 2.4 GHz, even a 1 ppm crystal mismatch creates a ~2.4 kHz frequency offset, which over a 50 μs packet produces a phase rotation of roughly 8π — completely swamping the ~0.5π phase change from human breathing.

**SFO** occurs because the receiver's ADC sampling clock also derives from its local oscillator. This creates a linear slope error across the subcarrier phase spectrum — the phase error grows with subcarrier index.

**Packet Detection Delay (PDD)** adds an additional random time offset each time the receiver detects a new packet, manifesting as another linear phase slope.

**Why raw phase is unusable:** These three errors are **random per-packet** and **much larger** than the human-induced phase changes. This is why your pipeline needs phase sanitization on the Pi (Module 3). However, there are two approaches that work on the ESP32-S3 directly:

1. **Amplitude-only processing:** CSI amplitude `|H(k,t)|` is not affected by CFO/SFO. For basic motion detection and activity recognition, amplitude alone is sufficient. Only phase-sensitive applications (breathing, fine localization) require sanitization.

2. **Conjugate multiplication across antenna pairs:** If you had two antennas on one RX board, multiplying one antenna's CSI by the conjugate of another's cancels the shared CFO/SFO. The ESP32-S3 has only 1 antenna per board in your config, so this must be done across boards on the Pi.

**Espressif's co-crystal solution:** Espressif's own reference design for high-quality CSI connects an ESP32-C3 (TX) and ESP32-S3 (RX) through a shared clock buffer to eliminate CFO entirely. This is an option if you want to upgrade one TX-RX pair for phase-sensitive applications like vital sign monitoring.

---

## 2. Network Topology Design

### 2.1 The 1TX + 3RX Architecture

```
                    ┌──────────────────────────────────────┐
                    │          ROOM (4m × 5m)               │
                    │                                        │
   [RX Node 1]─────│─── Fresnel Zone 1 ───────[TX Node]    │
   (corner A)       │                           (corner D)   │
                    │         ╲  ╱                           │
                    │          ╳   Person                    │
                    │         ╱  ╲                           │
   [RX Node 2]─────│─── Fresnel Zone 2 ───────────┘        │
   (corner B)       │                                        │
                    │                                        │
   [RX Node 3]─────│─── Fresnel Zone 3 ───────────┘        │
   (center wall)    │                                        │
                    └──────────────────────────────────────┘
                              │ UDP (WiFi)
                              ▼
                    ┌──────────────────┐
                    │   Raspberry Pi    │  ◄── Edge Aggregator
                    │  (192.168.4.1)    │      DSP + Forwarding
                    └────────┬─────────┘
                             │ TCP/ZMQ (Ethernet)
                             ▼
                    ┌──────────────────┐
                    │  RTX 4080 Server  │  ◄── ML Classification
                    └──────────────────┘
```

**TX Node role:** Sends ESP-NOW broadcast frames at a fixed rate (100 Hz target). It does NOT need to receive CSI — its sole job is injecting known frames into the channel. Configure it as a WiFi SoftAP or station, with ESP-NOW layered on top.

**RX Nodes role:** Run in promiscuous mode, capturing CSI from the TX node's frames. Each RX node streams raw CSI data via UDP to the Raspberry Pi.

**Why not have RX nodes also transmit?** You could implement a full-mesh TDMA scheme where each node takes turns transmitting, giving you 4×3=12 spatial links instead of 3. However, this adds significant complexity in time-slot coordination and reduces your per-link sampling rate by 4×. For a first implementation, 1TX-3RX gives you 3 high-quality spatial links at full rate.

### 2.2 WiFi Channel Strategy

All 4 ESP32-S3 nodes **must be on the same WiFi channel.** ESP-NOW broadcast operates on whatever channel the underlying WiFi interface is configured to use.

**Recommended approach — Pi as SoftAP:**

The Raspberry Pi runs a WiFi access point (hostapd) on a **dedicated channel** (e.g., channel 6). All 4 ESP32-S3 boards connect as stations to this AP. This gives you:
- All nodes locked to the same channel automatically
- A WiFi data path from each ESP32 to the Pi for UDP CSI streaming
- NTP time sync from Pi to all nodes over this same network
- The TX node sends ESP-NOW broadcasts on this channel, which the RX nodes' promiscuous mode captures

**Alternative — dedicated CSI channel separate from data channel:**

If you need the data uplink on a different channel (e.g., to reduce self-interference), you can have the TX node broadcast on channel 1 while the RX nodes sniff channel 1 in promiscuous mode, but connect to the Pi's AP on channel 11 for UDP streaming. This is more complex and requires careful channel-switching on the RX nodes.

### 2.3 Antenna Placement Guidelines

Your 2dBi stick dipole antennas have an omnidirectional radiation pattern in the horizontal plane (donut-shaped). Key placement rules:

- **Height:** 1.2–1.5m off the ground (mid-torso height maximizes body interaction)
- **Orientation:** Vertical (matches the antenna's polarization plane)
- **Clearance:** Minimum 30 cm from walls, metal surfaces, or large furniture
- **TX position:** One corner or center of a wall
- **RX positions:** Remaining corners or walls, maximizing angular diversity
- **Avoid colinear placement:** Don't put TX and an RX on the same wall at the same height — the Fresnel zone will be too narrow to cover the room

---

## 3. Time Synchronization Across Distributed Nodes

### 3.1 Why Synchronization Matters

For single-link analysis (one TX-RX pair), you don't need inter-node time sync — each RX node's local timestamps are sufficient to track CSI changes over time. But for **multi-node sensor fusion** (correlating CSI across all 3 RX nodes to localize a person), you need all nodes' timestamps aligned to within a few milliseconds.

| Application | Required Sync Accuracy |
|-------------|----------------------|
| Per-link motion detection | Not needed (local timestamps fine) |
| Multi-link presence detection | ~10 ms |
| Multi-link localization | ~1–2 ms |
| Coherent phase fusion | ~10 μs (impractical without shared clock) |

### 3.2 Synchronization Strategy: TX Sequence Numbers + Pi-side NTP

The most robust approach for your topology combines three mechanisms:

**Layer 1 — TX sequence numbers (packet-level ordering):**

The TX node embeds an incrementing sequence number in every ESP-NOW broadcast payload. Each RX node extracts this from the received frame and includes it in the UDP packet sent to the Pi. The Pi can then align CSI samples from all 3 RX nodes by matching sequence numbers — if RX1 and RX3 both report seq=42857, those CSI measurements came from the same transmitted frame.

This is the most reliable alignment method and costs zero additional network traffic.

**Layer 2 — NTP sync to Pi (millisecond wall-clock alignment):**

All ESP32-S3 nodes sync their clocks to the Pi via SNTP. The ESP-IDF provides built-in SNTP support with microsecond-resolution timekeeping. On a local network with sub-1ms RTT, you can expect ~1–5ms clock accuracy after synchronization.

```c
// components/time_sync/time_sync.c
#include "esp_sntp.h"
#include "esp_log.h"

static const char *TAG = "time_sync";

static void time_sync_notification(struct timeval *tv) {
    ESP_LOGI(TAG, "Time synchronized: %lld.%06ld",
             (long long)tv->tv_sec, tv->tv_usec);
}

esp_err_t time_sync_init(void) {
    ESP_LOGI(TAG, "Initializing SNTP");

    esp_sntp_config_t config = ESP_NETIF_SNTP_DEFAULT_CONFIG("192.168.4.1");
    config.sync_cb = time_sync_notification;
    esp_netif_sntp_init(&config);

    // Wait for initial sync (up to 15 seconds)
    int retry = 0;
    while (esp_netif_sntp_sync_wait(pdMS_TO_TICKS(2000)) != ESP_OK && retry < 7) {
        ESP_LOGI(TAG, "Waiting for SNTP sync... (%d)", ++retry);
    }

    return ESP_OK;
}

// Get current time with microsecond resolution
int64_t get_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000LL + (int64_t)tv.tv_usec;
}
```

**Layer 3 — Local high-resolution timer (sample-level precision):**

The ESP32-S3's `esp_timer_get_time()` returns a 64-bit microsecond counter from boot. This has no drift within a single device and is ideal for measuring inter-packet intervals. Combine this with periodic NTP corrections to get both precision (local timer) and accuracy (NTP wall clock).

### 3.3 Pi-side Alignment Algorithm

```python
import numpy as np
from collections import defaultdict

class CSIAligner:
    """Align CSI packets from multiple RX nodes using TX sequence numbers."""

    def __init__(self, n_nodes=3, max_age_ms=50):
        self.n_nodes = n_nodes
        self.max_age_ms = max_age_ms
        self.buffers = defaultdict(dict)  # seq_num -> {node_id: csi_packet}

    def add_packet(self, packet):
        """Add a received CSI packet. Returns aligned group if complete."""
        seq = packet['seq_num']
        nid = packet['node_id']
        self.buffers[seq][nid] = packet

        # Check if we have all nodes for this sequence number
        if len(self.buffers[seq]) == self.n_nodes:
            aligned = self.buffers.pop(seq)
            self._cleanup_old(seq)
            return aligned

        return None

    def _cleanup_old(self, current_seq):
        """Remove stale entries (packets that will never complete)."""
        stale = [s for s in self.buffers if s < current_seq - 100]
        for s in stale:
            del self.buffers[s]
```

---

## 4. Data Serialization & Streaming Protocol

### 4.1 Why UDP (Not TCP, MQTT, or Protocol Buffers)

For streaming 100 Hz CSI data from 3 nodes, protocol choice is critical:

| Protocol | Overhead | Latency | Reliability | Verdict |
|----------|----------|---------|-------------|---------|
| **Raw UDP** | 8 bytes | ~1 ms | Best-effort | **Best for CSI** |
| TCP | 20+ bytes + ACK | ~5-50 ms | Guaranteed | Too slow (head-of-line blocking) |
| MQTT | TCP + MQTT header | ~10-100 ms | Guaranteed | Massive overhead |
| Protobuf/UDP | Protobuf + 8 bytes | ~1 ms | Best-effort | Unnecessary complexity |

**UDP wins** because:
- CSI data is ephemeral — a dropped packet from 10 ms ago is worthless; the next packet carries newer information
- No head-of-line blocking: TCP retransmits stall the entire stream
- Minimal overhead: your CSI packet is ~280 bytes, UDP header adds 8 bytes
- The ESP32-S3 can push UDP at well over 200 KB/s on a local network

**Handling packet loss:** At 100 Hz per node (300 packets/sec total), occasional UDP loss (1-3%) is acceptable. The TX sequence number lets the Pi detect gaps and interpolate or skip.

### 4.2 Binary Packet Format (ADR-018 Compatible)

Use a compact binary header instead of text/JSON to minimize serialization overhead on the ESP32:

```c
// Shared header definition (csi_protocol.h)
// Total header: 20 bytes + CSI payload (256 bytes typical) = 276 bytes per packet

typedef struct __attribute__((packed)) {
    uint32_t magic;          // 0xC5110001 — identifies CSI packets
    uint8_t  node_id;        // 1, 2, or 3 — identifies which RX node
    uint8_t  channel;        // WiFi channel number
    uint16_t n_subcarriers;  // Number of I/Q pairs in payload
    int8_t   rssi;           // RSSI of received frame (dBm)
    int8_t   noise_floor;    // RF noise floor (dBm)
    uint16_t reserved;       // Alignment padding
    uint32_t seq_num;        // TX sequence number (from ESP-NOW payload)
    uint32_t timestamp_us;   // Local microsecond timestamp (low 32 bits)
} csi_packet_header_t;

_Static_assert(sizeof(csi_packet_header_t) == 20, "Header must be 20 bytes");
```

### 4.3 Bandwidth Calculation

Per RX node at 100 Hz:
```
Packet size = 20 (header) + 256 (108 subcarriers × 2 bytes + padding) = 276 bytes
Data rate   = 276 × 100 = 27,600 bytes/sec = ~27.6 KB/s = ~221 Kbps
```

For 3 RX nodes: **~83 KB/s = ~663 Kbps total**

This is well within the ESP32-S3's WiFi throughput capacity and the Pi's ability to receive on a single UDP socket. Even on a congested 2.4 GHz network, this is a modest load.

### 4.4 ESP32-S3 Sender Implementation with PSRAM Ring Buffer

The CSI callback must be fast (runs in WiFi task context). Use a lock-free ring buffer in PSRAM to decouple capture from transmission:

```c
// components/csi_stream/csi_stream.c
#include "esp_wifi.h"
#include "lwip/sockets.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/ringbuf.h"

static const char *TAG = "csi_stream";

// Configuration
#define CSI_RING_BUF_SIZE   (32 * 1024)  // 32 KB in PSRAM (~115 packets)
#define TARGET_IP           CONFIG_CSI_TARGET_IP
#define TARGET_PORT         CONFIG_CSI_TARGET_PORT
#define NODE_ID             CONFIG_CSI_NODE_ID

static RingbufHandle_t csi_ringbuf = NULL;
static int udp_sock = -1;
static struct sockaddr_in target_addr;

// TX MAC address to filter (set during init)
static uint8_t tx_mac[6];
static bool tx_mac_set = false;

// Statistics
static volatile uint32_t cb_count = 0;
static volatile uint32_t drop_count = 0;
static volatile uint32_t send_count = 0;

// CSI callback — runs in WiFi task on Core 0
// MUST be fast: copy to ring buffer and return
static void IRAM_ATTR csi_rx_callback(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf || info->len == 0) return;

    // Filter by TX MAC address
    if (tx_mac_set) {
        if (memcmp(info->mac, tx_mac, 6) != 0) return;
    }

    cb_count++;

    // Build packet in stack (fast)
    csi_packet_header_t hdr = {
        .magic = 0xC5110001,
        .node_id = NODE_ID,
        .channel = info->rx_ctrl.channel,
        .n_subcarriers = info->len / 2,
        .rssi = info->rx_ctrl.rssi,
        .noise_floor = info->rx_ctrl.noise_floor,
        .reserved = 0,
        .seq_num = 0,  // TODO: extract from ESP-NOW payload
        .timestamp_us = (uint32_t)(esp_timer_get_time() & 0xFFFFFFFF),
    };

    // Try to send header + CSI data to ring buffer
    // Use xRingbufferSend with 0 timeout (non-blocking)
    size_t total = sizeof(hdr) + info->len;
    uint8_t *item = NULL;

    // Allocate contiguous item in ring buffer
    if (xRingbufferSendAcquire(csi_ringbuf, (void **)&item, total, 0) == pdTRUE) {
        memcpy(item, &hdr, sizeof(hdr));
        memcpy(item + sizeof(hdr), info->buf, info->len);
        xRingbufferSendComplete(csi_ringbuf, item);
    } else {
        drop_count++;  // Ring buffer full — sending can't keep up
    }
}

// UDP sender task — runs on Core 1
static void udp_sender_task(void *pvParams) {
    ESP_LOGI(TAG, "UDP sender started on core %d", xPortGetCoreID());

    while (1) {
        size_t item_size = 0;
        void *item = xRingbufferReceive(csi_ringbuf, &item_size, pdMS_TO_TICKS(100));

        if (item != NULL) {
            int ret = sendto(udp_sock, item, item_size, MSG_DONTWAIT,
                           (struct sockaddr *)&target_addr, sizeof(target_addr));
            if (ret > 0) send_count++;

            vRingbufferReturnItem(csi_ringbuf, item);
        }
    }
}

// Statistics reporting task
static void stats_task(void *pvParams) {
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        ESP_LOGI(TAG, "CSI stats: cb=%lu sent=%lu dropped=%lu heap=%lu psram=%lu",
                 cb_count, send_count, drop_count,
                 (unsigned long)esp_get_free_heap_size(),
                 (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
        cb_count = 0;
        send_count = 0;
        drop_count = 0;
    }
}

esp_err_t csi_stream_init(const uint8_t *filter_mac) {
    // Store TX MAC for filtering
    if (filter_mac) {
        memcpy(tx_mac, filter_mac, 6);
        tx_mac_set = true;
        ESP_LOGI(TAG, "Filtering CSI for TX MAC: %02X:%02X:%02X:%02X:%02X:%02X",
                 tx_mac[0], tx_mac[1], tx_mac[2],
                 tx_mac[3], tx_mac[4], tx_mac[5]);
    }

    // Create ring buffer in PSRAM
    csi_ringbuf = xRingbufferCreateWithCaps(
        CSI_RING_BUF_SIZE,
        RINGBUF_TYPE_NOSPLIT,
        MALLOC_CAP_SPIRAM
    );
    if (!csi_ringbuf) {
        ESP_LOGE(TAG, "Failed to create ring buffer in PSRAM");
        return ESP_FAIL;
    }

    // Setup UDP socket
    udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udp_sock < 0) {
        ESP_LOGE(TAG, "Socket creation failed");
        return ESP_FAIL;
    }

    // Increase socket send buffer
    int sndbuf = 16384;
    setsockopt(udp_sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    memset(&target_addr, 0, sizeof(target_addr));
    target_addr.sin_family = AF_INET;
    target_addr.sin_port = htons(TARGET_PORT);
    inet_pton(AF_INET, TARGET_IP, &target_addr.sin_addr);

    // Configure and enable CSI
    wifi_csi_config_t csi_cfg = {
        .lltf_en = true,
        .htltf_en = true,
        .stbc_htltf2_en = false,
        .ltf_merge_en = false,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = false,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_cfg));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_rx_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    // Start sender on Core 1 (keep WiFi on Core 0)
    xTaskCreatePinnedToCore(udp_sender_task, "csi_udp_tx", 4096, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(stats_task, "csi_stats", 2048, NULL, 1, NULL, 1);

    ESP_LOGI(TAG, "CSI streaming to %s:%d (node %d)", TARGET_IP, TARGET_PORT, NODE_ID);
    return ESP_OK;
}
```

### 4.5 Kconfig for Per-Node Configuration

Instead of hardcoding IPs and node IDs, use Kconfig so you flash the same firmware with different `sdkconfig` per device:

```kconfig
# components/csi_stream/Kconfig
menu "CSI Stream Configuration"

    config CSI_NODE_ID
        int "Node ID (unique per RX device)"
        range 1 255
        default 1
        help
            Unique identifier for this CSI receiver node.
            TX node = 0, RX nodes = 1, 2, 3.

    config CSI_TARGET_IP
        string "Aggregator IP address"
        default "192.168.4.1"
        help
            IP address of the Raspberry Pi aggregator.

    config CSI_TARGET_PORT
        int "Aggregator UDP port"
        range 1024 65535
        default 5005
        help
            UDP port the aggregator listens on.

    config CSI_TX_RATE_HZ
        int "TX frame injection rate (Hz)"
        range 10 500
        default 100
        help
            Rate at which the TX node sends ESP-NOW frames.
            100 Hz is good for motion; 200 Hz for gestures.

endmenu
```

### 4.6 NVS-Based Runtime Configuration (Alternative to Kconfig)

For field deployment where you don't want to recompile per device, store configuration in NVS and provision via a serial command or BLE during setup:

```c
// components/csi_config/csi_config.c
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_log.h"

static const char *TAG = "csi_config";

typedef struct {
    uint8_t  node_id;
    char     target_ip[16];
    uint16_t target_port;
    uint8_t  wifi_channel;
    uint16_t tx_rate_hz;
} csi_config_t;

static csi_config_t config;

esp_err_t csi_config_load(void) {
    nvs_handle_t nvs;
    esp_err_t err = nvs_open("csi_cfg", NVS_READONLY, &nvs);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "NVS namespace not found, using defaults");
        config.node_id = CONFIG_CSI_NODE_ID;
        strncpy(config.target_ip, CONFIG_CSI_TARGET_IP, sizeof(config.target_ip));
        config.target_port = CONFIG_CSI_TARGET_PORT;
        config.wifi_channel = 6;
        config.tx_rate_hz = CONFIG_CSI_TX_RATE_HZ;
        return ESP_OK;
    }

    uint8_t u8val;
    if (nvs_get_u8(nvs, "node_id", &u8val) == ESP_OK) {
        config.node_id = u8val;
        ESP_LOGI(TAG, "Loaded node_id=%d from NVS", u8val);
    }

    size_t len = sizeof(config.target_ip);
    if (nvs_get_str(nvs, "target_ip", config.target_ip, &len) == ESP_OK) {
        ESP_LOGI(TAG, "Loaded target_ip=%s from NVS", config.target_ip);
    }

    uint16_t u16val;
    if (nvs_get_u16(nvs, "target_port", &u16val) == ESP_OK) {
        config.target_port = u16val;
    }

    nvs_close(nvs);
    return ESP_OK;
}

const csi_config_t *csi_config_get(void) {
    return &config;
}
```

---

## 5. Raspberry Pi Aggregator Architecture

### 5.1 Pi Role in the Pipeline

The Raspberry Pi 4/5 serves as the **edge aggregator** between the ESP32-S3 mesh and the RTX 4080 server. Its responsibilities:

1. **UDP receiver:** Ingest CSI packets from 3 RX nodes at ~300 packets/sec
2. **Packet parsing & alignment:** Match packets by TX sequence number
3. **Real-time DSP:** Phase sanitization, filtering, PCA (Module 3)
4. **Feature extraction:** Compute amplitude variance, Doppler spectrograms
5. **Streaming to GPU server:** Forward sanitized tensors via TCP/ZMQ over Ethernet

### 5.2 Pi Network Setup

The Pi needs two network interfaces:

- **WiFi (wlan0):** Running hostapd as SoftAP for the ESP32-S3 mesh. Dedicated 2.4 GHz channel.
- **Ethernet (eth0):** Wired connection to the RTX 4080 server for high-bandwidth, low-latency tensor streaming.

```bash
# /etc/hostapd/hostapd.conf
interface=wlan0
driver=nl80211
ssid=CSI_SENSING_NET
hw_mode=g
channel=6
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=YourSecurePassword
wpa_key_mgmt=WPA-PSK
wpa_pairwise=TKIP
rsn_pairwise=CCMP

# /etc/dnsmasq.conf (DHCP for ESP32s)
interface=wlan0
dhcp-range=192.168.4.10,192.168.4.50,255.255.255.0,24h
```

### 5.3 High-Performance UDP Receiver (Python + asyncio)

For 300 packets/sec, a simple blocking `recvfrom` loop works. For robustness and headroom, use asyncio with a dedicated receive buffer:

```python
#!/usr/bin/env python3
"""
pi_aggregator.py — Raspberry Pi CSI aggregator
Receives UDP CSI from 3 ESP32-S3 nodes, aligns by sequence number,
and forwards to the DSP pipeline.
"""

import asyncio
import struct
import numpy as np
import time
from collections import defaultdict, deque

# Protocol constants
HEADER_FMT = '<I B B H b b H I I'  # little-endian, matches csi_packet_header_t
HEADER_SIZE = struct.calcsize(HEADER_FMT)
MAGIC = 0xC5110001

# Valid subcarrier indices (HT20, LLTF + HT-LTF)
LLTF_VALID = list(range(6, 32)) + list(range(33, 59))
HTLTF_VALID = list(range(66, 94)) + list(range(100, 128))
ALL_VALID = LLTF_VALID + HTLTF_VALID


class CSIPacket:
    __slots__ = ['node_id', 'channel', 'n_sc', 'rssi', 'noise',
                 'seq', 'timestamp', 'csi_complex', 'amplitude', 'phase']

    def __init__(self, data):
        fields = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
        magic, self.node_id, self.channel, self.n_sc, \
            self.rssi, self.noise, _, self.seq, self.timestamp = fields

        if magic != MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X}")

        # Parse I/Q pairs
        payload = data[HEADER_SIZE:]
        iq = np.frombuffer(payload, dtype=np.int8).reshape(-1, 2)
        raw_complex = iq[:, 1].astype(np.float32) + 1j * iq[:, 0].astype(np.float32)

        # Extract valid subcarriers
        if len(raw_complex) >= 128:
            self.csi_complex = raw_complex[ALL_VALID]
        else:
            self.csi_complex = raw_complex[LLTF_VALID]

        self.amplitude = np.abs(self.csi_complex)
        self.phase = np.angle(self.csi_complex)


class Aggregator:
    def __init__(self, n_nodes=3, window_size=500, max_pending=200):
        self.n_nodes = n_nodes
        self.window_size = window_size
        self.pending = {}       # seq -> {node_id: CSIPacket}
        self.max_pending = max_pending

        # Per-node time series buffers (circular)
        self.amplitude_buffers = {
            i: deque(maxlen=window_size) for i in range(1, n_nodes + 1)
        }

        # Statistics
        self.stats = defaultdict(int)
        self.last_stats_time = time.time()

    def process_packet(self, data):
        """Parse and buffer a raw UDP packet."""
        try:
            pkt = CSIPacket(data)
        except (ValueError, struct.error) as e:
            self.stats['parse_errors'] += 1
            return None

        self.stats[f'node_{pkt.node_id}_rx'] += 1

        # Buffer amplitude for DSP pipeline
        self.amplitude_buffers[pkt.node_id].append(pkt.amplitude)

        # Sequence-based alignment
        seq = pkt.seq
        if seq not in self.pending:
            self.pending[seq] = {}
        self.pending[seq][pkt.node_id] = pkt

        # Check for complete group
        if len(self.pending[seq]) == self.n_nodes:
            group = self.pending.pop(seq)
            self._cleanup(seq)
            self.stats['aligned_groups'] += 1
            return group

        # Cleanup stale entries
        if len(self.pending) > self.max_pending:
            self._cleanup(seq)

        return None

    def _cleanup(self, current_seq):
        stale = [s for s in self.pending if s < current_seq - 50]
        for s in stale:
            self.stats['dropped_incomplete'] += 1
            del self.pending[s]

    def print_stats(self):
        elapsed = time.time() - self.last_stats_time
        if elapsed >= 5.0:
            rates = {k: v / elapsed for k, v in self.stats.items() if '_rx' in k}
            aligned_rate = self.stats.get('aligned_groups', 0) / elapsed
            print(f"  Rates: {rates}")
            print(f"  Aligned groups: {aligned_rate:.1f}/s")
            print(f"  Errors: parse={self.stats.get('parse_errors', 0)} "
                  f"stale={self.stats.get('dropped_incomplete', 0)}")
            self.stats.clear()
            self.last_stats_time = time.time()


class UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, aggregator):
        self.aggregator = aggregator

    def datagram_received(self, data, addr):
        group = self.aggregator.process_packet(data)
        if group:
            # group is a dict {node_id: CSIPacket} — all 3 nodes aligned
            # Hand off to DSP pipeline (Module 3)
            self._forward_to_dsp(group)

        self.aggregator.print_stats()

    def _forward_to_dsp(self, group):
        """Placeholder: forward aligned group to DSP pipeline."""
        # In Module 3, this becomes the input to phase sanitization,
        # filtering, and PCA
        pass


async def main():
    print("CSI Aggregator starting on UDP :5005")
    loop = asyncio.get_event_loop()
    aggregator = Aggregator(n_nodes=3)

    transport, protocol = await loop.create_datagram_endpoint(
        lambda: UDPProtocol(aggregator),
        local_addr=('0.0.0.0', 5005)
    )

    try:
        await asyncio.sleep(float('inf'))  # Run forever
    finally:
        transport.close()


if __name__ == '__main__':
    asyncio.run(main())
```

---

## 6. ESP-IDF Project Structure

Following the ESP-IDF skill conventions for both TX and RX firmware:

```
csi-sensing/
├── tx-node/                          # TX firmware
│   ├── CMakeLists.txt
│   ├── sdkconfig.defaults
│   ├── sdkconfig.defaults.esp32s3
│   ├── partitions.csv
│   ├── main/
│   │   ├── CMakeLists.txt
│   │   └── main.c                    # Lean main: init NVS, WiFi, start TX
│   └── components/
│       ├── wifi_manager/             # STA connection to Pi AP
│       ├── csi_tx/                   # ESP-NOW frame injection
│       ├── time_sync/               # SNTP sync
│       └── health_check/            # Watchdog + heap monitor
│
├── rx-node/                          # RX firmware (same binary, different Kconfig)
│   ├── CMakeLists.txt
│   ├── sdkconfig.defaults
│   ├── sdkconfig.defaults.esp32s3
│   ├── partitions.csv
│   ├── main/
│   │   ├── CMakeLists.txt
│   │   └── main.c
│   └── components/
│       ├── wifi_manager/             # STA connection to Pi AP
│       ├── csi_stream/              # Promiscuous CSI capture + UDP streaming
│       ├── csi_config/              # NVS-based configuration
│       ├── time_sync/               # SNTP sync
│       └── health_check/
│
└── pi-aggregator/                    # Python aggregator on Raspberry Pi
    ├── pi_aggregator.py             # UDP receiver + alignment
    ├── dsp_pipeline.py              # Module 3: filtering + feature extraction
    ├── gpu_forwarder.py             # ZMQ/TCP sender to RTX 4080
    └── requirements.txt
```

**Key `sdkconfig.defaults.esp32s3` for RX nodes:**

```
# ESP32-S3 N16R8 specific
CONFIG_IDF_TARGET="esp32s3"
CONFIG_SPIRAM=y
CONFIG_SPIRAM_MODE_OCT=y
CONFIG_SPIRAM_SPEED_80M=y

# WiFi CSI
CONFIG_ESP_WIFI_CSI_ENABLED=y
CONFIG_ESP32S3_WIFI_STATIC_RX_BUFFER_NUM=16
CONFIG_ESP32S3_WIFI_DYNAMIC_RX_BUFFER_NUM=64
CONFIG_ESP32S3_WIFI_STATIC_TX_BUFFER=y
CONFIG_ESP32S3_WIFI_TX_BUFFER_TYPE=0
CONFIG_ESP32S3_WIFI_STATIC_TX_BUFFER_NUM=16

# Pin WiFi to Core 0
CONFIG_ESP32S3_WIFI_TASK_PINNED_TO_CORE_0=y

# PSRAM allocation preference
CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL=4096
CONFIG_SPIRAM_USE_MALLOC=y

# High-resolution timer
CONFIG_ESP_TIMER_PROFILING=n

# Flash size
CONFIG_ESPTOOLPY_FLASHSIZE_16MB=y
CONFIG_PARTITION_TABLE_CUSTOM=y
CONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions.csv"
```

---

## 7. Troubleshooting & Practical Tips

### Common Issues

**Problem: CSI callback fires but `info->len` is 0 or very small**
→ The promiscuous mode is capturing non-HT (legacy 802.11b/g) frames. Filter for `info->rx_ctrl.sig_mode == 1` (HT mode) to ensure you only process 802.11n frames with full LLTF+HT-LTF.

**Problem: UDP packets arrive on Pi but have corrupted headers**
→ Byte alignment issue. Ensure `__attribute__((packed))` on the header struct, and match the exact `struct.unpack` format string on the Pi side.

**Problem: One RX node gets significantly fewer CSI packets than others**
→ That node may be too far from the TX, or blocked by a wall. Check RSSI in the packet headers — if consistently below -75 dBm, move the node closer or adjust TX power with `esp_wifi_set_max_tx_power()`.

**Problem: PSRAM allocation fails at runtime**
→ Ensure `CONFIG_SPIRAM_USE_MALLOC=y` and that you're requesting PSRAM explicitly with `MALLOC_CAP_SPIRAM`. The default `malloc()` only uses internal SRAM unless the allocation exceeds `CONFIG_SPIRAM_MALLOC_ALWAYSINTERNAL` threshold.

**Problem: ESP32-S3 reboots under heavy CSI load**
→ The WiFi task stack may be too small, or the CSI callback is doing too much work. Increase WiFi task stack size and ensure the callback only does a `memcpy` + ring buffer push.

### Performance Benchmarks to Verify

After flashing all nodes and starting the Pi aggregator, verify these metrics:

| Metric | Target | Action if Not Met |
|--------|--------|-------------------|
| Per-node packet rate | 95-100 pkt/s | Check TX rate, WiFi interference |
| Aligned groups | >90/s | Check sequence numbers match |
| Packet loss | <3% | Increase socket buffer, reduce WiFi congestion |
| Latency (TX→Pi) | <5 ms | Verify all nodes on same subnet |
| RSSI per node | > -70 dBm | Move antennas, reduce obstacles |
| PSRAM free | >6 MB | Check for memory leaks in callbacks |

---

*Module 2 Complete — Next: Module 3 (Pi-side DSP & Feature Extraction) or Module 4 (GPU Deep Learning & Domain Adaptation)*
