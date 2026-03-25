/**
 * csi_stream.c — CSI capture via promiscuous mode + UDP streaming to Pi.
 *
 * IRAM_ATTR CSI callback filters by TX MAC, builds binary packet header
 * (csi_protocol.h), copies header+CSI to a PSRAM ring buffer.
 * Separate UDP sender task on Core 1 pulls from ring buffer and sends.
 * Stats task logs packet rates every 5 seconds.
 */

#include "csi_stream.h"
#include "csi_protocol.h"
#include "csi_config.h"
#include "time_sync.h"

#include <string.h>
#include "esp_wifi.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/ringbuf.h"
#include "lwip/sockets.h"

static const char *TAG = "csi_stream";

#define CSI_RING_BUF_SIZE   (CONFIG_CSI_RING_BUF_SIZE_KB * 1024)

static RingbufHandle_t s_csi_ringbuf = NULL;
static int s_udp_sock = -1;
static struct sockaddr_in s_target_addr;

/* TX MAC address to filter */
static uint8_t s_tx_mac[6];
static bool s_tx_mac_filter = false;

/* Statistics (reset every 5s by stats task) */
static volatile uint32_t s_cb_count = 0;
static volatile uint32_t s_drop_count = 0;
static volatile uint32_t s_send_count = 0;

/**
 * Extract TX sequence number from the ESP-NOW payload embedded in the received frame.
 * ESP-NOW action frames have vendor-specific content; the TX node puts
 * seq_num (uint32_t LE) as the first 4 bytes of the ESP-NOW payload.
 * The wifi_csi_info_t doesn't directly give us the payload, but the
 * sig_len and payload_len fields tell us where data is.
 *
 * Fallback: use the WiFi sequence control field from rx_ctrl if available,
 * or a local counter.
 */
static uint32_t s_local_seq = 0;

static uint32_t extract_seq_num(const wifi_csi_info_t *info)
{
    /*
     * The ESP-NOW payload carrying the TX seq_num isn't directly accessible
     * from the CSI callback. We use the 802.11 sequence control field from
     * rx_ctrl.sig_len as a proxy for frame ordering, combined with our own
     * local counter for unique identification. The Pi aligns by TX seq_num
     * which the TX node embeds — for full alignment, the RX promiscuous
     * callback would need to also capture the raw frame. For now, use
     * a monotonic local counter that the Pi can still align across nodes
     * by timestamp proximity.
     *
     * TODO: Implement raw frame capture via promiscuous RX callback to
     * extract the actual ESP-NOW payload seq_num.
     */
    (void)info;
    return s_local_seq++;
}

/**
 * CSI callback — runs in WiFi task on Core 0.
 * MUST be fast: build header, copy to ring buffer, return.
 */
static void IRAM_ATTR csi_rx_callback(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || info->len == 0) {
        return;
    }

    /* Filter by TX MAC address */
    if (s_tx_mac_filter) {
        if (memcmp(info->mac, s_tx_mac, 6) != 0) {
            return;
        }
    }

    s_cb_count++;

    const csi_config_t *cfg = csi_config_get();

    /* Determine flags */
    uint8_t flags = 0;
    if (info->len > 128) {
        flags |= CSI_FLAG_HAS_HTLTF;
    }
    if (info->first_word_invalid) {
        flags |= CSI_FLAG_FIRST_INVALID;
    }

    /* Build packet header */
    csi_packet_header_t hdr = {
        .magic         = CSI_MAGIC,
        .version       = CSI_PROTOCOL_VER,
        .node_id       = cfg->node_id,
        .n_subcarriers = (uint16_t)(info->len / 2),
        .rssi          = info->rx_ctrl.rssi,
        .noise_floor   = info->rx_ctrl.noise_floor,
        .channel       = (uint8_t)info->rx_ctrl.channel,
        .flags         = flags,
        .seq_num       = extract_seq_num(info),
        .timestamp_us  = (uint32_t)(esp_timer_get_time() & 0xFFFFFFFF),
    };

    /* Copy header + CSI data to ring buffer (non-blocking) */
    size_t total = sizeof(hdr) + (size_t)info->len;
    uint8_t *item = NULL;

    if (xRingbufferSendAcquire(s_csi_ringbuf, (void **)&item, total, 0) == pdTRUE) {
        memcpy(item, &hdr, sizeof(hdr));
        memcpy(item + sizeof(hdr), info->buf, info->len);
        xRingbufferSendComplete(s_csi_ringbuf, item);
    } else {
        s_drop_count++;
    }
}

/**
 * UDP sender task — runs on Core 1.
 * Pulls items from ring buffer and sends via UDP to the Pi.
 */
static void udp_sender_task(void *pvParams)
{
    ESP_LOGI(TAG, "UDP sender started on core %d", xPortGetCoreID());

    while (1) {
        size_t item_size = 0;
        void *item = xRingbufferReceive(s_csi_ringbuf, &item_size, pdMS_TO_TICKS(100));

        if (item != NULL) {
            int ret = sendto(s_udp_sock, item, item_size, MSG_DONTWAIT,
                             (struct sockaddr *)&s_target_addr, sizeof(s_target_addr));
            if (ret > 0) {
                s_send_count++;
            }

            vRingbufferReturnItem(s_csi_ringbuf, item);
        }
    }
}

/**
 * Statistics reporting task — logs packet rates every 5 seconds.
 */
static void stats_task(void *pvParams)
{
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(CONFIG_CSI_STATS_INTERVAL_MS));

        uint32_t cb = s_cb_count;
        uint32_t sent = s_send_count;
        uint32_t dropped = s_drop_count;

        /* Reset counters */
        s_cb_count = 0;
        s_send_count = 0;
        s_drop_count = 0;

        ESP_LOGI(TAG, "5s stats: cb=%lu sent=%lu dropped=%lu heap=%lu psram=%lu",
                 (unsigned long)cb,
                 (unsigned long)sent,
                 (unsigned long)dropped,
                 (unsigned long)esp_get_free_heap_size(),
                 (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    }
}

esp_err_t csi_stream_init(const uint8_t *filter_mac)
{
    const csi_config_t *cfg = csi_config_get();

    /* Store TX MAC for filtering */
    if (filter_mac) {
        memcpy(s_tx_mac, filter_mac, 6);
        /* Check if it's the broadcast address (no filtering) */
        const uint8_t bcast[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
        if (memcmp(filter_mac, bcast, 6) != 0) {
            s_tx_mac_filter = true;
            ESP_LOGI(TAG, "Filtering CSI for TX MAC: %02X:%02X:%02X:%02X:%02X:%02X",
                     s_tx_mac[0], s_tx_mac[1], s_tx_mac[2],
                     s_tx_mac[3], s_tx_mac[4], s_tx_mac[5]);
        } else {
            ESP_LOGI(TAG, "TX MAC is broadcast — accepting all CSI frames");
        }
    }

    /* Create ring buffer in PSRAM */
    s_csi_ringbuf = xRingbufferCreateWithCaps(
        CSI_RING_BUF_SIZE,
        RINGBUF_TYPE_NOSPLIT,
        MALLOC_CAP_SPIRAM);
    if (!s_csi_ringbuf) {
        ESP_LOGE(TAG, "Failed to create ring buffer in PSRAM");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "Ring buffer created: %d bytes in PSRAM", CSI_RING_BUF_SIZE);

    /* Setup UDP socket */
    s_udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s_udp_sock < 0) {
        ESP_LOGE(TAG, "Socket creation failed: errno=%d", errno);
        return ESP_FAIL;
    }

    /* Increase socket send buffer */
    int sndbuf = CONFIG_CSI_UDP_SNDBUF_SIZE;
    setsockopt(s_udp_sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    memset(&s_target_addr, 0, sizeof(s_target_addr));
    s_target_addr.sin_family = AF_INET;
    s_target_addr.sin_port = htons(cfg->target_port);
    inet_pton(AF_INET, cfg->target_ip, &s_target_addr.sin_addr);

    /* Configure CSI collection */
    wifi_csi_config_t csi_cfg = {
#ifdef CONFIG_CSI_LLTF_EN
        .lltf_en = true,
#else
        .lltf_en = false,
#endif
#ifdef CONFIG_CSI_HTLTF_EN
        .htltf_en = true,
#else
        .htltf_en = false,
#endif
#ifdef CONFIG_CSI_STBC_HTLTF2_EN
        .stbc_htltf2_en = true,
#else
        .stbc_htltf2_en = false,
#endif
        .ltf_merge_en = false,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = false,
    };

    esp_err_t ret = esp_wifi_set_csi_config(&csi_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "CSI config failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = esp_wifi_set_csi_rx_cb(csi_rx_callback, NULL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "CSI callback registration failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = esp_wifi_set_csi(true);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "CSI enable failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /* Enable promiscuous mode to capture all frames (CSI from TX broadcasts) */
    ret = esp_wifi_set_promiscuous(true);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Promiscuous mode failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /* Start sender task on Core 1 (keep WiFi on Core 0) */
    BaseType_t xret = xTaskCreatePinnedToCore(
        udp_sender_task, "csi_udp_tx", CONFIG_CSI_SENDER_TASK_STACK, NULL, 5, NULL, 1);
    if (xret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create UDP sender task");
        return ESP_ERR_NO_MEM;
    }

    /* Start stats task on Core 1 */
    xret = xTaskCreatePinnedToCore(
        stats_task, "csi_stats", 2048, NULL, 1, NULL, 1);
    if (xret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create stats task");
        return ESP_ERR_NO_MEM;
    }

    ESP_LOGI(TAG, "CSI streaming to %s:%d (node %d)",
             cfg->target_ip, cfg->target_port, cfg->node_id);

    return ESP_OK;
}
