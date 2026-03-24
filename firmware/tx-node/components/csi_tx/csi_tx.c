/**
 * csi_tx.c — ESP-NOW broadcast sender at configurable rate.
 *
 * Sends broadcast frames with an incrementing 4-byte sequence number
 * as the payload. Uses vTaskDelayUntil for precise timing.
 * Task pinned to Core 0 (same as WiFi stack).
 */

#include "csi_tx.h"

#include <string.h>
#include "esp_mac.h"
#include "esp_now.h"
#include "esp_wifi.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "time_sync.h"

static const char *TAG = "csi_tx";

/* Broadcast MAC address */
static const uint8_t s_broadcast_mac[ESP_NOW_ETH_ALEN] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF
};

/* TX payload: seq_num (4 bytes) + timestamp_us (4 bytes) */
typedef struct __attribute__((packed)) {
    uint32_t seq_num;
    uint32_t timestamp_us;
} csi_tx_payload_t;

static volatile uint32_t s_seq_num = 0;

static void csi_tx_task(void *pvParams)
{
    const TickType_t period_ticks = pdMS_TO_TICKS(1000 / CONFIG_CSI_TX_RATE_HZ);
    TickType_t last_wake = xTaskGetTickCount();
    csi_tx_payload_t payload;

    ESP_LOGI(TAG, "TX task started on core %d, rate=%d Hz, period=%lu ticks",
             xPortGetCoreID(), CONFIG_CSI_TX_RATE_HZ, (unsigned long)period_ticks);

    while (1) {
        payload.seq_num = s_seq_num++;
        payload.timestamp_us = (uint32_t)(get_timestamp_us() & 0xFFFFFFFF);

        esp_err_t ret = esp_now_send(s_broadcast_mac,
                                     (const uint8_t *)&payload,
                                     sizeof(payload));
        if (ret != ESP_OK) {
            ESP_LOGD(TAG, "esp_now_send failed: %s", esp_err_to_name(ret));
        }

        vTaskDelayUntil(&last_wake, period_ticks);
    }
}

static void esp_now_send_cb(const esp_now_send_info_t *tx_info, esp_now_send_status_t status)
{
    if (status != ESP_NOW_SEND_SUCCESS) {
        ESP_LOGD(TAG, "ESP-NOW send failed to %02x:%02x:%02x:%02x:%02x:%02x",
                 MAC2STR(tx_info->des_addr));
    }
}

esp_err_t csi_tx_init(void)
{
    ESP_LOGI(TAG, "Initializing ESP-NOW TX at %d Hz", CONFIG_CSI_TX_RATE_HZ);

    /* Initialize ESP-NOW */
    esp_err_t ret = esp_now_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "esp_now_init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /* Register send callback */
    ret = esp_now_register_send_cb(esp_now_send_cb);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to register send callback: %s", esp_err_to_name(ret));
        return ret;
    }

    /* Add broadcast peer */
    esp_now_peer_info_t peer = {
        .channel = CONFIG_CSI_WIFI_CHANNEL,
        .ifidx = WIFI_IF_STA,
        .encrypt = false,
    };
    memcpy(peer.peer_addr, s_broadcast_mac, ESP_NOW_ETH_ALEN);

    ret = esp_now_add_peer(&peer);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to add broadcast peer: %s", esp_err_to_name(ret));
        return ret;
    }

    /* Start TX task on Core 0 (WiFi core) */
    BaseType_t xret = xTaskCreatePinnedToCore(
        csi_tx_task, "csi_tx", 4096, NULL, 5, NULL, 0);
    if (xret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create TX task");
        return ESP_ERR_NO_MEM;
    }

    return ESP_OK;
}

uint32_t csi_tx_get_seq_num(void)
{
    return s_seq_num;
}
