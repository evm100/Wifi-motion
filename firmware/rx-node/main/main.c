/**
 * main.c — RX node entry point.
 *
 * Initializes NVS, event loop, loads config, then starts components:
 *   csi_config → wifi_manager → time_sync → csi_stream → health_check
 */

#include <stdio.h>
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_event.h"

#include "wifi_manager.h"
#include "time_sync.h"
#include "csi_stream.h"
#include "csi_config.h"
#include "health_check.h"

static const char *TAG = "main";

void app_main(void)
{
    ESP_LOGI(TAG, "CSI RX Node starting...");

    /* Initialize NVS — required for WiFi and config storage */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGW(TAG, "NVS partition truncated, erasing...");
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* Create default event loop */
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /* Load configuration from NVS (falls back to Kconfig defaults) */
    ret = csi_config_load();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Config load failed: %s", esp_err_to_name(ret));
        return;
    }

    const csi_config_t *cfg = csi_config_get();
    ESP_LOGI(TAG, "Config: node_id=%d target=%s:%d channel=%d",
             cfg->node_id, cfg->target_ip, cfg->target_port, cfg->wifi_channel);

    /* Start WiFi as STA, connect to hotspot */
    ret = wifi_manager_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "WiFi manager init failed: %s", esp_err_to_name(ret));
        return;
    }

    /* Wait for IP before continuing */
    ret = wifi_manager_wait_connected(pdMS_TO_TICKS(30000));
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "WiFi connection timed out");
        return;
    }

    /* Sync time via SNTP */
    ret = time_sync_init();
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Time sync init failed: %s (continuing anyway)", esp_err_to_name(ret));
    }

    /* Parse TX MAC from config string */
    uint8_t tx_mac[6];
    ret = csi_config_parse_mac(cfg->tx_mac_str, tx_mac);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Invalid TX MAC address: %s", cfg->tx_mac_str);
        return;
    }

    /* Start CSI capture + UDP streaming */
    ret = csi_stream_init(tx_mac);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "CSI stream init failed: %s", esp_err_to_name(ret));
        return;
    }

    /* Start health monitoring */
    ret = health_check_init();
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Health check init failed: %s", esp_err_to_name(ret));
    }

    ESP_LOGI(TAG, "RX Node %d initialized successfully", cfg->node_id);
}
