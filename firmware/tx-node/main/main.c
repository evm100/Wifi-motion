/**
 * main.c — TX node entry point.
 *
 * Initializes NVS, event loop, then starts components in order:
 *   wifi_manager → time_sync → csi_tx → health_check
 */

#include <stdio.h>
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_event.h"

#include "wifi_manager.h"
#include "time_sync.h"
#include "csi_tx.h"
#include "health_check.h"

static const char *TAG = "main";

void app_main(void)
{
    ESP_LOGI(TAG, "CSI TX Node starting...");

    /* Initialize NVS — required for WiFi */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGW(TAG, "NVS partition truncated, erasing...");
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* Create default event loop */
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /* Start WiFi as STA, connect to Pi AP */
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

    /* Sync time via SNTP to Pi */
    ret = time_sync_init();
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Time sync init failed: %s (continuing anyway)", esp_err_to_name(ret));
    }

    /* Start ESP-NOW broadcast */
    ret = csi_tx_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "CSI TX init failed: %s", esp_err_to_name(ret));
        return;
    }

    /* Start health monitoring */
    ret = health_check_init();
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "Health check init failed: %s", esp_err_to_name(ret));
    }

    ESP_LOGI(TAG, "TX Node initialized successfully");
}
