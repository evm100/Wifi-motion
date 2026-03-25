/**
 * time_sync.c — SNTP synchronization to Raspberry Pi.
 */

#include "time_sync.h"

#include <sys/time.h>
#include "esp_sntp.h"
#include "esp_netif_sntp.h"
#include "esp_log.h"

static const char *TAG = "time_sync";
static bool s_synced = false;

static void time_sync_notification(struct timeval *tv)
{
    ESP_LOGI(TAG, "Time synchronized: %lld.%06ld",
             (long long)tv->tv_sec, (long)tv->tv_usec);
    s_synced = true;
}

esp_err_t time_sync_init(void)
{
    ESP_LOGI(TAG, "Initializing SNTP to %s", CONFIG_CSI_SNTP_SERVER);

    esp_sntp_config_t config = ESP_NETIF_SNTP_DEFAULT_CONFIG(CONFIG_CSI_SNTP_SERVER);
    config.sync_cb = time_sync_notification;
    esp_err_t ret = esp_netif_sntp_init(&config);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SNTP init failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /* Wait for initial sync (up to ~15 seconds) */
    int retry = 0;
    while (esp_netif_sntp_sync_wait(pdMS_TO_TICKS(2000)) != ESP_OK && retry < CONFIG_CSI_SNTP_SYNC_TIMEOUT) {
        ESP_LOGI(TAG, "Waiting for SNTP sync... (%d/%d)", ++retry, CONFIG_CSI_SNTP_SYNC_TIMEOUT);
    }

    if (!s_synced) {
        ESP_LOGW(TAG, "SNTP sync not completed within timeout");
    }

    return ESP_OK;
}

int64_t get_timestamp_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000000LL + (int64_t)tv.tv_usec;
}

bool time_sync_is_synced(void)
{
    return s_synced;
}
