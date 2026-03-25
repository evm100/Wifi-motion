/**
 * health_check.c — Periodic health monitoring.
 *
 * Logs heap, PSRAM, and uptime stats at a configurable interval.
 * Does NOT subscribe to the Task Watchdog — the idle-task WDT on both
 * cores already provides system-level watchdog coverage.  Subscribing a
 * monitoring task whose sleep interval equals the WDT timeout causes a
 * guaranteed WDT trip (sleep + logging > timeout).
 */

#include "health_check.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "health_check";

#define HEALTH_CHECK_INTERVAL_MS  CONFIG_CSI_HEALTH_INTERVAL_MS

static void health_check_task(void *pvParams)
{
    while (1) {
        int64_t uptime_s = esp_timer_get_time() / 1000000LL;

        ESP_LOGI(TAG, "uptime=%llds heap=%lu psram=%lu min_heap=%lu",
                 (long long)uptime_s,
                 (unsigned long)esp_get_free_heap_size(),
                 (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                 (unsigned long)esp_get_minimum_free_heap_size());

        vTaskDelay(pdMS_TO_TICKS(HEALTH_CHECK_INTERVAL_MS));
    }
}

esp_err_t health_check_init(void)
{
    ESP_LOGI(TAG, "Starting health check (interval=%d ms)", HEALTH_CHECK_INTERVAL_MS);

    BaseType_t ret = xTaskCreatePinnedToCore(
        health_check_task, "health_chk", CONFIG_CSI_HEALTH_TASK_STACK, NULL, 1, NULL, 1);
    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create health check task");
        return ESP_ERR_NO_MEM;
    }

    return ESP_OK;
}
