/**
 * health_check.c — Periodic health monitoring and watchdog feeding.
 */

#include "health_check.h"

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_task_wdt.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "health_check";

#define HEALTH_CHECK_INTERVAL_MS  10000

static void health_check_task(void *pvParams)
{
    /* Subscribe this task to the task watchdog */
    esp_task_wdt_add(NULL);

    while (1) {
        int64_t uptime_s = esp_timer_get_time() / 1000000LL;

        ESP_LOGI(TAG, "uptime=%llds heap=%lu psram=%lu min_heap=%lu",
                 (long long)uptime_s,
                 (unsigned long)esp_get_free_heap_size(),
                 (unsigned long)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                 (unsigned long)esp_get_minimum_free_heap_size());

        /* Feed the watchdog */
        esp_task_wdt_reset();

        vTaskDelay(pdMS_TO_TICKS(HEALTH_CHECK_INTERVAL_MS));
    }
}

esp_err_t health_check_init(void)
{
    ESP_LOGI(TAG, "Starting health check (interval=%d ms)", HEALTH_CHECK_INTERVAL_MS);

    BaseType_t ret = xTaskCreatePinnedToCore(
        health_check_task, "health_chk", 2048, NULL, 1, NULL, 1);
    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create health check task");
        return ESP_ERR_NO_MEM;
    }

    return ESP_OK;
}
