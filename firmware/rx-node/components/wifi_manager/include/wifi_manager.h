#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include "esp_err.h"
#include "freertos/FreeRTOS.h"

/**
 * Initialize WiFi in STA mode and begin connection to the configured AP.
 * Must be called after esp_event_loop_create_default().
 */
esp_err_t wifi_manager_init(void);

/**
 * Block until WiFi is connected and IP is obtained, or timeout.
 * @param timeout_ticks  Maximum time to wait (use pdMS_TO_TICKS).
 * @return ESP_OK on success, ESP_ERR_TIMEOUT on timeout.
 */
esp_err_t wifi_manager_wait_connected(TickType_t timeout_ticks);

/**
 * Check if WiFi is currently connected with an IP address.
 */
bool wifi_manager_is_connected(void);

#endif /* WIFI_MANAGER_H */
