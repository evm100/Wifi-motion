/**
 * wifi_manager.c — STA connection to WiFi hotspot with auto-reconnect.
 *
 * Uses CONFIG_CSI_WIFI_SSID, CONFIG_CSI_WIFI_PASSWORD, CONFIG_CSI_WIFI_CHANNEL
 * from the csi_config Kconfig menu.
 */

#include "wifi_manager.h"

#include <string.h>
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

static const char *TAG = "wifi_manager";

#define WIFI_CONNECTED_BIT  BIT0
#define WIFI_FAIL_BIT       BIT1

static EventGroupHandle_t s_wifi_event_group;
static int s_retry_count = 0;
static bool s_connected = false;

static void event_handler(void *arg, esp_event_base_t event_base,
                          int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        ESP_LOGI(TAG, "STA started, connecting...");
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        s_connected = false;
        wifi_event_sta_disconnected_t *disconn = (wifi_event_sta_disconnected_t *)event_data;
        s_retry_count++;

        /* Always retry — phone hotspots are flaky and CSI capture
         * depends on staying on the correct channel. */
        ESP_LOGW(TAG, "Disconnected (reason=%d), retry #%d...",
                 disconn->reason, s_retry_count);
        vTaskDelay(pdMS_TO_TICKS(1000));  /* back off 1 s to avoid scan storm */
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_count = 0;
        s_connected = true;

        /* Re-enforce power save off after every (re)association */
        esp_wifi_set_ps(WIFI_PS_NONE);

        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

esp_err_t wifi_manager_init(void)
{
    s_wifi_event_group = xEventGroupCreate();
    if (!s_wifi_event_group) {
        return ESP_ERR_NO_MEM;
    }

    /* Initialize TCP/IP stack */
    esp_err_t ret = esp_netif_init();
    if (ret != ESP_OK) return ret;

    esp_netif_create_default_wifi_sta();

    /* Initialize WiFi with default config */
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ret = esp_wifi_init(&cfg);
    if (ret != ESP_OK) return ret;

    /* Register event handlers */
    ret = esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &event_handler, NULL, NULL);
    if (ret != ESP_OK) return ret;

    ret = esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &event_handler, NULL, NULL);
    if (ret != ESP_OK) return ret;

    /* Configure STA */
    wifi_config_t wifi_config = {
        .sta = {
            .ssid = CONFIG_CSI_WIFI_SSID,
            .password = CONFIG_CSI_WIFI_PASSWORD,
            .channel = CONFIG_CSI_WIFI_CHANNEL,
            .threshold.authmode = WIFI_AUTH_WPA_PSK,
        },
    };

    ret = esp_wifi_set_mode(WIFI_MODE_STA);
    if (ret != ESP_OK) return ret;

    ret = esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    if (ret != ESP_OK) return ret;

    ret = esp_wifi_start();
    if (ret != ESP_OK) return ret;

    /* Disable power save — the radio must stay on continuously so
     * promiscuous-mode CSI capture does not miss frames.  Default
     * WIFI_PS_MIN_MODEM sleeps between beacons and drops most CSI. */
    esp_wifi_set_ps(WIFI_PS_NONE);

    ESP_LOGI(TAG, "WiFi STA init complete, connecting to SSID: %s (ch %d)",
             CONFIG_CSI_WIFI_SSID, CONFIG_CSI_WIFI_CHANNEL);

    return ESP_OK;
}

esp_err_t wifi_manager_wait_connected(TickType_t timeout_ticks)
{
    EventBits_t bits = xEventGroupWaitBits(
        s_wifi_event_group,
        WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
        pdFALSE, pdFALSE,
        timeout_ticks);

    if (bits & WIFI_CONNECTED_BIT) {
        return ESP_OK;
    }
    return ESP_ERR_TIMEOUT;
}

bool wifi_manager_is_connected(void)
{
    return s_connected;
}
