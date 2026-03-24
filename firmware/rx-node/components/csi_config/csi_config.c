/**
 * csi_config.c — NVS-based configuration loader with Kconfig fallback.
 */

#include "csi_config.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_log.h"

static const char *TAG = "csi_config";

static csi_config_t s_config;

esp_err_t csi_config_load(void)
{
    /* Start with Kconfig defaults */
    s_config.node_id = (uint8_t)CONFIG_CSI_NODE_ID;
    strncpy(s_config.target_ip, CONFIG_CSI_TARGET_IP, sizeof(s_config.target_ip) - 1);
    s_config.target_ip[sizeof(s_config.target_ip) - 1] = '\0';
    s_config.target_port = (uint16_t)CONFIG_CSI_TARGET_PORT;
    s_config.wifi_channel = (uint8_t)CONFIG_CSI_WIFI_CHANNEL;
    strncpy(s_config.tx_mac_str, CONFIG_CSI_TX_MAC, sizeof(s_config.tx_mac_str) - 1);
    s_config.tx_mac_str[sizeof(s_config.tx_mac_str) - 1] = '\0';

    /* Try to override from NVS */
    nvs_handle_t nvs;
    esp_err_t err = nvs_open("csi_cfg", NVS_READONLY, &nvs);
    if (err != ESP_OK) {
        ESP_LOGI(TAG, "No NVS config found, using Kconfig defaults");
        return ESP_OK;
    }

    uint8_t u8val;
    if (nvs_get_u8(nvs, "node_id", &u8val) == ESP_OK) {
        s_config.node_id = u8val;
        ESP_LOGI(TAG, "NVS override: node_id=%d", u8val);
    }

    size_t len = sizeof(s_config.target_ip);
    if (nvs_get_str(nvs, "target_ip", s_config.target_ip, &len) == ESP_OK) {
        ESP_LOGI(TAG, "NVS override: target_ip=%s", s_config.target_ip);
    }

    uint16_t u16val;
    if (nvs_get_u16(nvs, "target_port", &u16val) == ESP_OK) {
        s_config.target_port = u16val;
        ESP_LOGI(TAG, "NVS override: target_port=%d", u16val);
    }

    if (nvs_get_u8(nvs, "wifi_ch", &u8val) == ESP_OK) {
        s_config.wifi_channel = u8val;
        ESP_LOGI(TAG, "NVS override: wifi_channel=%d", u8val);
    }

    len = sizeof(s_config.tx_mac_str);
    if (nvs_get_str(nvs, "tx_mac", s_config.tx_mac_str, &len) == ESP_OK) {
        ESP_LOGI(TAG, "NVS override: tx_mac=%s", s_config.tx_mac_str);
    }

    nvs_close(nvs);
    return ESP_OK;
}

const csi_config_t *csi_config_get(void)
{
    return &s_config;
}

esp_err_t csi_config_parse_mac(const char *mac_str, uint8_t mac_out[6])
{
    if (!mac_str || !mac_out) {
        return ESP_ERR_INVALID_ARG;
    }

    unsigned int m[6];
    int parsed = sscanf(mac_str, "%02X:%02X:%02X:%02X:%02X:%02X",
                        &m[0], &m[1], &m[2], &m[3], &m[4], &m[5]);
    if (parsed != 6) {
        ESP_LOGE(TAG, "Failed to parse MAC: '%s'", mac_str);
        return ESP_ERR_INVALID_ARG;
    }

    for (int i = 0; i < 6; i++) {
        mac_out[i] = (uint8_t)m[i];
    }

    return ESP_OK;
}
