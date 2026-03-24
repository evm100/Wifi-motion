#ifndef CSI_CONFIG_H
#define CSI_CONFIG_H

#include "esp_err.h"
#include <stdint.h>

typedef struct {
    uint8_t  node_id;
    char     target_ip[16];
    uint16_t target_port;
    uint8_t  wifi_channel;
    char     tx_mac_str[18];   /* "AA:BB:CC:DD:EE:FF\0" */
} csi_config_t;

/**
 * Load configuration from NVS, falling back to Kconfig defaults.
 * Must be called after nvs_flash_init().
 */
esp_err_t csi_config_load(void);

/**
 * Get a pointer to the loaded configuration (read-only).
 */
const csi_config_t *csi_config_get(void);

/**
 * Parse a MAC address string "AA:BB:CC:DD:EE:FF" into a 6-byte array.
 * @return ESP_OK on success, ESP_ERR_INVALID_ARG on parse failure.
 */
esp_err_t csi_config_parse_mac(const char *mac_str, uint8_t mac_out[6]);

#endif /* CSI_CONFIG_H */
