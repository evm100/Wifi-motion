#ifndef CSI_STREAM_H
#define CSI_STREAM_H

#include "esp_err.h"
#include <stdint.h>

/**
 * Initialize CSI capture and UDP streaming.
 *
 * Enables promiscuous mode, registers CSI callback with TX MAC filter,
 * creates PSRAM ring buffer, and starts UDP sender + stats tasks.
 *
 * @param filter_mac  6-byte TX MAC address to filter. NULL to accept all.
 * @return ESP_OK on success.
 */
esp_err_t csi_stream_init(const uint8_t *filter_mac);

#endif /* CSI_STREAM_H */
