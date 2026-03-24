#ifndef CSI_TX_H
#define CSI_TX_H

#include "esp_err.h"

/**
 * Initialize ESP-NOW and start broadcasting frames at CONFIG_CSI_TX_RATE_HZ.
 * Each broadcast carries an incrementing sequence number in the payload.
 * Must be called after WiFi is initialized and connected.
 */
esp_err_t csi_tx_init(void);

/**
 * Get the current TX sequence number.
 */
uint32_t csi_tx_get_seq_num(void);

#endif /* CSI_TX_H */
