#ifndef TIME_SYNC_H
#define TIME_SYNC_H

#include "esp_err.h"
#include <stdbool.h>
#include <stdint.h>

/**
 * Initialize SNTP time synchronization to the Pi (192.168.4.1).
 * Blocks until first sync or timeout (~15s).
 */
esp_err_t time_sync_init(void);

/**
 * Get current timestamp in microseconds using gettimeofday.
 */
int64_t get_timestamp_us(void);

/**
 * Check if time has been synchronized at least once.
 */
bool time_sync_is_synced(void);

#endif /* TIME_SYNC_H */
