#ifndef HEALTH_CHECK_H
#define HEALTH_CHECK_H

#include "esp_err.h"

/**
 * Start periodic health monitoring task.
 * Logs free heap, PSRAM free, and uptime every 10 seconds.
 * Feeds the task watchdog.
 */
esp_err_t health_check_init(void);

#endif /* HEALTH_CHECK_H */
