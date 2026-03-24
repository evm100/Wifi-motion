idf_component_register(
    SRCS "csi_tx.c"
    INCLUDE_DIRS "include"
    REQUIRES esp_wifi esp_timer log time_sync
)
