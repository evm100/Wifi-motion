idf_component_register(
    SRCS "csi_stream.c"
    INCLUDE_DIRS "include" "${CMAKE_CURRENT_LIST_DIR}/../../../shared/include"
    REQUIRES esp_wifi log esp_timer esp_ringbuf lwip esp_psram csi_config time_sync
)
