idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES wifi_manager csi_tx time_sync health_check
)
