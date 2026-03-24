idf_component_register(
    SRCS "health_check.c"
    INCLUDE_DIRS "include"
    REQUIRES esp_timer esp_system log esp_hw_support
)
