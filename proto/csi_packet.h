/**
 * csi_packet.h — Shared binary packet header for WiFi CSI streaming.
 *
 * This is the contract between ESP32-S3 firmware (C) and Raspberry Pi (Python).
 * Total header: 20 bytes. Payload: n_subcarriers * 2 bytes of int8 I/Q pairs.
 *
 * Byte order: little-endian (matches ESP32-S3 and x86/ARM).
 */

#ifndef CSI_PACKET_H
#define CSI_PACKET_H

#include <stdint.h>

#define CSI_MAGIC           0xC5110001
#define CSI_PROTOCOL_VER    1

/* Flag bits */
#define CSI_FLAG_HAS_HTLTF      (1 << 0)  /* HT-LTF data present in payload */
#define CSI_FLAG_FIRST_INVALID  (1 << 1)  /* First 4 bytes of CSI payload invalid */

typedef struct __attribute__((packed)) {
    uint32_t magic;          /* Always CSI_MAGIC (0xC5110001) */
    uint8_t  version;        /* Protocol version (CSI_PROTOCOL_VER) */
    uint8_t  node_id;        /* RX node identifier (1, 2, or 3) */
    uint16_t n_subcarriers;  /* Number of I/Q pairs in payload */
    int8_t   rssi;           /* Received signal strength (dBm) */
    int8_t   noise_floor;    /* RF noise floor (dBm) */
    uint8_t  channel;        /* WiFi channel number */
    uint8_t  flags;          /* Bit flags (CSI_FLAG_*) */
    uint32_t seq_num;        /* TX sequence number */
    uint32_t timestamp_us;   /* Local microsecond timestamp (low 32 bits) */
} csi_packet_header_t;

_Static_assert(sizeof(csi_packet_header_t) == 20, "Header must be exactly 20 bytes");

#define CSI_HEADER_SIZE  sizeof(csi_packet_header_t)

#endif /* CSI_PACKET_H */
