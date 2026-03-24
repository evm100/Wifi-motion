"""
constants.py — Shared constants for WiFi CSI sensing pipeline.

Physical, protocol, and DSP constants used across edge and GPU tiers.
"""

# --- WiFi Physical Constants ---
WIFI_FREQUENCY_HZ = 2_437_000_000   # Channel 6 center frequency
WIFI_WAVELENGTH_M = 0.125           # ~c / 2.4 GHz
WIFI_CHANNEL = 6
WIFI_BANDWIDTH_MHZ = 20             # HT20 mode
SUBCARRIER_SPACING_KHZ = 312.5

# --- CSI Subcarrier Layout (HT20, LLTF + HT-LTF) ---
# LLTF: 52 valid subcarriers (indices 6..31 and 33..58)
LLTF_VALID = list(range(6, 32)) + list(range(33, 59))  # 52

# HT-LTF: 56 valid subcarriers (indices 66..93 and 100..127)
HTLTF_VALID = list(range(66, 94)) + list(range(100, 128))  # 56

# Combined: 108 valid subcarriers
ALL_VALID = LLTF_VALID + HTLTF_VALID  # 108

N_LLTF = len(LLTF_VALID)       # 52
N_HTLTF = len(HTLTF_VALID)     # 56
N_VALID_SUBCARRIERS = len(ALL_VALID)  # 108

# --- CSI Sampling ---
CSI_SAMPLING_RATE_HZ = 100
CSI_PACKET_SIZE = 276           # 20 header + 256 payload
CSI_PAYLOAD_SIZE = 256          # 128 subcarriers * 2 bytes (I/Q as int8)
PER_NODE_BANDWIDTH_BPS = CSI_PACKET_SIZE * CSI_SAMPLING_RATE_HZ  # ~27.6 KB/s

# --- Protocol ---
CSI_MAGIC = 0xC5110001
CSI_PROTOCOL_VER = 1
CSI_HEADER_SIZE = 20
UDP_PORT = 5005
N_RX_NODES = 3
