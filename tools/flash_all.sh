#!/usr/bin/env bash
# flash_all.sh — Interactive provisioning and sequential flashing for all 4 ESP32-S3 nodes
#
# Prompts for WiFi credentials and node configuration, then builds firmware with
# those values baked in and flashes each board in sequence:
#   1. TX node  (ESP-NOW broadcaster)
#   2. RX node 1 (CSI receiver, node_id=1)
#   3. RX node 2 (CSI receiver, node_id=2)
#   4. RX node 3 (CSI receiver, node_id=3)
#
# Each RX node gets a unique CONFIG_CSI_NODE_ID compiled in. The TX node and all
# RX nodes share the same WiFi credentials and channel.
#
# Usage:
#   ./tools/flash_all.sh
#
# Requirements:
#   - ESP-IDF 5.5+ environment activated (source export.sh or 'esp' alias)
#   - All 4 boards connected via USB (can be plugged in one at a time if prompted)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TX_DIR="$REPO_ROOT/firmware/tx-node"
RX_DIR="$REPO_ROOT/firmware/rx-node"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# --- Helpers ---

prompt() {
    local varname="$1" text="$2" default="$3" input
    read -rp "$(echo -e "  ${CYAN}${text}${NC} [${DIM}${default}${NC}]: ")" input
    eval "$varname=\"\${input:-\$default}\""
}

prompt_password() {
    local varname="$1" text="$2" default="$3" input
    read -rsp "$(echo -e "  ${CYAN}${text}${NC} [${DIM}********${NC}]: ")" input
    echo
    eval "$varname=\"\${input:-\$default}\""
}

separator() {
    echo -e "${DIM}--------------------------------------------${NC}"
}

# --- Prerequisites ---

if ! command -v idf.py &>/dev/null; then
    echo -e "${RED}Error: idf.py not found in PATH.${NC}"
    echo "  Activate the ESP-IDF environment first:"
    echo "    source ~/esp/esp-idf/export.sh"
    echo "    (or use your 'esp' alias)"
    exit 1
fi

echo -e "${BOLD}"
echo "============================================"
echo "  WiFi CSI Sensing — Provision & Flash"
echo "  1 TX node + 3 RX nodes (ESP32-S3)"
echo "============================================"
echo -e "${NC}"

# =============================================
# Step 1: Gather configuration
# =============================================

echo -e "${BOLD}Network Configuration${NC}"
prompt         WIFI_SSID     "WiFi AP SSID"            "CSI_SENSING_NET"
prompt_password WIFI_PASS    "WiFi AP password"         "changeme123"
prompt         WIFI_CHANNEL  "WiFi channel (1-13)"      "6"
prompt         TARGET_IP     "Pi aggregator IP"         "192.168.4.1"
prompt         TARGET_PORT   "Pi aggregator UDP port"   "5005"
echo

echo -e "${BOLD}TX Node Configuration${NC}"
prompt         TX_RATE       "TX frame rate in Hz"      "100"
echo

echo -e "${BOLD}RX Node Configuration${NC}"
echo -e "  ${DIM}The TX MAC lets RX nodes filter CSI to only frames from your TX board.${NC}"
echo -e "  ${DIM}Use FF:FF:FF:FF:FF:FF to accept frames from any transmitter.${NC}"
echo -e "  ${DIM}Tip: run 'esptool.py --port <TX_PORT> read_mac' to find your TX MAC.${NC}"
prompt         TX_MAC        "TX node MAC filter"       "FF:FF:FF:FF:FF:FF"
echo

# =============================================
# Step 2: Assign serial ports
# =============================================

echo -e "${BOLD}Serial Port Assignment${NC}"

# Auto-detect available ports
DETECTED_PORTS=()
for dev in /dev/ttyUSB* /dev/ttyACM*; do
    [ -e "$dev" ] && DETECTED_PORTS+=("$dev")
done
if [ ${#DETECTED_PORTS[@]} -gt 0 ]; then
    echo -e "  Detected ports: ${GREEN}${DETECTED_PORTS[*]}${NC}"
else
    echo -e "  ${YELLOW}No serial ports detected. Enter paths manually.${NC}"
fi
echo -e "  ${DIM}Plug boards in one at a time to identify which port maps to which node.${NC}"
echo

DEF_TX="${DETECTED_PORTS[0]:-/dev/ttyUSB0}"
DEF_RX1="${DETECTED_PORTS[1]:-/dev/ttyUSB1}"
DEF_RX2="${DETECTED_PORTS[2]:-/dev/ttyUSB2}"
DEF_RX3="${DETECTED_PORTS[3]:-/dev/ttyUSB3}"

prompt TX_PORT    "TX  node serial port"          "$DEF_TX"
prompt RX1_PORT   "RX1 node serial port (id=1)"   "$DEF_RX1"
prompt RX2_PORT   "RX2 node serial port (id=2)"   "$DEF_RX2"
prompt RX3_PORT   "RX3 node serial port (id=3)"   "$DEF_RX3"
echo

# Validate ports
WARN_COUNT=0
for pair in "TX:$TX_PORT" "RX1:$RX1_PORT" "RX2:$RX2_PORT" "RX3:$RX3_PORT"; do
    label="${pair%%:*}"
    port="${pair#*:}"
    if [ ! -e "$port" ]; then
        echo -e "  ${YELLOW}Warning: $port ($label) does not exist yet${NC}"
        WARN_COUNT=$((WARN_COUNT + 1))
    fi
done

# Check for duplicates
ALL_PORTS=("$TX_PORT" "$RX1_PORT" "$RX2_PORT" "$RX3_PORT")
SORTED_PORTS=($(printf '%s\n' "${ALL_PORTS[@]}" | sort))
for ((i=1; i<${#SORTED_PORTS[@]}; i++)); do
    if [ "${SORTED_PORTS[$i]}" = "${SORTED_PORTS[$((i-1))]}" ]; then
        echo -e "  ${RED}Error: Duplicate port ${SORTED_PORTS[$i]} — each node needs its own port.${NC}"
        exit 1
    fi
done

# =============================================
# Step 3: Confirm
# =============================================

echo
separator
echo -e "${BOLD}  Configuration Summary${NC}"
separator
echo "  WiFi SSID:        $WIFI_SSID"
echo "  WiFi Channel:     $WIFI_CHANNEL"
echo "  Aggregator:       $TARGET_IP:$TARGET_PORT"
echo "  TX Rate:          ${TX_RATE} Hz"
echo "  TX MAC Filter:    $TX_MAC"
separator
echo "  TX  → $TX_PORT"
echo "  RX1 → $RX1_PORT   (node_id=1)"
echo "  RX2 → $RX2_PORT   (node_id=2)"
echo "  RX3 → $RX3_PORT   (node_id=3)"
separator
echo

read -rp "$(echo -e "${YELLOW}Proceed with build and flash? [Y/n]: ${NC}")" CONFIRM
if [[ "${CONFIRM,,}" == n* ]]; then
    echo "Aborted."
    exit 0
fi

# =============================================
# Step 4: Generate sdkconfig override files
# =============================================

TX_PROVISION="$TX_DIR/sdkconfig.provision"
RX_PROVISION="$RX_DIR/sdkconfig.provision"

# ESP-IDF merges these in order; later files win.
# sdkconfig.defaults → sdkconfig.defaults.esp32s3 → sdkconfig.provision
TX_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.esp32s3;sdkconfig.provision"
RX_DEFAULTS="sdkconfig.defaults;sdkconfig.defaults.esp32s3;sdkconfig.provision"

write_tx_provision() {
    cat > "$TX_PROVISION" <<PROV
# Auto-generated by flash_all.sh — do not check in
CONFIG_CSI_WIFI_SSID="$WIFI_SSID"
CONFIG_CSI_WIFI_PASSWORD="$WIFI_PASS"
CONFIG_CSI_WIFI_CHANNEL=$WIFI_CHANNEL
CONFIG_CSI_TX_RATE_HZ=$TX_RATE
PROV
}

write_rx_provision() {
    local node_id="$1"
    cat > "$RX_PROVISION" <<PROV
# Auto-generated by flash_all.sh — do not check in
CONFIG_CSI_NODE_ID=$node_id
CONFIG_CSI_WIFI_SSID="$WIFI_SSID"
CONFIG_CSI_WIFI_PASSWORD="$WIFI_PASS"
CONFIG_CSI_WIFI_CHANNEL=$WIFI_CHANNEL
CONFIG_CSI_TARGET_IP="$TARGET_IP"
CONFIG_CSI_TARGET_PORT=$TARGET_PORT
CONFIG_CSI_TX_MAC="$TX_MAC"
PROV
}

# Clean up provision files on exit (success or failure)
cleanup() {
    rm -f "$TX_PROVISION" "$RX_PROVISION"
}
trap cleanup EXIT

# =============================================
# Step 5: Build and flash each node
# =============================================

declare -A RESULTS
TOTAL=4
STEP=0
FAIL_COUNT=0
START_TIME=$SECONDS

# --- TX Node ---

STEP=$((STEP + 1))
echo
echo -e "${BOLD}[$STEP/$TOTAL] TX Node → $TX_PORT${NC}"
separator

write_tx_provision

# Full clean build (set-target wipes build dir)
rm -f "$TX_DIR/sdkconfig"
echo -e "  ${DIM}Setting target and building...${NC}"
if idf.py -C "$TX_DIR" \
    -DSDKCONFIG_DEFAULTS="$TX_DEFAULTS" \
    set-target esp32s3 2>&1 | tail -3 \
  && idf.py -C "$TX_DIR" build 2>&1 | tail -5; then
    echo -e "  ${GREEN}Build OK${NC}"
else
    echo -e "  ${RED}Build FAILED${NC}"
    RESULTS["TX → $TX_PORT"]="BUILD_FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

if [ "${RESULTS["TX → $TX_PORT"]:-}" != "BUILD_FAIL" ]; then
    echo -e "  ${DIM}Flashing...${NC}"
    if idf.py -C "$TX_DIR" -p "$TX_PORT" flash 2>&1 | tail -5; then
        RESULTS["TX → $TX_PORT"]="OK"
        echo -e "  ${GREEN}TX node flashed successfully${NC}"
    else
        RESULTS["TX → $TX_PORT"]="FLASH_FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo -e "  ${RED}TX node flash FAILED${NC}"
    fi
fi

# --- RX Nodes ---

RX_TARGET_SET=false

for node_id in 1 2 3; do
    STEP=$((STEP + 1))
    port_var="RX${node_id}_PORT"
    port="${!port_var}"
    label="RX${node_id} (id=${node_id}) → $port"

    echo
    echo -e "${BOLD}[$STEP/$TOTAL] $label${NC}"
    separator

    write_rx_provision "$node_id"
    rm -f "$RX_DIR/sdkconfig"

    BUILD_OK=true

    if [ "$RX_TARGET_SET" = "false" ]; then
        # First RX build: full set-target (cleans build dir)
        echo -e "  ${DIM}Setting target and building (full build)...${NC}"
        if idf.py -C "$RX_DIR" \
            -DSDKCONFIG_DEFAULTS="$RX_DEFAULTS" \
            set-target esp32s3 2>&1 | tail -3 \
          && idf.py -C "$RX_DIR" build 2>&1 | tail -5; then
            echo -e "  ${GREEN}Build OK${NC}"
        else
            echo -e "  ${RED}Build FAILED${NC}"
            BUILD_OK=false
        fi
        RX_TARGET_SET=true
    else
        # Subsequent RX builds: incremental (only node_id changed)
        echo -e "  ${DIM}Rebuilding with node_id=$node_id (incremental)...${NC}"
        if idf.py -C "$RX_DIR" \
            -DSDKCONFIG_DEFAULTS="$RX_DEFAULTS" \
            build 2>&1 | tail -5; then
            echo -e "  ${GREEN}Build OK${NC}"
        else
            echo -e "  ${RED}Build FAILED${NC}"
            BUILD_OK=false
        fi
    fi

    if [ "$BUILD_OK" = "true" ]; then
        echo -e "  ${DIM}Flashing...${NC}"
        if idf.py -C "$RX_DIR" -p "$port" flash 2>&1 | tail -5; then
            RESULTS["$label"]="OK"
            echo -e "  ${GREEN}RX node $node_id flashed successfully${NC}"
        else
            RESULTS["$label"]="FLASH_FAIL"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo -e "  ${RED}RX node $node_id flash FAILED${NC}"
        fi
    else
        RESULTS["$label"]="BUILD_FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# =============================================
# Step 6: Summary
# =============================================

ELAPSED=$(( SECONDS - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

echo
echo -e "${BOLD}============================================"
echo "  Flash Summary   (${MINUTES}m ${SECS}s elapsed)"
echo -e "============================================${NC}"

for key in $(printf '%s\n' "${!RESULTS[@]}" | sort); do
    status="${RESULTS[$key]}"
    if [ "$status" = "OK" ]; then
        echo -e "  ${GREEN}✓${NC} $key"
    else
        echo -e "  ${RED}✗${NC} $key  — $status"
    fi
done

echo
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}All 4 nodes provisioned and flashed successfully!${NC}"
    echo
    echo -e "${DIM}Next steps:${NC}"
    echo -e "${DIM}  Monitor TX:  idf.py -p $TX_PORT monitor${NC}"
    echo -e "${DIM}  Monitor RX1: idf.py -p $RX1_PORT monitor${NC}"
    echo -e "${DIM}  Read TX MAC: esptool.py --port $TX_PORT read_mac${NC}"
    exit 0
else
    echo -e "${RED}${FAIL_COUNT} node(s) failed. Review the output above.${NC}"
    exit 1
fi
