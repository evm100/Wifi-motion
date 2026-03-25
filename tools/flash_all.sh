#!/usr/bin/env bash
# flash_all.sh — Interactive provisioning and sequential flashing for all 4 ESP32-S3 nodes
#
# Prompts for WiFi credentials and node configuration, then builds and flashes
# each board one at a time through the SAME USB port. The user swaps boards
# between flashes.
#
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

wait_for_board() {
    local label="$1"
    echo
    echo -e "  ${YELLOW}>>> Plug in the ${BOLD}${label}${NC}${YELLOW} board via USB <<<${NC}"
    read -rp "$(echo -e "  ${CYAN}Press Enter when ready (or 's' to skip)...${NC} ")" REPLY
    if [[ "${REPLY,,}" == s* ]]; then
        return 1
    fi
    return 0
}

detect_port() {
    # Find the first available serial port
    for dev in /dev/ttyUSB* /dev/ttyACM*; do
        [ -e "$dev" ] && { echo "$dev"; return 0; }
    done
    return 1
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
echo -e "  ${DIM}Boards are flashed one at a time through the same USB port.${NC}"
echo -e "  ${DIM}You will be prompted to swap boards between each flash.${NC}"
echo

# =============================================
# Step 1: Gather configuration
# =============================================

echo -e "${BOLD}Network Configuration${NC}"
prompt         WIFI_SSID     "WiFi hotspot SSID"                "iPhone"
prompt_password WIFI_PASS    "WiFi hotspot password"             "changeme123"
prompt         WIFI_CHANNEL  "WiFi channel (0 = auto-scan)"      "0"
prompt         TARGET_IP     "Pi aggregator IP"                  "172.20.10.2"
prompt         TARGET_PORT   "Pi aggregator UDP port"            "5005"
echo

echo -e "${BOLD}TX Node Configuration${NC}"
prompt         TX_RATE       "TX frame rate in Hz"               "100"
echo

echo -e "${BOLD}RX Node Configuration${NC}"
echo -e "  ${DIM}The TX MAC lets RX nodes filter CSI to only frames from your TX board.${NC}"
echo -e "  ${DIM}Use FF:FF:FF:FF:FF:FF to accept frames from any transmitter.${NC}"
echo -e "  ${DIM}Tip: after flashing TX, run 'esptool.py --port <PORT> read_mac' to find its MAC.${NC}"
prompt         TX_MAC        "TX node MAC filter"                "FF:FF:FF:FF:FF:FF"
echo

echo -e "${BOLD}Serial Port${NC}"
DETECTED=$(detect_port 2>/dev/null || true)
if [ -n "$DETECTED" ]; then
    echo -e "  Detected: ${GREEN}${DETECTED}${NC}"
fi
prompt         FLASH_PORT    "Serial port for flashing"          "${DETECTED:-/dev/ttyUSB0}"
echo

# =============================================
# Step 2: Confirm
# =============================================

separator
echo -e "${BOLD}  Configuration Summary${NC}"
separator
echo "  WiFi SSID:        $WIFI_SSID"
echo "  WiFi Channel:     $WIFI_CHANNEL (0 = auto)"
echo "  Aggregator:       $TARGET_IP:$TARGET_PORT"
echo "  TX Rate:          ${TX_RATE} Hz"
echo "  TX MAC Filter:    $TX_MAC"
echo "  Flash port:       $FLASH_PORT"
separator
echo "  Flash order:"
echo "    1. TX  node"
echo "    2. RX1 node  (node_id=1)"
echo "    3. RX2 node  (node_id=2)"
echo "    4. RX3 node  (node_id=3)"
separator
echo

read -rp "$(echo -e "${YELLOW}Proceed? [Y/n]: ${NC}")" CONFIRM
if [[ "${CONFIRM,,}" == n* ]]; then
    echo "Aborted."
    exit 0
fi

# =============================================
# Step 3: Generate sdkconfig override files
# =============================================

TX_PROVISION="$TX_DIR/sdkconfig.provision"
RX_PROVISION="$RX_DIR/sdkconfig.provision"

# ESP-IDF merges these in order; later files win.
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

cleanup() {
    rm -f "$TX_PROVISION" "$RX_PROVISION"
}
trap cleanup EXIT

# =============================================
# Step 4: Build and flash each node
# =============================================

declare -A RESULTS
TOTAL=4
STEP=0
FAIL_COUNT=0
START_TIME=$SECONDS
RX_TARGET_SET=false

# ---- TX Node ----

STEP=$((STEP + 1))
echo
echo -e "${BOLD}[$STEP/$TOTAL] TX Node${NC}"
separator

write_tx_provision
rm -f "$TX_DIR/sdkconfig"

echo -e "  ${DIM}Setting target and building TX firmware...${NC}"
if idf.py -C "$TX_DIR" \
    -DSDKCONFIG_DEFAULTS="$TX_DEFAULTS" \
    set-target esp32s3 2>&1 | tail -3 \
  && idf.py -C "$TX_DIR" build 2>&1 | tail -5; then
    echo -e "  ${GREEN}Build OK${NC}"

    if wait_for_board "TX"; then
        echo -e "  ${DIM}Flashing TX to $FLASH_PORT...${NC}"
        if idf.py -C "$TX_DIR" -p "$FLASH_PORT" flash 2>&1 | tail -5; then
            RESULTS["1-TX"]="OK"
            echo -e "  ${GREEN}TX node flashed successfully${NC}"
        else
            RESULTS["1-TX"]="FLASH_FAIL"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            echo -e "  ${RED}TX node flash FAILED${NC}"
        fi
    else
        RESULTS["1-TX"]="SKIPPED"
        echo -e "  ${YELLOW}TX node skipped${NC}"
    fi
else
    echo -e "  ${RED}TX build FAILED${NC}"
    RESULTS["1-TX"]="BUILD_FAIL"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# ---- RX Nodes ----

for node_id in 1 2 3; do
    STEP=$((STEP + 1))
    label="RX${node_id} (id=${node_id})"

    echo
    echo -e "${BOLD}[$STEP/$TOTAL] $label${NC}"
    separator

    write_rx_provision "$node_id"
    rm -f "$RX_DIR/sdkconfig"

    if [ "$RX_TARGET_SET" = "false" ]; then
        echo -e "  ${DIM}Setting target and building RX firmware (node_id=$node_id, full build)...${NC}"
        if idf.py -C "$RX_DIR" \
            -DSDKCONFIG_DEFAULTS="$RX_DEFAULTS" \
            set-target esp32s3 2>&1 | tail -3 \
          && idf.py -C "$RX_DIR" build 2>&1 | tail -5; then
            echo -e "  ${GREEN}Build OK${NC}"
            BUILD_OK=true
        else
            echo -e "  ${RED}Build FAILED${NC}"
            BUILD_OK=false
        fi
        RX_TARGET_SET=true
    else
        echo -e "  ${DIM}Rebuilding RX firmware with node_id=$node_id (incremental)...${NC}"
        if idf.py -C "$RX_DIR" \
            -DSDKCONFIG_DEFAULTS="$RX_DEFAULTS" \
            build 2>&1 | tail -5; then
            echo -e "  ${GREEN}Build OK${NC}"
            BUILD_OK=true
        else
            echo -e "  ${RED}Build FAILED${NC}"
            BUILD_OK=false
        fi
    fi

    if [ "$BUILD_OK" = "true" ]; then
        if wait_for_board "$label"; then
            echo -e "  ${DIM}Flashing RX node $node_id to $FLASH_PORT...${NC}"
            if idf.py -C "$RX_DIR" -p "$FLASH_PORT" flash 2>&1 | tail -5; then
                RESULTS["$((node_id+1))-RX${node_id}"]="OK"
                echo -e "  ${GREEN}RX node $node_id flashed successfully${NC}"
            else
                RESULTS["$((node_id+1))-RX${node_id}"]="FLASH_FAIL"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                echo -e "  ${RED}RX node $node_id flash FAILED${NC}"
            fi
        else
            RESULTS["$((node_id+1))-RX${node_id}"]="SKIPPED"
            echo -e "  ${YELLOW}RX node $node_id skipped${NC}"
        fi
    else
        RESULTS["$((node_id+1))-RX${node_id}"]="BUILD_FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

# =============================================
# Step 5: Summary
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
    name="${key#*-}"  # strip sort prefix
    if [ "$status" = "OK" ]; then
        echo -e "  ${GREEN}✓${NC} $name"
    elif [ "$status" = "SKIPPED" ]; then
        echo -e "  ${YELLOW}–${NC} $name  (skipped)"
    else
        echo -e "  ${RED}✗${NC} $name  — $status"
    fi
done

echo
if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}All nodes provisioned and flashed successfully!${NC}"
    echo
    echo -e "${DIM}Next steps:${NC}"
    echo -e "${DIM}  Plug in any node and monitor: idf.py -p $FLASH_PORT monitor${NC}"
    echo -e "${DIM}  Read a board's MAC:           esptool.py --port $FLASH_PORT read_mac${NC}"
    exit 0
else
    echo -e "${RED}${FAIL_COUNT} node(s) failed. Review the output above.${NC}"
    exit 1
fi
