#!/usr/bin/env bash
# flash_all.sh — Flash TX and RX firmware to multiple ESP32-S3 boards.
#
# First serial port gets tx-node firmware; remaining ports get rx-node firmware.
#
# Usage:
#   ./tools/flash_all.sh /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyUSB2 /dev/ttyUSB3
#
# Requirements:
#   - ESP-IDF environment (the 'esp' alias or source export.sh)
#   - Firmware already built: firmware/tx-node/build/ and firmware/rx-node/build/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TX_DIR="$REPO_ROOT/firmware/tx-node"
RX_DIR="$REPO_ROOT/firmware/rx-node"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <tx_port> <rx_port1> [rx_port2] [rx_port3] ..."
    echo ""
    echo "  tx_port   — Serial port for the TX node (first argument)"
    echo "  rx_portN  — Serial ports for RX nodes (remaining arguments)"
    echo ""
    echo "Example:"
    echo "  $0 /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyUSB2 /dev/ttyUSB3"
    exit 1
}

if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Need at least 2 serial ports (1 TX + 1 RX).${NC}"
    usage
fi

TX_PORT="$1"
shift
RX_PORTS=("$@")

# Verify serial ports exist
for port in "$TX_PORT" "${RX_PORTS[@]}"; do
    if [ ! -e "$port" ]; then
        echo -e "${RED}Error: Serial port not found: $port${NC}"
        exit 1
    fi
done

# Verify build directories exist
if [ ! -d "$TX_DIR/build" ]; then
    echo -e "${RED}Error: TX firmware not built. Run: cd firmware/tx-node && idf.py build${NC}"
    exit 1
fi

if [ ! -d "$RX_DIR/build" ]; then
    echo -e "${RED}Error: RX firmware not built. Run: cd firmware/rx-node && idf.py build${NC}"
    exit 1
fi

echo "=========================================="
echo " WiFi CSI Firmware Flash Tool"
echo "=========================================="
echo ""
echo -e "  TX node:  ${YELLOW}$TX_PORT${NC}"
for i in "${!RX_PORTS[@]}"; do
    echo -e "  RX node $((i+1)): ${YELLOW}${RX_PORTS[$i]}${NC}"
done
echo ""

# Track results
declare -A RESULTS

# Flash TX node
echo -e "${YELLOW}[1/$((${#RX_PORTS[@]}+1))] Flashing TX node on $TX_PORT...${NC}"
if idf.py -C "$TX_DIR" -p "$TX_PORT" flash 2>&1 | tail -5; then
    RESULTS["TX:$TX_PORT"]="OK"
    echo -e "${GREEN}  TX node flashed successfully.${NC}"
else
    RESULTS["TX:$TX_PORT"]="FAIL"
    echo -e "${RED}  TX node flash FAILED.${NC}"
fi
echo ""

# Flash RX nodes
for i in "${!RX_PORTS[@]}"; do
    port="${RX_PORTS[$i]}"
    node_num=$((i + 1))
    step=$((i + 2))
    total=$((${#RX_PORTS[@]} + 1))

    echo -e "${YELLOW}[$step/$total] Flashing RX node $node_num on $port...${NC}"
    if idf.py -C "$RX_DIR" -p "$port" flash 2>&1 | tail -5; then
        RESULTS["RX$node_num:$port"]="OK"
        echo -e "${GREEN}  RX node $node_num flashed successfully.${NC}"
    else
        RESULTS["RX$node_num:$port"]="FAIL"
        echo -e "${RED}  RX node $node_num flash FAILED.${NC}"
    fi
    echo ""
done

# Print summary
echo "=========================================="
echo " Flash Summary"
echo "=========================================="
FAIL_COUNT=0
for key in $(echo "${!RESULTS[@]}" | tr ' ' '\n' | sort); do
    status="${RESULTS[$key]}"
    if [ "$status" = "OK" ]; then
        echo -e "  $key: ${GREEN}$status${NC}"
    else
        echo -e "  $key: ${RED}$status${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}All devices flashed successfully.${NC}"
    exit 0
else
    echo -e "${RED}$FAIL_COUNT device(s) failed to flash.${NC}"
    exit 1
fi
