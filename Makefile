SHELL := /bin/bash
.DEFAULT_GOAL := help

REPO_ROOT := $(shell pwd)
EDGE_DIR := $(REPO_ROOT)/edge
GPU_DIR := $(REPO_ROOT)/gpu
FW_TX := $(REPO_ROOT)/firmware/tx-node
FW_RX := $(REPO_ROOT)/firmware/rx-node
VENV := $(REPO_ROOT)/.venv

# Serial ports (override on command line: make flash-tx PORT=/dev/ttyUSB0)
PORT ?= /dev/ttyUSB0

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Firmware
# ──────────────────────────────────────────────

.PHONY: build-tx
build-tx: ## Build TX node firmware
	cd $(FW_TX) && idf.py set-target esp32s3 && idf.py build

.PHONY: build-rx
build-rx: ## Build RX node firmware
	cd $(FW_RX) && idf.py set-target esp32s3 && idf.py build

.PHONY: flash-tx
flash-tx: ## Flash TX node (PORT=/dev/ttyUSB0)
	cd $(FW_TX) && idf.py -p $(PORT) flash monitor

.PHONY: flash-rx
flash-rx: ## Flash RX node (PORT=/dev/ttyUSB0)
	cd $(FW_RX) && idf.py -p $(PORT) flash monitor

# ──────────────────────────────────────────────
# Raspberry Pi Edge
# ──────────────────────────────────────────────

.PHONY: pi-setup
pi-setup: ## Set up Pi environment (venv + deps)
	cd $(EDGE_DIR) && bash setup.sh

.PHONY: pi-start
pi-start: ## Start the CSI aggregator service
	sudo systemctl start csi-aggregator

.PHONY: pi-stop
pi-stop: ## Stop the CSI aggregator service
	sudo systemctl stop csi-aggregator

.PHONY: pi-restart
pi-restart: ## Restart the CSI aggregator service
	sudo systemctl restart csi-aggregator

.PHONY: pi-logs
pi-logs: ## Follow CSI aggregator logs
	sudo journalctl -u csi-aggregator -f

# ──────────────────────────────────────────────
# GPU Server
# ──────────────────────────────────────────────

.PHONY: gpu-train
gpu-train: ## Start GPU training container
	cd $(GPU_DIR) && docker compose --profile training up

.PHONY: gpu-infer
gpu-infer: ## Start GPU inference server (daemon)
	cd $(GPU_DIR) && docker compose up -d

.PHONY: gpu-logs
gpu-logs: ## Follow GPU inference logs
	cd $(GPU_DIR) && docker compose logs -f

.PHONY: gpu-down
gpu-down: ## Stop GPU containers
	cd $(GPU_DIR) && docker compose down

# ──────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────

.PHONY: test
test: test-proto test-edge test-gpu ## Run all tests

.PHONY: test-proto
test-proto: ## Run protocol compatibility tests
	cd $(REPO_ROOT)/proto && $(VENV)/bin/python -m pytest test_protocol_compat.py -v

.PHONY: test-edge
test-edge: ## Run edge pipeline tests
	PYTHONPATH=$(REPO_ROOT) $(VENV)/bin/python -m pytest $(EDGE_DIR)/tests/ -v

.PHONY: test-gpu
test-gpu: ## Run GPU model tests
	PYTHONPATH=$(REPO_ROOT) $(VENV)/bin/python -m pytest $(GPU_DIR)/tests/ -v

# ──────────────────────────────────────────────
# Developer Tools
# ──────────────────────────────────────────────

.PHONY: collect
collect: ## Run labeled data collection session
	$(VENV)/bin/python $(REPO_ROOT)/tools/collect_data.py

.PHONY: visualize
visualize: ## Launch real-time CSI visualizer
	$(VENV)/bin/python $(REPO_ROOT)/tools/visualize_csi.py
