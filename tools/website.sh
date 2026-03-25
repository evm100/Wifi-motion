#!/bin/bash

source .venv/bin/activate
python -m edge.web --port 8080 --udp-port 5005
