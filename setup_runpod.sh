#!/bin/bash
# Start NVIDIA MPS daemon for multi-process parallel training
# Run before batch_train.py with parallel > 1

set -e

echo quit | nvidia-cuda-mps-control 2>/dev/null || true
sleep 1
nvidia-cuda-mps-control -d
echo "NVIDIA MPS daemon started"
