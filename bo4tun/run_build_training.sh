#!/bin/bash
# Build intrinsic metrics training data
# Usage:
#   ./run_build_training.sh              # Full dataset (~504 configs, ~17 hours)
#   ./run_build_training.sh --sample 50  # Sample of 50 (~100 min)
#   ./run_build_training.sh --detection-only --sample 100  # Fast validation (~5 min)

set -e
cd "$(dirname "$0")/.."
exec ./venv/bin/python -m bo4tun.build_training_data "$@"
