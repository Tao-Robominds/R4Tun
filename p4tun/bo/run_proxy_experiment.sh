#!/bin/bash
# Proxy BO Experiment - validate intrinsic-metrics predictor for tuning
#
# Compares: Baseline | Proxy BO (predicted mIoU) | True BO (oracle)
#
# Usage:
#   ./run_proxy_experiment.sh                    # Full (20 calls, ~1 hour)
#   ./run_proxy_experiment.sh --n-calls 6        # Quick test (~15 min)
#
# Tunnels: 1-4 works. 2-2, 3-1 may have param/config issues.
set -e
cd "$(dirname "$0")/../.."
exec ./venv/bin/python -m p4tun.bo.proxy_bo_experiment "$@"
