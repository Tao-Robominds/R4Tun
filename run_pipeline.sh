#!/bin/bash

# Pipeline script to run the complete SAM4Tun processing pipeline
# Usage: ./run_pipeline.sh <tunnel_id>
# Example: ./run_pipeline.sh 1-4

set -e  # Exit on error

# Check if tunnel_id is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <tunnel_id>"
    echo "Example: $0 1-4"
    exit 1
fi

TUNNEL_ID=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv directory not found. Please ensure virtual environment is set up."
    exit 1
fi

echo "=========================================="
echo "Starting SAM4Tun Pipeline for tunnel: $TUNNEL_ID"
echo "=========================================="

# Function to check if a command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Step 1: Unfolding
echo ""
echo "Step 1/7: Running unfolding..."
python configurable/configurable_unfolding.py "$TUNNEL_ID"
check_status "Unfolding"

# Step 2: Denoising
echo ""
echo "Step 2/7: Running denoising..."
python configurable/configurable_denoising.py "$TUNNEL_ID"
check_status "Denoising"

# Step 3: Enhancing
echo ""
echo "Step 3/7: Running enhancing..."
python configurable/configurable_enhancing.py "$TUNNEL_ID"
check_status "Enhancing"

# Step 4: Detecting
echo ""
echo "Step 4/7: Running detecting..."
python configurable/configurable_detecting.py "$TUNNEL_ID"
check_status "Detecting"

# Step 5: Kill GPU processes before SAM
echo ""
echo "Step 5/7: Killing GPU processes before SAM..."
# Find Python processes using GPU
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -v "^$" || true)

if [ -n "$GPU_PIDS" ]; then
    echo "Found GPU processes: $GPU_PIDS"
    for pid in $GPU_PIDS; do
        # Check if it's a Python process and not this script
        if ps -p "$pid" -o comm= 2>/dev/null | grep -q python; then
            echo "Killing Python process $pid using GPU..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    echo "GPU processes killed"
else
    echo "No GPU processes found"
fi

# Clear GPU memory
echo "Clearing GPU memory cache..."
python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Step 6: SAM Segmentation
echo ""
echo "Step 6/7: Running SAM segmentation..."
python configurable/configurable_sam.py "$TUNNEL_ID"
check_status "SAM segmentation"

# Step 7: Evaluation
echo ""
echo "Step 7/7: Running evaluation..."
python configurable/evaluation.py "$TUNNEL_ID"
check_status "Evaluation"

echo ""
echo "=========================================="
echo "✅ Pipeline completed successfully for tunnel: $TUNNEL_ID"
echo "=========================================="

