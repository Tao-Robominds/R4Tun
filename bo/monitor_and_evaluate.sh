#!/bin/bash
# Monitor the build and run evaluation when complete

cd "$(dirname "$0")/.."

echo "Monitoring build progress..."
echo "Press Ctrl+C to stop monitoring (build continues in background)"
echo ""

while true; do
    # Check if build is still running
    if ! pgrep -f "build_training_data" > /dev/null; then
        echo ""
        echo "Build completed!"
        
        # Check if output was created
        if [ -f "bo4tun/training/intrinsic_training_data.csv" ]; then
            echo "Dataset created. Running evaluation..."
            ./venv/bin/python -m p4tun.bo.evaluate_predictor
        else
            echo "Error: No output file created"
            cat bo4tun/training/build_det_variation.log
        fi
        exit 0
    fi
    
    # Show progress
    SAM_PROC=$(ps aux | grep "4-2_sam.py" | grep -v grep | awk '{print $NF}')
    NOW=$(date +%H:%M:%S)
    CSV_SIZE=$(wc -l bo4tun/training/intrinsic_training_data.csv 2>/dev/null | awk '{print $1}')
    echo -ne "\r[$NOW] Building... SAM on: $SAM_PROC | Output rows: ${CSV_SIZE:-0}      "
    
    sleep 10
done
