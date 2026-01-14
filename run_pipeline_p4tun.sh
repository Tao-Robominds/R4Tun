#!/bin/bash
#
# GT-Free Tunnel Segmentation Pipeline (P4Tun Version)
#
# Usage:
#   ./run_pipeline_p4tun.sh <tunnel_id>              # Run full pipeline
#   ./run_pipeline_p4tun.sh <tunnel_id> --from <stage>  # Start from specific stage
#   ./run_pipeline_p4tun.sh --all                    # Run on all datasets
#   ./run_pipeline_p4tun.sh --list                   # List available datasets
#
# Stages:
#   1 = unfolding
#   2 = denoising
#   3 = enhancing
#   4 = detection
#   5 = sam
#   6 = evaluation
#
# Examples:
#   ./run_pipeline_p4tun.sh 1-4                  # Full pipeline on 1-4
#   ./run_pipeline_p4tun.sh 1-4 --from 4         # Start from detection
#   ./run_pipeline_p4tun.sh 4-1 --base patterns  # Use data/patterns/4-1
#   ./run_pipeline_p4tun.sh --all                # All datasets
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BASE_DIR="data"
START_STAGE=1
END_STAGE=6
SKIP_SAM=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Print banner
print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║       GT-FREE TUNNEL SEGMENTATION PIPELINE (P4TUN VERSION)       ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Print usage
usage() {
    echo "Usage: $0 <tunnel_id> [options]"
    echo ""
    echo "Options:"
    echo "  --from <stage>    Start from stage (1-6)"
    echo "  --to <stage>      End at stage (1-6)"
    echo "  --base <dir>      Base directory (default: data)"
    echo "                    Use 'patterns' for data/patterns/"
    echo "                    Use 'configurable' for data/configurable/"
    echo "  --skip-sam        Skip SAM segmentation (stage 5)"
    echo "  --all             Run on all available datasets"
    echo "  --list            List available datasets"
    echo "  -h, --help        Show this help"
    echo ""
    echo "Stages:"
    echo "  1 = unfolding     Convert raw point cloud to cylindrical coords"
    echo "  2 = denoising     Remove noise from point cloud"
    echo "  3 = enhancing     Enhance point cloud and generate depth map"
    echo "  4 = detection     Detect lines and infer segment positions"
    echo "  5 = sam           SAM-based segmentation"
    echo "  6 = evaluation    Evaluate results"
    echo ""
    echo "Examples:"
    echo "  $0 1-4                      # Full pipeline on data/1-4"
    echo "  $0 1-4 --from 4             # Start from detection"
    echo "  $0 4-1 --base patterns      # Use data/patterns/4-1"
    echo "  $0 --all --skip-sam         # All datasets, skip SAM"
}

# List available datasets
list_datasets() {
    echo -e "${YELLOW}Available datasets:${NC}"
    echo ""
    
    echo -e "${GREEN}data/${NC}"
    for d in data/*/; do
        if [ -f "${d}ring_count.txt" ] || [ -f "${d}unwrapped.csv" ]; then
            name=$(basename "$d")
            echo "  $name"
        fi
    done
    
    echo ""
    echo -e "${GREEN}data/configurable/${NC}"
    for d in data/configurable/*/; do
        if [ -d "$d" ]; then
            name=$(basename "$d")
            echo "  $name  (use --base configurable)"
        fi
    done
    
    echo ""
    echo -e "${GREEN}data/patterns/${NC}"
    for d in data/patterns/*/; do
        if [ -d "$d" ]; then
            name=$(basename "$d")
            echo "  $name  (use --base patterns)"
        fi
    done
}

# Run a single stage
# Note: Scripts expect tunnel_id as relative path from data/ (e.g., "1-4" or "patterns/4-1")
run_stage() {
    local stage=$1
    local tunnel_id=$2
    local tunnel_dir=$3
    
    case $stage in
        1)
            echo -e "${YELLOW}[Stage 1/6] Unfolding...${NC}"
            python3 p4tun/1_unfolding.py "$tunnel_id"
            ;;
        2)
            echo -e "${YELLOW}[Stage 2/6] Denoising...${NC}"
            python3 p4tun/2_denoising.py "$tunnel_id"
            ;;
        3)
            echo -e "${YELLOW}[Stage 3/6] Enhancing...${NC}"
            python3 p4tun/3_enhancing.py "$tunnel_id"
            ;;
        4)
            echo -e "${YELLOW}[Stage 4/6] Detection & Pattern Inference...${NC}"
            python3 p4tun/4-1_detection.py "$tunnel_id"
            ;;
        5)
            if [ "$SKIP_SAM" = true ]; then
                echo -e "${YELLOW}[Stage 5/6] SAM Segmentation... SKIPPED${NC}"
            else
                echo -e "${YELLOW}[Stage 5/6] SAM Segmentation...${NC}"
                python3 p4tun/4-2_sam.py "$tunnel_id"
            fi
            ;;
        6)
            echo -e "${YELLOW}[Stage 6/6] Evaluation...${NC}"
            python3 p4tun/evaluation.py "$tunnel_id"
            ;;
    esac
}

# Run pipeline on a single dataset
run_pipeline() {
    local tunnel_name=$1
    local tunnel_dir="${BASE_DIR}/${tunnel_name}"
    
    # Construct tunnel_id for scripts (relative path from data/)
    # e.g., if BASE_DIR="data/patterns" and tunnel_name="4-1", tunnel_id="patterns/4-1"
    local tunnel_id
    if [ "$BASE_DIR" = "data" ]; then
        tunnel_id="$tunnel_name"
    else
        # Extract the subdirectory from BASE_DIR (e.g., "patterns" from "data/patterns")
        local subdir="${BASE_DIR#data/}"
        tunnel_id="${subdir}/${tunnel_name}"
    fi
    
    echo -e "${GREEN}Processing: ${tunnel_name}${NC}"
    echo -e "${BLUE}Directory: ${tunnel_dir}${NC}"
    echo -e "${BLUE}Tunnel ID: ${tunnel_id}${NC}"
    echo ""
    
    # Create directory if it doesn't exist
    if [ ! -d "$tunnel_dir" ]; then
        echo -e "${YELLOW}Creating directory: ${tunnel_dir}${NC}"
        mkdir -p "$tunnel_dir"
    fi
    
    # Check if input file exists (for stage 1, need the raw point cloud)
    if [ "$START_STAGE" -eq 1 ]; then
        local input_file="data/${tunnel_name}.txt"
        if [ ! -f "$input_file" ]; then
            echo -e "${RED}Error: Input file not found: ${input_file}${NC}"
            echo -e "${RED}Stage 1 (unfolding) requires the raw point cloud file.${NC}"
            return 1
        fi
    fi
    
    # Run stages
    for stage in $(seq $START_STAGE $END_STAGE); do
        run_stage $stage "$tunnel_id" "$tunnel_dir"
        echo ""
    done
    
    echo -e "${GREEN}✓ Pipeline complete for ${tunnel_name}${NC}"
}

# Run on all datasets
run_all() {
    local datasets=()
    
    # Collect datasets from data/
    for d in data/*/; do
        if [ -f "${d}ring_count.txt" ] || [ -f "${d}unwrapped.csv" ]; then
            name=$(basename "$d")
            datasets+=("$name")
        fi
    done
    
    echo -e "${YELLOW}Running pipeline on ${#datasets[@]} datasets...${NC}"
    echo ""
    
    for tunnel_id in "${datasets[@]}"; do
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        run_pipeline "$tunnel_id"
        echo ""
    done
    
    echo -e "${GREEN}✓ All datasets processed${NC}"
}

# Parse arguments
TUNNEL_ID=""
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --from)
            START_STAGE="$2"
            shift 2
            ;;
        --to)
            END_STAGE="$2"
            shift 2
            ;;
        --base)
            case $2 in
                patterns)
                    BASE_DIR="data/patterns"
                    ;;
                configurable)
                    BASE_DIR="data/configurable"
                    ;;
                *)
                    BASE_DIR="$2"
                    ;;
            esac
            shift 2
            ;;
        --skip-sam)
            SKIP_SAM=true
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --list)
            list_datasets
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [ -z "$TUNNEL_ID" ]; then
                TUNNEL_ID="$1"
            fi
            shift
            ;;
    esac
done

# Main execution
print_banner

if [ "$RUN_ALL" = true ]; then
    run_all
elif [ -n "$TUNNEL_ID" ]; then
    run_pipeline "$TUNNEL_ID"
else
    usage
    exit 1
fi

