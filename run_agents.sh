#!/bin/bash

# End-to-end configurable pipeline with ablation support.
# Parameters are loaded directly from configurable/ablation/{condition}/ — no sync step.
#
# Usage:
#   ./run_agents.sh <tunnel_id> --ablation <code> [--model <tag>] [--schema 6|7|auto|both]
#   ./run_agents.sh --all --ablation <code> [--model <tag>] [--schema 6|7|auto|both]
#
# Ablation codes: sam4tun, m, m_s, m_s_k, r
# Model tag: LLM model suffix for parameter filenames (default: opus4.6)
#
# Examples:
#   ./run_agents.sh 1-1 --ablation m --schema auto
#   ./run_agents.sh 1-1 --ablation m --model gemini2.5 --schema auto
#   ./run_agents.sh --all --ablation m --schema auto

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "❌ Error: arguments required"
    echo "Usage: $0 <tunnel_id|--all> --ablation <code> [--schema 6|7|auto|both]"
    echo "Ablation codes: sam4tun, m, m_s, m_s_k, r"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PIPELINE_T0=$(date +%s)
PIPELINE_T0_ISO=$(date -Iseconds)

pipeline_print_runtime() {
    local t1 sec m s
    t1=$(date +%s)
    sec=$((t1 - PIPELINE_T0))
    m=$((sec / 60))
    s=$((sec % 60))
    echo ""
    echo "=========================================="
    echo "⏱ Pipeline wall clock: ${sec}s (${m}m ${s}s)"
    echo "   Started: ${PIPELINE_T0_ISO}"
    echo "   Finished: $(date -Iseconds)"
    echo "=========================================="
}
trap pipeline_print_runtime EXIT

# --- Parse arguments ---
TUNNEL_ID=""
ABLATION=""
MODEL="opus4.6"
EVAL_SCHEMA="auto"
RUN_ALL=0

while [ $# -gt 0 ]; do
    case "$1" in
        --all)
            RUN_ALL=1
            shift
            ;;
        --ablation|-a)
            shift
            ABLATION="${1:-}"
            shift || true
            ;;
        --model)
            shift
            MODEL="${1:-opus4.6}"
            shift || true
            ;;
        --schema)
            shift
            EVAL_SCHEMA="${1:-auto}"
            shift || true
            ;;
        6|7|auto|both)
            EVAL_SCHEMA="$1"
            shift
            ;;
        -*)
            echo "❌ Unknown flag: $1"
            exit 1
            ;;
        *)
            if [ -z "$TUNNEL_ID" ]; then
                TUNNEL_ID="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$ABLATION" ]; then
    echo "❌ Error: --ablation <code> is required"
    echo "Ablation codes: sam4tun, m, m_s, m_s_k, r"
    exit 1
fi

# Validate ablation code
case "$ABLATION" in
    sam4tun|m|m_s|m_s_k|r) ;;
    *)
        echo "❌ Unknown ablation code: $ABLATION"
        echo "Valid codes: sam4tun, m, m_s, m_s_k, r"
        exit 1
        ;;
esac

# Map ablation code → output prefix
declare -A ABLATION_PREFIX=(
    [sam4tun]="data/ablation/sam4tun"
    [m]="data/ablation/memory"
    [m_s]="data/ablation/memory+state"
    [m_s_k]="data/ablation/memory+state+knowledge"
    [r]="data/ablation/reflection"
)
export R4TUN_PIPELINE_OUT_PREFIX="${ABLATION_PREFIX[$ABLATION]}"

# --- Resolve tunnel list ---
if [ "$RUN_ALL" = 1 ]; then
    if [ "$ABLATION" = "sam4tun" ]; then
        TUNNEL_IDS=$(ls data/subsets/*.txt 2>/dev/null | xargs -I{} basename {} .txt | sort)
    else
        declare -A ABLATION_FOLDER=(
            [m]="memory"
            [m_s]="memory+state"
            [m_s_k]="memory+state+knowledge"
            [r]="reflection"
        )
        PARAM_DIR="configurable/ablation/${ABLATION_FOLDER[$ABLATION]}/parameters"
        if [ ! -d "$PARAM_DIR" ]; then
            echo "❌ No parameters directory: $PARAM_DIR"
            exit 1
        fi
        TUNNEL_IDS=$(ls "$PARAM_DIR" 2>/dev/null | sort)
    fi
    if [ -z "$TUNNEL_IDS" ]; then
        echo "❌ No tunnels found for --all with ablation=$ABLATION"
        exit 1
    fi
    echo "🔍 Discovered tunnels: $(echo $TUNNEL_IDS | tr '\n' ' ')"
elif [ -n "$TUNNEL_ID" ]; then
    TUNNEL_IDS="$TUNNEL_ID"
else
    echo "❌ Error: provide <tunnel_id> or --all"
    exit 1
fi

# --- Python interpreter ---
if [ -n "${PYTHON:-}" ] && [ -x "${PYTHON}" ]; then
    PY="${PYTHON}"
elif [ -x "${SCRIPT_DIR}/venv/bin/python3" ]; then
    PY="${SCRIPT_DIR}/venv/bin/python3"
elif [ -x "${SCRIPT_DIR}/venv/bin/python" ]; then
    PY="${SCRIPT_DIR}/venv/bin/python"
else
    PY="python3"
fi

if [ -d "venv" ]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
fi

if ! "${PY}" -c "import numpy" 2>/dev/null; then
    echo "❌ Error: numpy (and likely other deps) missing."
    echo "   Fix: ${PY} -m pip install -r requirements.txt"
    exit 1
fi

run_step() {
    local name=$1
    shift
    echo "=========================================="
    echo "🔄 ${name}"
    echo "=========================================="
    if "$@"; then
        echo "✅ ${name} complete"
    else
        echo "❌ ${name} failed"
        exit 1
    fi
    echo ""
}

# --- Preflight: verify all parameter JSONs exist for every tunnel ---
preflight_check() {
    local tid=$1
    local code=$2
    local mdl=$3
    "$PY" -c "
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath('.')), 'configurable'))
sys.path.insert(0, 'configurable')
from pipeline_data import resolve_ablation_param_file
stages = ['unfolding', 'denoising', 'enhancing', 'detecting', 'sam']
missing = []
for s in stages:
    p = resolve_ablation_param_file('${tid}', s, '${code}', '${mdl}')
    if not os.path.isfile(p):
        missing.append(p)
if missing:
    for m in missing:
        print(m)
    sys.exit(1)
" 2>&1
}

echo ""
echo "=========================================="
echo "🔎 Preflight: checking parameter files..."
echo "=========================================="
PREFLIGHT_OK=1
for TID in $TUNNEL_IDS; do
    MISSING=$(preflight_check "$TID" "$ABLATION" "$MODEL" 2>&1) || {
        echo "❌ Missing parameter files for tunnel ${TID}:"
        echo "$MISSING" | while read -r line; do echo "   $line"; done
        PREFLIGHT_OK=0
    }
done
if [ "$PREFLIGHT_OK" = 0 ]; then
    echo ""
    echo "❌ Preflight failed — fix missing parameter files before running."
    exit 1
fi
echo "✅ All parameter files verified."
echo ""

# --- Run pipeline for each tunnel ---
for TID in $TUNNEL_IDS; do
    TUNNEL_OUT="${R4TUN_PIPELINE_OUT_PREFIX}/${TID}"

    echo ""
    echo "=========================================="
    echo "🚀 Pipeline — tunnel: ${TID}, ablation: ${ABLATION}, model: ${MODEL}"
    echo "📂 Parameters: configurable/ablation/.../${TID}/"
    echo "📊 Evaluation schema: ${EVAL_SCHEMA}"
    echo "📁 Output: ${TUNNEL_OUT}/"
    echo "=========================================="
    echo ""

    # Precondition: raw point cloud exists
    if [ ! -f "data/subsets/${TID}.txt" ] && [ ! -f "data/${TID}.txt" ]; then
        echo "❌ Error: no point cloud for ${TID}"
        exit 1
    fi

    mkdir -p "${TUNNEL_OUT}"

    run_step "Step 1/6: Unfolding (${TID})"  "$PY" configurable/configurable_unfolding.py "$TID" --ablation "$ABLATION" --model "$MODEL"
    run_step "Step 2/6: Denoising (${TID})"  "$PY" configurable/configurable_denoising.py "$TID" --ablation "$ABLATION" --model "$MODEL"
    run_step "Step 3/6: Enhancing (${TID})"  "$PY" configurable/configurable_enhancing.py "$TID" --ablation "$ABLATION" --model "$MODEL"
    run_step "Step 4/6: Detecting (${TID})"  "$PY" configurable/configurable_detecting.py "$TID" --ablation "$ABLATION" --model "$MODEL"

    # GPU cleanup before SAM
    echo "=========================================="
    echo "🖥️  GPU cleanup before SAM (best effort)"
    echo "=========================================="
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -v "^$" || true)
    if [ -n "$GPU_PIDS" ]; then
        for pid in $GPU_PIDS; do
            if ps -p "$pid" -o comm= 2>/dev/null | grep -qi python; then
                echo "Killing Python GPU pid $pid ..."
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
        sleep 2
    fi
    "$PY" -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    echo ""

    run_step "Step 5/6: SAM (${TID})" "$PY" configurable/configurable_sam.py "$TID" --ablation "$ABLATION" --model "$MODEL"

    if [ -f "${TUNNEL_OUT}/only_label.csv" ]; then
        run_step "Step 6/6: Evaluation (${TID})" \
            "$PY" configurable/evaluation.py "$TID" --ablation "$ABLATION" --schema "$EVAL_SCHEMA"
    else
        echo "⚠️  Skipping evaluation: ${TUNNEL_OUT}/only_label.csv not found"
        echo ""
    fi

    echo "=========================================="
    echo "🎉 Pipeline finished for tunnel: ${TID} (ablation: ${ABLATION})"
    echo "=========================================="
    echo ""
done
