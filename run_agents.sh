#!/bin/bash

# End-to-end configurable pipeline (sam4tun stages driven by JSON under configurable/<tunnel_id>/).
# Each step loads its own parameters from:
#   configurable/<tunnel_id>/parameters_unfolding.json
#   configurable/<tunnel_id>/parameters_denoising.json
#   configurable/<tunnel_id>/parameters_enhancing.json
#   configurable/<tunnel_id>/parameters_detecting.json
#   configurable/<tunnel_id>/parameters_sam.json
#
# Usage: ./run_agents.sh <tunnel_id> [--schema 6|7|auto|both] [--memory-ablation] [--save-ablation-memory] ...
# Examples:
#   ./run_agents.sh 1-4
#   ./run_agents.sh 1-4 --schema 6
#   ./run_agents.sh 1-4 --memory-ablation   # writes artefacts to data/ablation/memory/<id>/ (not data/<id>/)
#   ./run_agents.sh 1-4 --save-ablation-memory   # rsync data/<id>/ → data/ablation/memory/<id>/ (after default run)
# Extra words on the line are ignored unless they are a valid schema token or --schema <value>.

if [ $# -eq 0 ]; then
    echo "❌ Error: tunnel_id is required"
    echo "Usage: $0 <tunnel_id> [--schema 6|7|auto|both] [--memory-ablation] [--save-ablation-memory]"
    echo "Example: $0 1-4"
    echo "Example: $0 1-4 --schema 6"
    echo "Example: $0 1-4 --memory-ablation"
    echo "Example: $0 1-4 --save-ablation-memory"
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

TUNNEL_ID=$1
shift || exit 1

EVAL_SCHEMA=6
SAVE_ABLATION_MEMORY=0
MEMORY_ABLATION=0
while [ $# -gt 0 ]; do
    case "$1" in
        --schema)
            shift
            if [ -n "${1:-}" ]; then
                EVAL_SCHEMA="$1"
            fi
            shift || true
            ;;
        --memory-ablation)
            MEMORY_ABLATION=1
            export R4TUN_PIPELINE_OUT_PREFIX=data/ablation/memory
            shift
            ;;
        --save-ablation-memory)
            SAVE_ABLATION_MEMORY=1
            shift
            ;;
        6|7|auto|both)
            EVAL_SCHEMA="$1"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

PIPE_OUT="${R4TUN_PIPELINE_OUT_PREFIX:-data}"
TUNNEL_OUT="${PIPE_OUT}/${TUNNEL_ID}"

# Prefer the project venv interpreter so we never accidentally use system python3 without packages.
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
CONFIG_DIR="configurable/${TUNNEL_ID}"

echo "=========================================="
echo "⏱ Pipeline started: ${PIPELINE_T0_ISO}"
echo "🚀 Configurable pipeline — tunnel: ${TUNNEL_ID}"
echo "📂 Parameters: ${CONFIG_DIR}/parameters_*.json"
echo "🐍 Python: ${PY} ($("${PY}" --version 2>&1))"
echo "📊 Evaluation schema: ${EVAL_SCHEMA}"
if [ "${MEMORY_ABLATION}" = 1 ]; then
    echo "🧠 Memory ablation: artefacts → ${TUNNEL_OUT}/"
fi
if [ "${SAVE_ABLATION_MEMORY}" = 1 ] && [ "${MEMORY_ABLATION}" != 1 ]; then
    echo "📦 After success: rsync data/${TUNNEL_ID}/ → data/ablation/memory/${TUNNEL_ID}/"
fi
echo "=========================================="
echo ""

if ! "${PY}" -c "import numpy" 2>/dev/null; then
    echo "❌ Error: numpy (and likely other deps) missing in this environment."
    echo "   Fix: ${PY} -m pip install -r requirements.txt"
    echo "   (Use the venv’s python: ./venv/bin/python3 -m pip install -r requirements.txt)"
    exit 1
fi

# --- Preconditions (raw cloud: subsets first; optional data/<id>.txt e.g. sample) ---
if [ ! -f "data/subsets/${TUNNEL_ID}.txt" ] && [ ! -f "data/${TUNNEL_ID}.txt" ]; then
    echo "❌ Error: no point cloud for ${TUNNEL_ID}"
    echo "   Use data/subsets/${TUNNEL_ID}.txt (preferred) or data/${TUNNEL_ID}.txt"
    exit 1
fi

if [ ! -d "$CONFIG_DIR" ]; then
    echo "❌ Error: ${CONFIG_DIR}/ not found"
    exit 1
fi

for f in parameters_unfolding.json parameters_denoising.json parameters_enhancing.json parameters_detecting.json parameters_sam.json; do
    if [ ! -f "${CONFIG_DIR}/${f}" ]; then
        echo "❌ Error: missing ${CONFIG_DIR}/${f}"
        exit 1
    fi
done

mkdir -p "${TUNNEL_OUT}/analysis"

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

# --- Stages 1–5: configurable/*.py ---
run_step "Step 1/6: Unfolding" "$PY" configurable/configurable_unfolding.py "$TUNNEL_ID"
run_step "Step 2/6: Denoising" "$PY" configurable/configurable_denoising.py "$TUNNEL_ID"
run_step "Step 3/6: Enhancing" "$PY" configurable/configurable_enhancing.py "$TUNNEL_ID"
run_step "Step 4/6: Detecting" "$PY" configurable/configurable_detecting.py "$TUNNEL_ID"

# Optional: free GPU before SAM (same idea as run_pipeline.sh)
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

run_step "Step 5/6: SAM (configurable_sam)" "$PY" configurable/configurable_sam.py "$TUNNEL_ID"

# --- Stage 6: evaluation ---
if [ -f "${TUNNEL_OUT}/only_label.csv" ]; then
    run_step "Step 6/6: Evaluation (${EVAL_SCHEMA})" \
        "$PY" configurable/evaluation.py "$TUNNEL_ID" --schema "$EVAL_SCHEMA"
else
    echo "⚠️  Skipping evaluation: ${TUNNEL_OUT}/only_label.csv not found"
    echo ""
fi

echo "=========================================="
echo "🎉 Configurable pipeline finished for tunnel: ${TUNNEL_ID}"
echo "=========================================="
echo ""

if [ "${SAVE_ABLATION_MEMORY}" = 1 ] && [ "${MEMORY_ABLATION}" != 1 ]; then
    echo "📦 Copying data/${TUNNEL_ID}/ → data/ablation/memory/${TUNNEL_ID}/"
    mkdir -p "data/ablation/memory/${TUNNEL_ID}"
    rsync -a "data/${TUNNEL_ID}/" "data/ablation/memory/${TUNNEL_ID}/"
    echo "✅ Ablation memory copy complete"
    echo ""
fi

echo "📁 Outputs:"
echo "  - Parameters (read): ${CONFIG_DIR}/parameters_*.json"
echo "  - Artefacts: ${TUNNEL_OUT}/"
if [ -d "${TUNNEL_OUT}/evaluation" ]; then
    echo "  - 6-class metrics: ${TUNNEL_OUT}/evaluation/"
fi
if [ -d "${TUNNEL_OUT}/evaluation_7" ]; then
    echo "  - 7-class metrics: ${TUNNEL_OUT}/evaluation_7/"
fi
echo ""
