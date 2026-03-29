#!/usr/bin/env bash
# Run original sam4tun/*.py (not configurable) as true baseline.
# Input: data/subsets/{tunnel_id}.txt via symlink data/{tunnel_id}.txt
# Output: moved to data/ablation/sam4tun/{tunnel_id}/
#
# Usage:
#   ./run_sam4tun_baseline.sh              # all tunnels from data/subsets
#   ./run_sam4tun_baseline.sh 1-1          # single tunnel

set -euo pipefail

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
    echo "Sam4tun baseline wall clock: ${sec}s (${m}m ${s}s)"
    echo "   Started: ${PIPELINE_T0_ISO}"
    echo "   Finished: $(date -Iseconds)"
    echo "=========================================="
}
trap pipeline_print_runtime EXIT

if [ -n "${PYTHON:-}" ] && [ -x "${PYTHON}" ]; then
    PY="${PYTHON}"
elif [ -x "${SCRIPT_DIR}/venv/bin/python3" ]; then
    PY="${SCRIPT_DIR}/venv/bin/python3"
else
    PY="python3"
fi

if [ -d "venv" ]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
fi

# segment_anything lives under sam4tun/segment-anything/
export PYTHONPATH="${SCRIPT_DIR}/sam4tun/segment-anything${PYTHONPATH:+:${PYTHONPATH}}"

OUT_ROOT="data/ablation/sam4tun"
mkdir -p "$OUT_ROOT"
mkdir -p logs

run_step() {
    local name=$1
    shift
    echo "=========================================="
    echo "  ${name}"
    echo "=========================================="
    if "$@"; then
        echo "  OK ${name}"
    else
        echo "  FAIL ${name}"
        exit 1
    fi
    echo ""
}

gpu_cleanup_before_sam() {
    echo "=========================================="
    echo "  GPU cleanup before SAM (best effort)"
    echo "=========================================="
    local GPU_PIDS
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -v "^$" || true)
    if [ -n "$GPU_PIDS" ]; then
        for pid in $GPU_PIDS; do
            if ps -p "$pid" -o comm= 2>/dev/null | grep -qi python; then
                echo "  Killing Python GPU pid $pid ..."
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
        sleep 2
    fi
    "$PY" -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    echo ""
}

# 4-x / 5-x: 7-class SAM + eval; else 6-class
is_complex_family() {
    case "$1" in
        4-*|5-*) return 0 ;;
        *) return 1 ;;
    esac
}

extract_miou_from_perf_md() {
    local f=$1
    if [ -f "$f" ]; then
        grep -m1 'Mean IoU (mIoU):' "$f" 2>/dev/null | sed -E 's/.*: *//' | tr -d '\r' || true
    fi
}

# --- Tunnel list ---
if [ $# -ge 1 ] && [ -n "${1:-}" ] && [ "$1" != "--all" ]; then
    TUNNEL_IDS="$1"
else
    TUNNEL_IDS=$(ls data/subsets/*.txt 2>/dev/null | xargs -I{} basename {} .txt | sort)
fi

if [ -z "${TUNNEL_IDS}" ]; then
    echo "No tunnels found (data/subsets/*.txt)."
    exit 1
fi

TUNNEL_TOTAL=0
for _ in $TUNNEL_IDS; do TUNNEL_TOTAL=$((TUNNEL_TOTAL + 1)); done

echo ""
echo "=========================================="
echo "Sam4tun baseline (original scripts)"
echo "Tunnels: ${TUNNEL_TOTAL}"
echo "Output:  ${OUT_ROOT}/<tunnel_id>/"
echo "Python:  ${PY}"
echo "=========================================="
echo ""

TUNNEL_CURRENT=0
for TID in $TUNNEL_IDS; do
    TUNNEL_CURRENT=$((TUNNEL_CURRENT + 1))
    TUNNEL_T0=$(date +%s)

    SUBSET="data/subsets/${TID}.txt"
    LINK="data/${TID}.txt"
    WORK="data/${TID}"

    if [ ! -f "$SUBSET" ]; then
        echo "Missing subset: $SUBSET"
        exit 1
    fi

    if is_complex_family "$TID"; then
        SAM_EVAL_LABEL="7-class (4-2_sam_4+5 + evaluation_4+5)"
        SAM_SCRIPT="sam4tun/4-2_sam_4+5.py"
        EVAL_SCRIPT="sam4tun/evaluation_4+5.py"
    else
        SAM_EVAL_LABEL="6-class (4-2_sam + evaluation)"
        SAM_SCRIPT="sam4tun/4-2_sam.py"
        EVAL_SCRIPT="sam4tun/evaluation.py"
    fi

    echo ""
    echo "=========================================="
    echo "[${TUNNEL_CURRENT}/${TUNNEL_TOTAL}] tunnel=${TID}  ${SAM_EVAL_LABEL}"
    echo "=========================================="

    rm -rf "$WORK"
    rm -f "$LINK"
    ln -sfn "subsets/${TID}.txt" "$LINK"

    run_step "1/6 Unfolding (${TID})" "$PY" sam4tun/1_upfolding.py "$TID"
    run_step "2/6 Denoising (${TID})" "$PY" sam4tun/2_denoising.py "$TID"
    run_step "3/6 Enhancing (${TID})" "$PY" sam4tun/3_enhancing.py "$TID"
    run_step "4/6 Detection (${TID})" "$PY" sam4tun/4-1_detection.py "$TID"

    gpu_cleanup_before_sam
    run_step "5/6 SAM (${TID})" "$PY" "$SAM_SCRIPT" "$TID"
    run_step "6/6 Evaluation (${TID})" "$PY" "$EVAL_SCRIPT" "$TID"

    if [ ! -d "$WORK" ]; then
        echo "Expected output dir missing: $WORK"
        rm -f "$LINK"
        exit 1
    fi

    rm -rf "${OUT_ROOT}/${TID}"
    mkdir -p "$OUT_ROOT"
    mv "$WORK" "${OUT_ROOT}/${TID}"
    rm -f "$LINK"

    TUNNEL_T1=$(date +%s)
    TUNNEL_SEC=$((TUNNEL_T1 - TUNNEL_T0))
    PERF="${OUT_ROOT}/${TID}/evaluation/performance.md"
    MIOU_VAL=$(extract_miou_from_perf_md "$PERF")
    [ -z "$MIOU_VAL" ] && MIOU_VAL="n/a"

    echo "=========================================="
    echo "[${TUNNEL_CURRENT}/${TUNNEL_TOTAL}] done ${TID} — ${TUNNEL_SEC}s — mIoU: ${MIOU_VAL}"
    echo "=========================================="
    echo ""
done

echo "All ${TUNNEL_TOTAL} tunnels finished."
