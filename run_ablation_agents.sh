#!/usr/bin/env bash
# GLM ablation pipeline-only runner: sam4tun/agents with ablation params → data/ablation/{condition}/{tid}/
#
# Does NOT call the LLM — run run_ablation_glm.py (or run_memory_*_glm.py) first to write params.
#
# Usage:
#   ./run_ablation_agents.sh --ablation m_s_k --model glm --t1-t2
#   ./run_ablation_agents.sh --ablation m_s_k --model glm --sanity   # requires T1/T2 gate PASS
#   ./run_ablation_agents.sh --ablation m --model glm 1-1 2-1
#   ./run_ablation_agents.sh --ablation m_s_k --model glm --sanity --skip-existing

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
    echo "Ablation agents wall clock: ${sec}s (${m}m ${s}s)"
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

export PYTHONPATH="${SCRIPT_DIR}/sam4tun/segment-anything${PYTHONPATH:+:${PYTHONPATH}}"

SAM4TUN_DATA="sam4tun/data"
AGENTS_DIR="sam4tun/agents"
SANITY_IDS="1-1 2-1 3-1-1 4-1 5-1"
T1_T2_IDS="1-1 2-1"
T1_T2_GATE_FILE="data/ablation/t1_t2_gate.md"

ABLATION=""
MODEL_TAG="glm"
TUNNEL_IDS=""
RUN_SANITY=0
RUN_T1_T2=0
SKIP_EXISTING=0

while [ $# -gt 0 ]; do
    case "$1" in
        --ablation)
            ABLATION="${2:?--ablation requires m|m_s|m_s_k}"
            shift 2
            ;;
        --model)
            MODEL_TAG="${2:?--model requires tag e.g. glm}"
            shift 2
            ;;
        --sanity)
            RUN_SANITY=1
            shift
            ;;
        --t1-t2)
            RUN_T1_T2=1
            shift
            ;;
        --skip-existing)
            SKIP_EXISTING=1
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            TUNNEL_IDS="${TUNNEL_IDS} $1"
            shift
            ;;
    esac
done

case "$ABLATION" in
    m)    ABLATION_FOLDER="memory" ;;
    m_s)  ABLATION_FOLDER="memory+state" ;;
    m_s_k) ABLATION_FOLDER="memory+state+knowledge" ;;
    "")
        echo "ERROR: --ablation m|m_s|m_s_k is required"
        exit 1
        ;;
    *)
        echo "ERROR: unknown --ablation ${ABLATION}"
        exit 1
        ;;
esac

OUT_ROOT="data/ablation/${ABLATION_FOLDER}"
LOG_DIR="logs/ablation"
PARAM_ROOT="sam4tun/agents/parameters/${ABLATION_FOLDER}"
mkdir -p "$OUT_ROOT" "$LOG_DIR"

t1_t2_gate_passed() {
    [ -f "$T1_T2_GATE_FILE" ] && grep -q "Status: PASS" "$T1_T2_GATE_FILE"
}

if [ "$RUN_SANITY" -eq 1 ] && ! t1_t2_gate_passed; then
    echo "ERROR: --sanity requires T1/T2 gate PASS in ${T1_T2_GATE_FILE}"
    echo "Run: venv/bin/python3 run_ablation_glm.py --ablation m_s_k --t1-t2"
    exit 1
fi

if [ "$RUN_T1_T2" -eq 1 ]; then
    TUNNEL_IDS="$T1_T2_IDS"
elif [ "$RUN_SANITY" -eq 1 ]; then
    TUNNEL_IDS="$SANITY_IDS"
fi

TUNNEL_IDS=$(echo "$TUNNEL_IDS" | xargs)
if [ -z "$TUNNEL_IDS" ]; then
    echo "Provide tunnel IDs, --t1-t2, or --sanity"
    exit 1
fi

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

extract_miou_from_perf_md() {
    local f=$1
    if [ -f "$f" ]; then
        grep -m1 'Mean IoU (mIoU):' "$f" 2>/dev/null | sed -E 's/.*: *//' | tr -d '\r' || true
    fi
}

param_file_for_stage() {
    local TID=$1
    local STAGE=$2
    local SUFFIX
    case "$ABLATION" in
        m)    SUFFIX="_m_" ;;
        m_s)  SUFFIX="_m_s_" ;;
        m_s_k) SUFFIX="_m_s_k_" ;;
    esac
    echo "${PARAM_ROOT}/${TID}/parameters_${STAGE}${SUFFIX}${MODEL_TAG}.json"
}

run_tunnel_pipeline() {
    local TID=$1
    local SUBSET="data/subsets/${TID}.txt"
    local LINK="${SAM4TUN_DATA}/${TID}.txt"
    local WORK="${SAM4TUN_DATA}/${TID}"
    local DEST="${OUT_ROOT}/${TID}"
    local LOG="${LOG_DIR}/${TID}_${ABLATION}_${MODEL_TAG}_agents.log"

    if [ ! -f "$SUBSET" ]; then
        echo "Missing subset: $SUBSET"
        exit 1
    fi

    for stage in unfolding denoising enhancing detecting sam; do
        local pf
        pf=$(param_file_for_stage "$TID" "$stage")
        if [ ! -f "$pf" ]; then
            echo "Missing params (run GLM orchestrator first): $pf"
            exit 1
        fi
    done

    {
        echo "tunnel=${TID} ablation=${ABLATION} model=${MODEL_TAG} started=$(date -Iseconds)"
        rm -rf "$WORK" "$DEST"
        mkdir -p "$SAM4TUN_DATA"
        ln -sfn "../../data/subsets/${TID}.txt" "$LINK"

        run_step "1/5 Unfolding (${TID})" "$PY" "${AGENTS_DIR}/unfolding.py" "$TID" --ablation "$ABLATION" --model "$MODEL_TAG"
        run_step "2/5 Denoising (${TID})" "$PY" "${AGENTS_DIR}/denoising.py" "$TID" --ablation "$ABLATION" --model "$MODEL_TAG"
        run_step "3/5 Enhancing (${TID})" "$PY" "${AGENTS_DIR}/enhancing.py" "$TID" --ablation "$ABLATION" --model "$MODEL_TAG"
        run_step "4/5 Detecting (${TID})" "$PY" "${AGENTS_DIR}/detecting.py" "$TID" --ablation "$ABLATION" --model "$MODEL_TAG"
        gpu_cleanup_before_sam
        run_step "5/5 SAM (${TID})" "$PY" "${AGENTS_DIR}/sam.py" "$TID" --ablation "$ABLATION" --model "$MODEL_TAG"

        if [ ! -d "$WORK" ]; then
            echo "Expected output dir missing: $WORK"
            exit 1
        fi

        mkdir -p "$OUT_ROOT"
        mv "$WORK" "$DEST"
        rm -f "$LINK"

        run_step "6/6 Evaluation (${TID})" "$PY" "${AGENTS_DIR}/evaluate_static.py" "$TID" --data-root "$OUT_ROOT"
        echo "tunnel=${TID} finished=$(date -Iseconds)"
    } 2>&1 | tee "$LOG"
}

append_summary_row() {
    local TID=$1
    local SEC=$2
    local PERF="${OUT_ROOT}/${TID}/evaluation/performance.md"
    local MIOU
    MIOU=$(extract_miou_from_perf_md "$PERF")
    [ -z "$MIOU" ] && MIOU="n/a"
    if [ ! -f "${OUT_ROOT}/run_summary.csv" ]; then
        echo "tunnel_id,ablation,model,wall_sec,mIoU" > "${OUT_ROOT}/run_summary.csv"
    fi
    echo "${TID},${ABLATION},${MODEL_TAG},${SEC},${MIOU}" >> "${OUT_ROOT}/run_summary.csv"
}

echo ""
echo "=========================================="
echo "Ablation sam4tun/agents (params only)"
echo "Condition: ${ABLATION_FOLDER}"
echo "Model tag: ${MODEL_TAG}"
echo "Tunnels:   ${TUNNEL_IDS}"
echo "Output:    ${OUT_ROOT}/<tunnel_id>/"
echo "Python:    ${PY}"
echo "=========================================="
echo ""

TUNNEL_RUN=0
for TID in $TUNNEL_IDS; do
    if [ "$SKIP_EXISTING" -eq 1 ] && [ -f "${OUT_ROOT}/${TID}/evaluation/performance.md" ]; then
        echo "Skipping ${TID} (already has evaluation)"
        continue
    fi

    TUNNEL_RUN=$((TUNNEL_RUN + 1))
    TUNNEL_T0=$(date +%s)

    echo ""
    echo "=========================================="
    echo "[${TUNNEL_RUN}] tunnel=${TID}"
    echo "=========================================="

    run_tunnel_pipeline "$TID"

    TUNNEL_T1=$(date +%s)
    TUNNEL_SEC=$((TUNNEL_T1 - TUNNEL_T0))
    append_summary_row "$TID" "$TUNNEL_SEC"

    MIOU_VAL=$(extract_miou_from_perf_md "${OUT_ROOT}/${TID}/evaluation/performance.md")
    [ -z "$MIOU_VAL" ] && MIOU_VAL="n/a"

    echo "=========================================="
    echo "[${TUNNEL_RUN}] done ${TID} — ${TUNNEL_SEC}s — mIoU: ${MIOU_VAL}"
    echo "=========================================="
    echo ""
done

echo "Finished ${TUNNEL_RUN} tunnel(s). Summary: ${OUT_ROOT}/run_summary.csv"
