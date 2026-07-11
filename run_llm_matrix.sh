#!/usr/bin/env bash
# Run m / m+s / m+s+k with Opus-4.6, GPT-5.4, Gemini-3-Flash on sanity tunnels.
# Excludes GLM/DeepSeek. After each run, renames output dir to {tid}_{modeltag}.
#
# Usage:
#   ./run_llm_matrix.sh                    # all 45 runs (minus gate skip)
#   ./run_llm_matrix.sh --skip-gate        # include gate cell if re-running
#   ./run_llm_matrix.sh --model opus4.6    # one model only
#   ./run_llm_matrix.sh --condition m      # one condition only
#   ./run_llm_matrix.sh 1-1 2-1            # explicit tunnels

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH=".:sam4tun/segment-anything:${PYTHONPATH:-}"

if [ -x "${SCRIPT_DIR}/venv/bin/python3" ]; then
    PY="${SCRIPT_DIR}/venv/bin/python3"
else
    PY="python3"
fi

SANITY_IDS="1-1 2-1 3-1-1 4-1 5-1"
GATE_TUNNEL="1-1"
GATE_COND="m_s_k"
GATE_MODEL="opus4.6"
SKIP_GATE=0
FILTER_MODEL=""
FILTER_COND=""
TUNNEL_IDS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-gate) SKIP_GATE=1; shift ;;
        --model) FILTER_MODEL="${2:?}"; shift 2 ;;
        --condition) FILTER_COND="${2:?}"; shift 2 ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *) TUNNEL_IDS="${TUNNEL_IDS} $1"; shift ;;
    esac
done

TUNNEL_IDS=$(echo "${TUNNEL_IDS:-$SANITY_IDS}" | xargs)

cond_folder() {
    case "$1" in
        m) echo "memory" ;;
        m_s) echo "memory+state" ;;
        m_s_k) echo "memory+state+knowledge" ;;
        *) echo "ERROR: unknown condition $1" >&2; exit 1 ;;
    esac
}

orchestrator_for() {
    local cond=$1 model=$2
    case "${cond}:${model}" in
        m:opus4.6) echo "run_memory.py" ;;
        m:gpt5.4) echo "run_memory_gpt.py" ;;
        m:gemini3flash) echo "run_memory_gemini.py" ;;
        m_s:opus4.6) echo "run_memory_state.py" ;;
        m_s:gpt5.4) echo "run_memory_state_gpt.py" ;;
        m_s:gemini3flash) echo "run_memory_state_gemini.py" ;;
        m_s_k:opus4.6) echo "run_memory_state_knowledge.py" ;;
        m_s_k:gpt5.4) echo "run_memory_state_knowledge_gpt.py" ;;
        m_s_k:gemini3flash) echo "run_memory_state_knowledge_gemini.py" ;;
        *) echo "ERROR: no orchestrator for ${cond} ${model}" >&2; exit 1 ;;
    esac
}

extract_miou() {
    local f=$1
    if [ -f "$f" ]; then
        grep -m1 'Mean IoU (mIoU):' "$f" 2>/dev/null | sed -E 's/.*: *//' | tr -d '\r' || true
    fi
}

mkdir -p logs/ablation

MODELS=(opus4.6 gpt5.4 gemini3flash)
CONDS=(m m_s m_s_k)

RUN_ONE() {
    local TID=$1 COND=$2 MODEL=$3
    local FOLDER
    FOLDER=$(cond_folder "$COND")
    local ORCH
    ORCH=$(orchestrator_for "$COND" "$MODEL")
    local OUT_ROOT="data/ablation/${FOLDER}"
    local DEST="${OUT_ROOT}/${TID}_${MODEL}"
    local LOG="logs/ablation/${TID}_${COND}_${MODEL}.log"

    if [ -f "${DEST}/evaluation/performance.md" ]; then
        echo "SKIP ${TID} ${COND} ${MODEL} (already ${DEST})"
        return 0
    fi

    # Clean scratch for a full pipeline run (params live under agents/ablation/)
    rm -rf "${OUT_ROOT}/${TID}"

    echo ""
    echo "=========================================="
    echo "RUN ${TID} ${COND} ${MODEL}"
    echo "Orchestrator: ${ORCH}"
    echo "=========================================="

    local T0
    T0=$(date +%s)
    if "$PY" "$ORCH" "$TID" --model "$MODEL" 2>&1 | tee "$LOG"; then
        :
    else
        echo "FAIL ${TID} ${COND} ${MODEL} (see ${LOG})"
        return 1
    fi

    if [ ! -d "${OUT_ROOT}/${TID}" ]; then
        echo "FAIL ${TID} ${COND} ${MODEL}: missing ${OUT_ROOT}/${TID}"
        return 1
    fi

    if [ -d "$DEST" ]; then
        rm -rf "$DEST"
    fi
    mv "${OUT_ROOT}/${TID}" "$DEST"

    if [ ! -f "${DEST}/evaluation/performance.md" ]; then
        echo "FAIL ${TID} ${COND} ${MODEL}: missing ${DEST}/evaluation/performance.md"
        rm -rf "$DEST"
        return 1
    fi

    local T1 SEC MIOU
    T1=$(date +%s)
    SEC=$((T1 - T0))
    MIOU=$(extract_miou "${DEST}/evaluation/performance.md")
    [ -z "$MIOU" ] && MIOU="n/a"
    echo "DONE ${TID} ${COND} ${MODEL} — ${SEC}s — mIoU: ${MIOU} — ${DEST}"
    return 0
}

FAILURES=()
TOTAL=0
DONE=0

for MODEL in "${MODELS[@]}"; do
    [ -n "$FILTER_MODEL" ] && [ "$MODEL" != "$FILTER_MODEL" ] && continue
    for COND in "${CONDS[@]}"; do
        [ -n "$FILTER_COND" ] && [ "$COND" != "$FILTER_COND" ] && continue
        for TID in $TUNNEL_IDS; do
            if [ "$SKIP_GATE" -eq 0 ] && [ "$TID" = "$GATE_TUNNEL" ] && [ "$COND" = "$GATE_COND" ] && [ "$MODEL" = "$GATE_MODEL" ]; then
                echo "SKIP gate cell ${TID} ${COND} ${MODEL} (run separately)"
                continue
            fi
            TOTAL=$((TOTAL + 1))
            if RUN_ONE "$TID" "$COND" "$MODEL"; then
                DONE=$((DONE + 1))
            else
                FAILURES+=("${TID}:${COND}:${MODEL}")
            fi
        done
    done
done

echo ""
echo "=========================================="
echo "Matrix finished: ${DONE}/${TOTAL} succeeded"
if [ ${#FAILURES[@]} -gt 0 ]; then
    echo "Failures:"
    printf '  %s\n' "${FAILURES[@]}"
    exit 1
fi
echo "=========================================="
