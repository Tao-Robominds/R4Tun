#!/usr/bin/env bash
# Static baseline: sam4tun/agents on data/subsets → data/static/{tunnel_id}/
#
# Usage:
#   ./run_static_agents.sh --sanity              # gate: 1-1, 2-1, 3-1-1, 4-1, 5-1
#   ./run_static_agents.sh --all                 # all 30 subsets
#   ./run_static_agents.sh --all --skip-existing # skip tunnels with performance.md
#   ./run_static_agents.sh --families-2-5        # families 2,3,4,5 only (25 tunnels)
#   ./run_static_agents.sh 1-1 2-1               # explicit tunnel ids
#
# All stages use frozen parameters from sam4tun/agents/parameters/sample/ unless
# a tunnel-specific override exists (none for static baseline).

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
    echo "Static agents wall clock: ${sec}s (${m}m ${s}s)"
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

OUT_ROOT="data/static"
SAM4TUN_DATA="sam4tun/data"
AGENTS_DIR="sam4tun/agents"
LOG_DIR="logs/static"
SANITY_IDS="1-1 2-1 3-1-1 4-1 5-1"
FAMILIES_2_5_IDS="2-1 2-2 2-3 2-4 2-5 3-1-1 3-1-2 3-1-3 4-1 4-2 4-3 4-4 4-5 4-6 4-7 4-8 4-9 4-10 5-1 5-2 5-3 5-4 5-5 5-6 5-7"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

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

run_tunnel_pipeline() {
    local TID=$1
    local SUBSET="data/subsets/${TID}.txt"
    local LINK="${SAM4TUN_DATA}/${TID}.txt"
    local WORK="${SAM4TUN_DATA}/${TID}"
    local DEST="${OUT_ROOT}/${TID}"
    local LOG="${LOG_DIR}/${TID}.log"

    if [ ! -f "$SUBSET" ]; then
        echo "Missing subset: $SUBSET"
        exit 1
    fi

    {
        echo "tunnel=${TID} started=$(date -Iseconds)"
        rm -rf "$WORK" "$DEST"
        mkdir -p "$SAM4TUN_DATA"
        ln -sfn "../../data/subsets/${TID}.txt" "$LINK"

        run_step "1/5 Unfolding (${TID})" "$PY" "${AGENTS_DIR}/unfolding.py" "$TID"
        run_step "2/5 Denoising (${TID})" "$PY" "${AGENTS_DIR}/denoising.py" "$TID"
        run_step "3/5 Enhancing (${TID})" "$PY" "${AGENTS_DIR}/enhancing.py" "$TID"
        run_step "4/5 Detecting (${TID})" "$PY" "${AGENTS_DIR}/detecting.py" "$TID"
        gpu_cleanup_before_sam
        run_step "5/5 SAM (${TID})" "$PY" "${AGENTS_DIR}/sam.py" "$TID"

        if [ ! -d "$WORK" ]; then
            echo "Expected output dir missing: $WORK"
            exit 1
        fi

        mkdir -p "$OUT_ROOT"
        mv "$WORK" "$DEST"
        rm -f "$LINK"

        run_step "6/6 Evaluation (${TID})" "$PY" "${AGENTS_DIR}/evaluate_static.py" "$TID"
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
        echo "tunnel_id,wall_sec,mIoU" > "${OUT_ROOT}/run_summary.csv"
    fi
    echo "${TID},${SEC},${MIOU}" >> "${OUT_ROOT}/run_summary.csv"
}

write_sanity_gate() {
    local STATUS=$1
    local M11 M21 M311 M41 M51
    M11=$(extract_miou_from_perf_md "${OUT_ROOT}/1-1/evaluation/performance.md")
    M21=$(extract_miou_from_perf_md "${OUT_ROOT}/2-1/evaluation/performance.md")
    M311=$(extract_miou_from_perf_md "${OUT_ROOT}/3-1-1/evaluation/performance.md")
    M41=$(extract_miou_from_perf_md "${OUT_ROOT}/4-1/evaluation/performance.md")
    M51=$(extract_miou_from_perf_md "${OUT_ROOT}/5-1/evaluation/performance.md")

    cat > "${OUT_ROOT}/sanity_gate.md" <<EOF
# Static baseline sanity gate

- **Command:** \`./run_static_agents.sh --sanity\`
- **Started:** ${PIPELINE_T0_ISO}
- **Finished:** $(date -Iseconds)
- **Status:** ${STATUS}

## mIoU (7-class)

| tunnel_id | family | mIoU |
|-----------|--------|------|
| 1-1 | regular (1-*) | ${M11:-n/a} |
| 2-1 | regular (2-*) | ${M21:-n/a} |
| 3-1-1 | continuous | ${M311:-n/a} |
| 4-1 | complex | ${M41:-n/a} |
| 5-1 | complex | ${M51:-n/a} |

## Pass criteria

1. mIoU(1-1) and mIoU(2-1) > mIoU(3-1-1)
2. mIoU(3-1-1) > mIoU(4-1) and mIoU(5-1)
3. mIoU(4-1) and mIoU(5-1) < 0.15
EOF
}

check_sanity_gate() {
    local M11 M21 M311 M41 M51
    M11=$(extract_miou_from_perf_md "${OUT_ROOT}/1-1/evaluation/performance.md")
    M21=$(extract_miou_from_perf_md "${OUT_ROOT}/2-1/evaluation/performance.md")
    M311=$(extract_miou_from_perf_md "${OUT_ROOT}/3-1-1/evaluation/performance.md")
    M41=$(extract_miou_from_perf_md "${OUT_ROOT}/4-1/evaluation/performance.md")
    M51=$(extract_miou_from_perf_md "${OUT_ROOT}/5-1/evaluation/performance.md")

    for v in "$M11" "$M21" "$M311" "$M41" "$M51"; do
        if [ -z "$v" ] || [ "$v" = "n/a" ]; then
            write_sanity_gate "FAIL (missing metrics)"
            echo "Sanity gate FAIL: missing mIoU for one or more tunnels"
            return 1
        fi
    done

    "$PY" - <<PY
m11, m21, m311, m41, m51 = float("$M11"), float("$M21"), float("$M311"), float("$M41"), float("$M51")
ok = True
reasons = []
if not (m11 > m311 and m21 > m311):
    ok = False
    reasons.append("1-1/2-1 must beat 3-1-1")
if not (m311 > m41 and m311 > m51):
    ok = False
    reasons.append("3-1-1 must beat 4-1 and 5-1")
if not (m41 < 0.15 and m51 < 0.15):
    ok = False
    reasons.append("4-1 and 5-1 must be < 0.15")
if ok:
    print("PASS")
else:
    print("FAIL:", "; ".join(reasons))
    raise SystemExit(1)
PY
}

# --- Parse tunnel list ---
TUNNEL_IDS=""
RUN_SANITY=0
RUN_ALL=0
RUN_FAMILIES_2_5=0
SKIP_EXISTING=0

while [ $# -gt 0 ]; do
    case "$1" in
        --sanity)
            RUN_SANITY=1
            shift
            ;;
        --all)
            RUN_ALL=1
            shift
            ;;
        --families-2-5)
            RUN_FAMILIES_2_5=1
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

if [ "$RUN_SANITY" -eq 1 ]; then
    TUNNEL_IDS="$SANITY_IDS"
elif [ "$RUN_FAMILIES_2_5" -eq 1 ]; then
    TUNNEL_IDS="$FAMILIES_2_5_IDS"
elif [ "$RUN_ALL" -eq 1 ] || [ -z "${TUNNEL_IDS// }" ]; then
    TUNNEL_IDS=$(ls data/subsets/*.txt 2>/dev/null | xargs -I{} basename {} .txt | sort)
fi

TUNNEL_IDS=$(echo "$TUNNEL_IDS" | xargs)

if [ -z "$TUNNEL_IDS" ]; then
    echo "No tunnels found (data/subsets/*.txt)."
    exit 1
fi

TUNNEL_TOTAL=0
for _ in $TUNNEL_IDS; do TUNNEL_TOTAL=$((TUNNEL_TOTAL + 1)); done

echo ""
echo "=========================================="
echo "Static sam4tun/agents baseline"
echo "Tunnels: ${TUNNEL_TOTAL}"
echo "Output:  ${OUT_ROOT}/<tunnel_id>/"
echo "Python:  ${PY}"
echo "=========================================="
echo ""

TUNNEL_CURRENT=0
TUNNEL_RUN=0
for TID in $TUNNEL_IDS; do
    if [ "$SKIP_EXISTING" -eq 1 ] && [ -f "${OUT_ROOT}/${TID}/evaluation/performance.md" ]; then
        echo "Skipping ${TID} (already has ${OUT_ROOT}/${TID}/evaluation/performance.md)"
        continue
    fi

    TUNNEL_CURRENT=$((TUNNEL_CURRENT + 1))
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

if [ "$RUN_SANITY" -eq 1 ]; then
    if check_sanity_gate; then
        write_sanity_gate "PASS"
        echo "Sanity gate PASS — see ${OUT_ROOT}/sanity_gate.md"
    else
        write_sanity_gate "FAIL"
        exit 1
    fi
fi

echo "Finished ${TUNNEL_RUN} tunnel(s) (${TUNNEL_TOTAL} in list). Summary: ${OUT_ROOT}/run_summary.csv"
