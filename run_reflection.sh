#!/usr/bin/env bash
# Reflection ablation: LLM (detecting + SAM) + pipeline stages 4–5 + evaluation.
# Usage:
#   ./run_reflection.sh 1-1
#   ./run_reflection.sh --all
#   ./run_reflection.sh 1-1 --dry-run

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d venv ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
fi

if [ -x "${SCRIPT_DIR}/venv/bin/python3" ]; then
  PY="${SCRIPT_DIR}/venv/bin/python3"
elif [ -x "${SCRIPT_DIR}/venv/bin/python" ]; then
  PY="${SCRIPT_DIR}/venv/bin/python"
else
  PY="python3"
fi

exec "${PY}" "${SCRIPT_DIR}/run_reflection.py" "$@"
