#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Activate existing virtual environment
if [ -f "venv/bin/activate" ]; then
  # shellcheck source=/dev/null
  source venv/bin/activate
else
  echo "venv not found in project root. Please create/activate it manually."
  exit 1
fi

TUNNEL_ID="sample"
PARAM_DIR="configurable/council"
WORK_PARAM_DIR="configurable/${TUNNEL_ID}"
EVAL_DIR="data/${TUNNEL_ID}/evaluation"

mkdir -p "${WORK_PARAM_DIR}"
mkdir -p "${EVAL_DIR}"

run_one() {
  local model_name="$1"

  echo "===== Running pipeline for ${model_name} ====="

  # Copy that model's parameters into the place configurable_detecting.py expects
  cp "${PARAM_DIR}/${model_name}/parameters_detecting.json" \
     "${WORK_PARAM_DIR}/parameters_detecting.json"

  # Detection → SAM → evaluation
  python configurable/configurable_detecting.py "${TUNNEL_ID}"
  python sam4tun/4-2_sam.py "${TUNNEL_ID}"
  python sam4tun/evaluation.py "${TUNNEL_ID}"

  # Rename evaluation artifacts with model-specific suffix
  mv "${EVAL_DIR}/performance.md" \
     "${EVAL_DIR}/performance_${model_name}.md"
  mv "${EVAL_DIR}/iou_by_class.png" \
     "${EVAL_DIR}/iou_by_class_${model_name}.png"
  mv "${EVAL_DIR}/class_distribution.png" \
     "${EVAL_DIR}/class_distribution_${model_name}.png"
}

run_one "gemini_3"
run_one "gpt_5.2"
run_one "opus_4.5"
run_one "group"




