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
  local source_param_file="${PARAM_DIR}/${model_name}/parameters_detecting.json"
  local dest_param_file="${WORK_PARAM_DIR}/parameters_detecting.json"

  echo "===== Running pipeline for ${model_name} ====="

  # Verify source file exists
  if [ ! -f "${source_param_file}" ]; then
    echo "❌ Error: Source parameter file not found: ${source_param_file}"
    exit 1
  fi

  # Copy that model's parameters into the place configurable_detecting.py expects
  echo "📋 Copying parameters from ${source_param_file} to ${dest_param_file}"
  cp "${source_param_file}" "${dest_param_file}"
  
  if [ ! -f "${dest_param_file}" ]; then
    echo "❌ Error: Failed to copy parameters for ${model_name}"
    exit 1
  fi
  echo "✅ Parameters copied successfully for ${model_name}"

  # Detection → SAM → evaluation
  echo "🔍 Running detection with ${model_name} parameters..."
  python configurable/configurable_detecting.py "${TUNNEL_ID}"
  python sam4tun/4-2_sam.py "${TUNNEL_ID}"
  python sam4tun/evaluation.py "${TUNNEL_ID}"

  # Create model-specific evaluation directory
  local model_eval_dir="${EVAL_DIR}/${model_name}"
  mkdir -p "${model_eval_dir}"

  # Move evaluation artifacts to model-specific folder
  echo "📁 Saving evaluation results to ${model_eval_dir}/"
  if [ -f "${EVAL_DIR}/performance.md" ]; then
    mv "${EVAL_DIR}/performance.md" "${model_eval_dir}/performance.md"
  fi
  if [ -f "${EVAL_DIR}/iou_by_class.png" ]; then
    mv "${EVAL_DIR}/iou_by_class.png" "${model_eval_dir}/iou_by_class.png"
  fi
  if [ -f "${EVAL_DIR}/class_distribution.png" ]; then
    mv "${EVAL_DIR}/class_distribution.png" "${model_eval_dir}/class_distribution.png"
  fi
  echo "✅ Evaluation results saved to ${model_eval_dir}/"
}

# run_one "gemini_3"
# run_one "gpt_5.2"
run_one "opus_4.5"
run_one "group"






