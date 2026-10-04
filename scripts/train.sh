#!/usr/bin/env bash
# Train one method, or every method in the ablation table.
#
#   MODEL_PATH=/path/to/LLaDA-8B-Instruct \
#   DATA_PATH=/path/to/math_200.jsonl \
#   DATASET=math METHOD=pivot_sd \
#   bash scripts/train.sh
#
# METHOD=all trains pivot_sd, positive_all, positive_pivot, entropy_pivot,
# posneg_all, random_step, and random_token.
# random_step reads pivots_random.jsonl. entropy_pivot reads pivots_entropy.jsonl.
# The others read pivots_ig.jsonl.
# Reported jobs used two GPUs and saved checkpoints at steps 500 and 1000.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require MODEL_PATH
require DATA_PATH

train_one() {
  local method="$1"
  local pivot job
  pivot="$(pivot_jsonl_for_method "${method}")"
  if [ -n "${PIVOT_JSONL:-}" ] && [ "${method}" = "${METHOD}" ]; then
    pivot="${PIVOT_JSONL}"
  fi
  if [ "${DRY_RUN:-0}" != "1" ] && [ ! -f "${pivot}" ]; then
    echo "Pivot file not found: ${pivot}" >&2
    echo "Run scripts/build_pivots.sh first." >&2
    exit 1
  fi
  job="${DATASET}_${method}"
  echo "train method=${method} dataset=${DATASET} pivot=${pivot}"
  echo "checkpoints: ${OUTPUT_DIR}/${job}/checkpoint-${SAVE_STEPS} and checkpoint-${MAX_STEPS}"
  launch python -m pivot_sd.train \
    --method "${method}" \
    --dataset "${DATASET}" \
    --model-name "${MODEL_PATH}" \
    --pivot-jsonl "${pivot}" \
    --train-json "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --job-name "${job}" \
    --max-steps "${MAX_STEPS}" \
    --save-steps "${SAVE_STEPS}" \
    --seed "${SEED}" \
    --report-to "${REPORT_TO:-none}"
}

if [ "${METHOD}" = "all" ]; then
  for method in pivot_sd positive_all positive_pivot entropy_pivot posneg_all random_step random_token; do
    train_one "${method}"
  done
else
  train_one "${METHOD}"
fi
