#!/usr/bin/env bash
# Sample 4 denoising trajectories per problem (alg=default, temperature 0.9).
#
#   MODEL_PATH=/path/to/LLaDA-8B-Instruct \
#   DATA_PATH=/path/to/math_200.jsonl \
#   DATASET=math \
#   bash scripts/generate.sh
#
# Reported jobs used two GPUs: GPU_ID=0,1 NUM_GPUS=2
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require MODEL_PATH
require DATA_PATH

echo "generate dataset=${DATASET} gpus=${NUM_GPUS} save_dir=${SAVE_DIR}"
launch python -m pivot_sd.generate \
  --dataset "${DATASET}" \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --save-dir "${SAVE_DIR}" \
  --gen-length "${GEN_LENGTH}" \
  --block-length "${BLOCK_LENGTH}" \
  --steps "${STEPS}" \
  --temperature "${TEMPERATURE}" \
  --num-samples "${NUM_SAMPLES}" \
  --alg "${ALG}" \
  --backbone "${BACKBONE}" \
  "$@"

echo "stepwise: $(trace_run_dir)/stepwise"
echo "results:  $(trace_run_dir)/all_results.json"
