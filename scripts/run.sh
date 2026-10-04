#!/usr/bin/env bash
# Run the sample-200 pipeline: generate, build pivots, train, evaluate.
#
# Main method on MATH:
#   MODEL_PATH=/path/to/LLaDA-8B-Instruct \
#   DATA_PATH=/path/to/math_200.jsonl \
#   TEST_DATA=/path/to/math_test.json \
#   DATASET=math METHOD=pivot_sd \
#   bash scripts/run.sh
#
# One stage: STAGE=build bash scripts/run.sh
# Every ablation: METHOD=all STAGE=train bash scripts/run.sh
# Two GPUs, as in the reported jobs: GPU_ID=0,1 NUM_GPUS=2
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require MODEL_PATH
STAGE="${STAGE:-all}"
HERE="$(dirname "${BASH_SOURCE[0]}")"

run_stage() {
  case "$1" in
    generate) bash "${HERE}/generate.sh" ;;
    build) bash "${HERE}/build_pivots.sh" ;;
    train) bash "${HERE}/train.sh" ;;
    eval|evaluate) bash "${HERE}/evaluate.sh" ;;
    *)
      echo "Unknown STAGE=${1}. Use generate, build, train, eval, or all." >&2
      exit 1
      ;;
  esac
}

if [ "${STAGE}" = "all" ]; then
  require DATA_PATH
  if [ "${DATASET}" != "acecode" ]; then
    require TEST_DATA
  fi
  run_stage generate
  run_stage build
  run_stage train
  run_stage eval
else
  run_stage "${STAGE}"
fi
