#!/usr/bin/env bash
# Build the pivot JSONL files used for training.
# ig_density is Pivot-SD. random_step and entropy are the selection ablations.
# Advantage is min-max scaled at the 1st and 99th percentiles. No rebalance.
#
#   MODEL_PATH=/path/to/LLaDA-8B-Instruct \
#   DATA_PATH=/path/to/math_200.jsonl \
#   DATASET=math \
#   bash scripts/build_pivots.sh
#
# SELECTORS="ig_density" builds only the main file.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

require MODEL_PATH
require DATA_PATH

RUN_DIR="$(trace_run_dir)"
STEPWISE_DIR="${STEPWISE_DIR:-${RUN_DIR}/stepwise}"
ALL_RESULTS="${ALL_RESULTS:-${RUN_DIR}/all_results.json}"
SELECTORS="${SELECTORS:-ig_density random_step entropy}"

if [ "${DRY_RUN:-0}" != "1" ] && [ ! -d "${STEPWISE_DIR}" ]; then
  echo "Stepwise directory not found: ${STEPWISE_DIR}" >&2
  echo "Run scripts/generate.sh first, or set SAVE_DIR / STEPWISE_DIR." >&2
  exit 1
fi

mkdir -p "${PIVOT_DIR}"
echo "build dataset=${DATASET} stepwise=${STEPWISE_DIR}"

for selector in ${SELECTORS}; do
  case "${selector}" in
    ig_density) out="${PIVOT_DIR}/pivots_ig.jsonl" ;;
    random_step) out="${PIVOT_DIR}/pivots_random.jsonl" ;;
    entropy) out="${PIVOT_DIR}/pivots_entropy.jsonl" ;;
    *)
      echo "Unknown selector: ${selector}" >&2
      exit 1
      ;;
  esac
  args=(
    -m pivot_sd.build_pivots
    --selector "${selector}"
    --dataset "${DATASET}"
    --stepwise-dir "${STEPWISE_DIR}"
    --train-json "${DATA_PATH}"
    --tokenizer "${MODEL_PATH}"
    --out-jsonl "${out}"
    --top-k "${TOP_K}"
    --p-low "${P_LOW}"
    --p-high "${P_HIGH}"
    --random-seed "${SEED}"
    --gen-length "${GEN_LENGTH}"
    --block-size "${PIVOT_BLOCK_SIZE}"
    --qid-offset "${QID_OFFSET:-0}"
  )
  if [ -f "${ALL_RESULTS}" ]; then
    args+=(--all-results-json "${ALL_RESULTS}")
  elif [ "${DATASET}" = "acecode" ]; then
    echo "AceCode pivot building needs ${ALL_RESULTS}" >&2
    exit 1
  fi
  echo "selector=${selector} -> ${out}"
  py "${args[@]}"
done
