#!/usr/bin/env bash
# Greedy eval of the training checkpoints (alg=default, temperature 0).
# Length 256 uses 256 steps. Set GEN_LENGTH=512 to use 512 steps.
#
#   DATASET=math METHOD=pivot_sd \
#   TEST_DATA=/path/to/math_test.json \
#   bash scripts/evaluate.sh
#
# AceCode trains one model and evaluates HumanEval+ and MBPP+:
#   DATASET=acecode METHOD=pivot_sd bash scripts/evaluate.sh
# CHECKPOINTS="1000" evaluates a single step. Default is 500 and 1000.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CHECKPOINTS="${CHECKPOINTS:-500 ${MAX_STEPS}}"

eval_one() {
  local method="$1"
  local eval_dataset="$2"
  local job ckpt save
  job="${DATASET}_${method}"
  for step in ${CHECKPOINTS}; do
    ckpt="${OUTPUT_DIR}/${job}/checkpoint-${step}"
    if [ "${DRY_RUN:-0}" != "1" ] && [ ! -d "${ckpt}" ]; then
      echo "Skipping missing checkpoint: ${ckpt}" >&2
      continue
    fi
    save="${EVAL_DIR}/${method}/${eval_dataset}/checkpoint-${step}"
    echo "eval method=${method} dataset=${eval_dataset} checkpoint=${ckpt}"
    args=(
      python -m pivot_sd.evaluate
      --model-path "${ckpt}"
      --save-dir "${save}"
      --dataset "${eval_dataset}"
      --gen-length "${GEN_LENGTH}"
      --block-length "${BLOCK_LENGTH}"
      --steps "${STEPS}" \
      --backbone "${BACKBONE}"
    )
    if [ "${eval_dataset}" = "math" ] || [ "${eval_dataset}" = "gsm8k" ]; then
      require TEST_DATA
      args+=(--data-path "${TEST_DATA}")
    fi
    launch "${args[@]}"
  done
}

methods=("${METHOD}")
if [ "${METHOD}" = "all" ]; then
  methods=(pivot_sd positive_all positive_pivot entropy_pivot posneg_all random_step random_token)
fi

for method in "${methods[@]}"; do
  if [ "${DATASET}" = "acecode" ]; then
    benchmarks="${BENCHMARK:-humaneval_plus mbpp_plus}"
    for bench in ${benchmarks}; do
      eval_one "${method}" "${bench}"
    done
  else
    eval_one "${method}" "${DATASET}"
  fi
done
