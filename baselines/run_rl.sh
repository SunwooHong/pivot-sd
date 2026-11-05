#!/usr/bin/env bash
# Budget-matched diffusion RL from the paper.
#
#   METHOD=wd1pp DATASET=math \
#   MODEL_PATH=GSAI-ML/LLaDA-8B-Instruct \
#   DATA_PATH=/path/to/math_200.jsonl \
#   GPU_ID=0,1 NUM_GPUS=2 \
#   bash baselines/run_rl.sh
#
# METHOD is wd1pp or diffu_grpo. DATASET is math, gsm8k, or acecode.
# Reported runs used two GPUs, 4 completions per prompt, and 1000 steps.
# The longer RL rows in the paper use the same config on 2000 prompts
# with MAX_STEPS=5000.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

METHOD="${METHOD:-wd1pp}"
DATASET="${DATASET:-math}"
NUM_GPUS="${NUM_GPUS:-2}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-}"

case "${METHOD}" in
  wd1pp|wd1++) TRAINER=wll_d1_neg ;;
  diffu_grpo|diffu-grpo|d1) TRAINER=d1 ;;
  *)
    echo "METHOD must be wd1pp or diffu_grpo." >&2
    exit 1
    ;;
esac

: "${MODEL_PATH:?Set MODEL_PATH.}"
: "${DATA_PATH:?Set DATA_PATH to the training JSONL.}"
if [ ! -f "${DATA_PATH}" ]; then
  echo "Training file not found: ${DATA_PATH}" >&2
  exit 1
fi

case "${DATASET}" in
  math)
    CFG="wd1/train_math_bf16_s200_max512drop.yaml"
    export MATH_TRAIN_JSON="${DATA_PATH}"
    ;;
  gsm8k)
    CFG="wd1/train_gsm8k_bf16_s200_max512drop.yaml"
    export GSM8K_TRAIN_JSON="${DATA_PATH}"
    ;;
  acecode)
    CFG="wd1/train_acecode_bf16_s200_max512drop.yaml"
    export ACECODE_TRAIN_JSON="${DATA_PATH}"
    export CODE_PROMPT_STYLE=raw_with_tests
    export ACECODE_RUN_VARIANT=with_tests
    export ACECODE_INCLUDE_TESTS_IN_PROMPT=1
    export ACECODE_PROMPT_NUM_TESTS="${ACECODE_PROMPT_NUM_TESTS:-3}"
    ;;
  *)
    echo "DATASET must be math, gsm8k, or acecode." >&2
    exit 1
    ;;
esac

if [ -n "${GPU_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
if [ "${NUM_GPUS}" -le 1 ]; then
  ACCEL="wd1/accelerate_1gpu.yaml"
elif [ "${NUM_GPUS}" -eq 2 ]; then
  ACCEL="wd1/accelerate_2gpu.yaml"
else
  ACCEL="wd1/accelerate.yaml"
fi

export MODEL_PATH
export PROMPT_LENGTH_POLICY="${PROMPT_LENGTH_POLICY:-drop}"
export MAX_PROMPT_DROP_TOKENS="${MAX_PROMPT_DROP_TOKENS:-512}"
export WANDB_PROJECT="${WANDB_PROJECT:-pivot-sd-${METHOD}}"
export WANDB_MODE="${WANDB_MODE:-offline}"

RUN_NAME="${RUN_NAME:-${DATASET}_${METHOD}_s${SEED}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/../runs/${DATASET}/rl/${METHOD}}"
mkdir -p "${OUTPUT_DIR}"

EXTRA=()
if [ -n "${MAX_STEPS}" ]; then
  EXTRA+=(--max_steps "${MAX_STEPS}")
fi

echo "rl method=${METHOD} trainer=${TRAINER} dataset=${DATASET}"
echo "data=${DATA_PATH}"
echo "output=${OUTPUT_DIR}"

accelerate launch \
  --config_file "${ACCEL}" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT:-12370}" \
  wd1/run_train.py \
  --config "${CFG}" \
  --model_path "${MODEL_PATH}" \
  --num_iterations "${NUM_ITER:-12}" \
  --dataset "${DATASET}" \
  --trainer_type "${TRAINER}" \
  --run_name "${RUN_NAME}" \
  --wandb_project "${WANDB_PROJECT}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  "${EXTRA[@]}"
