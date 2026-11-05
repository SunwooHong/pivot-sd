# Shared settings for the sample-200 Pivot-SD scripts.
# Source this file; do not execute it.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${HOME}/.local/bin:${PATH}"
case ":${PYTHONPATH:-}:" in
  *":${ROOT}:"*) ;;
  *) export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
esac
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATASET="${DATASET:-math}"
METHOD="${METHOD:-pivot_sd}"
BACKBONE="${BACKBONE:-llada}"
case "${BACKBONE}" in
  llada|LLaDA)
    BACKBONE=llada
    FAMILY=LLaDA
    ALG="${ALG:-default}"
    PIVOT_BLOCK_SIZE="${PIVOT_BLOCK_SIZE:-64}"
    ;;
  dream|Dream)
    BACKBONE=dream
    FAMILY=Dream
    ALG="${ALG:-maskgit_plus}"
    PIVOT_BLOCK_SIZE="${PIVOT_BLOCK_SIZE:-0}"
    ;;
  *)
    echo "BACKBONE must be llada or dream, got ${BACKBONE}." >&2
    exit 1
    ;;
esac
GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-64}"
STEPS="${STEPS:-${GEN_LENGTH}}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_K="${TOP_K:-10}"
MAX_STEPS="${MAX_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SEED="${SEED:-42}"
P_LOW="${P_LOW:-1}"
P_HIGH="${P_HIGH:-99}"

if [ -n "${GPU_ID:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
if [ -z "${NUM_GPUS:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
  else
    NUM_GPUS=1
  fi
fi
MASTER_PORT="${MASTER_PORT:-$((10000 + $$ % 50000))}"

SAVE_DIR="${SAVE_DIR:-${ROOT}/runs/${DATASET}/gen}"
PIVOT_DIR="${PIVOT_DIR:-${ROOT}/runs/${DATASET}/pivots}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/${DATASET}/train}"
EVAL_DIR="${EVAL_DIR:-${ROOT}/runs/${DATASET}/eval}"

# Use the parent repo environment when this folder sits inside KLASS.
if [ -z "${UV_PROJECT:-}" ] && [ -f "${ROOT}/../pyproject.toml" ] && [ -d "${ROOT}/../.venv" ]; then
  UV_PROJECT="${ROOT}/.."
fi

py() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf 'PYTHONPATH=%q ' "${PYTHONPATH}"
    if command -v uv >/dev/null 2>&1; then
      if [ -n "${UV_PROJECT:-}" ]; then
        printf 'uv run --project %q python' "${UV_PROJECT}"
      else
        printf 'uv run python'
      fi
    else
      printf 'python3'
    fi
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  if command -v uv >/dev/null 2>&1; then
    if [ -n "${UV_PROJECT:-}" ]; then
      uv run --project "${UV_PROJECT}" python "$@"
    else
      uv run python "$@"
    fi
  else
    python3 "$@"
  fi
}

launch() {
  # Generation, training, and eval were started with accelerate.
  # Callers pass `python -m pivot_sd....`. accelerate launch already runs
  # inside the project interpreter, so the leading python executable is dropped.
  if [ "${1:-}" = "python" ] || [ "${1:-}" = "python3" ]; then
    shift
  fi
  if [ "${DRY_RUN:-0}" = "1" ]; then
    if command -v uv >/dev/null 2>&1; then
      if [ -n "${UV_PROJECT:-}" ]; then
        printf 'uv run --project %q accelerate launch' "${UV_PROJECT}"
      else
        printf 'uv run accelerate launch'
      fi
    else
      printf 'accelerate launch'
    fi
    printf ' --main_process_port %q --num_processes %q' "${MASTER_PORT}" "${NUM_GPUS}"
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  if command -v uv >/dev/null 2>&1; then
    if [ -n "${UV_PROJECT:-}" ]; then
      uv run --project "${UV_PROJECT}" accelerate launch \
        --main_process_port "${MASTER_PORT}" \
        --num_processes "${NUM_GPUS}" \
        "$@"
    else
      uv run accelerate launch \
        --main_process_port "${MASTER_PORT}" \
        --num_processes "${NUM_GPUS}" \
        "$@"
    fi
  else
    accelerate launch \
      --main_process_port "${MASTER_PORT}" \
      --num_processes "${NUM_GPUS}" \
      "$@"
  fi
}

require() {
  local name="$1"
  if [ -z "${!name:-}" ]; then
    echo "Set ${name}." >&2
    exit 1
  fi
}

trace_run_dir() {
  local dataset_save="${DATASET}"
  if [ "${DATASET}" = "acecode" ]; then
    dataset_save="acecode/with_tests"
  fi
  printf '%s/%s/%s/%s/all/len_%s_block_%s/steps_%s/s%s\n' \
    "${SAVE_DIR}" "${FAMILY}" "${dataset_save}" "${ALG}" \
    "${GEN_LENGTH}" "${BLOCK_LENGTH}" "${STEPS}" "${NUM_SAMPLES}"
}

pivot_jsonl_for_method() {
  local method="$1"
  case "${method}" in
    random_step) printf '%s/pivots_random.jsonl\n' "${PIVOT_DIR}" ;;
    entropy_pivot) printf '%s/pivots_entropy.jsonl\n' "${PIVOT_DIR}" ;;
    *) printf '%s/pivots_ig.jsonl\n' "${PIVOT_DIR}" ;;
  esac
}
