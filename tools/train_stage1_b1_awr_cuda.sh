#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SAVE_DIR="${SAVE_DIR:-/workspace/b1-data/checkpoints}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-stage1_b1_epoch51_selected_actor}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${SAVE_DIR}/${SOURCE_RUN_NAME}/ckpts/policy_weights/policy_actor_bc_final.pt}"
RUN_NAME="${RUN_NAME:-stage1_b1_epoch51_awr}"
export TAUROSV1_SAVE_DIR="${TAUROSV1_SAVE_DIR:-${SAVE_DIR}}"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-/workspace/b1-data/cache}"
export GOOD_REPLAYS_DIR="${GOOD_REPLAYS_DIR:-${METAMON_CACHE_DIR}/strong_model_replays_parsed}"
export SELECTED_STRONG_REPLAYS_DIR="${SELECTED_STRONG_REPLAYS_DIR:-${METAMON_CACHE_DIR}/strong_model_replays_selected}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ "${ENABLE_TORCH_COMPILE:-0}" != "1" ]; then
    export TORCHDYNAMO_DISABLE=1
fi

EPOCHS="${EPOCHS:-200}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
DLOADER_WORKERS="${DLOADER_WORKERS:-8}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-500}"
PATIENCE="${PATIENCE:-15}"
MIN_DELTA="${MIN_DELTA:-0.0}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1}"
DATASET_CONFIG="${ROOT_DIR}/metamon/rl/configs/datasets/stage1_b1_selected_strong_only.yaml"

mkdir -p "${METAMON_CACHE_DIR}" "${SAVE_DIR}"
if [ ! -f "${SOURCE_CHECKPOINT}" ]; then
    echo "ERROR: completed actor checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
    exit 1
fi
if [ -e "${SAVE_DIR}/${RUN_NAME}" ]; then
    echo "ERROR: AWR run already exists: ${SAVE_DIR}/${RUN_NAME}" >&2
    echo "Choose a new RUN_NAME; the completed source run will not be modified." >&2
    exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required to prevent concurrent training runs." >&2
    exit 1
fi
exec 9>"${SAVE_DIR}/.${RUN_NAME}.lock"
if ! flock -n 9; then
    echo "ERROR: another ${RUN_NAME} launcher is already running." >&2
    exit 1
fi

cd "${ROOT_DIR}"
if ! "${PYTHON_BIN}" -c 'from importlib.metadata import version; version("poke-env"); version("amago")' >/dev/null 2>&1; then
    echo "Installing Metamon and pinned Python dependencies"
    "${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}"
fi
if ! "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 1)'; then
    echo "ERROR: a BF16-capable CUDA GPU is required." >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c 'import flash_attn' >/dev/null 2>&1; then
    echo "ERROR: flash-attn is not installed in ${PYTHON_BIN}." >&2
    exit 1
fi
if [ ! -d "${TAUROSV1_SAVE_DIR}/mediumg1online/ckpts" ]; then
    echo "ERROR: TaurosV1 architecture directory is missing: ${TAUROSV1_SAVE_DIR}/mediumg1online/ckpts" >&2
    exit 1
fi
if [ ! -d "${GOOD_REPLAYS_DIR}/gen1ou" ] && [ ! -f "${GOOD_REPLAYS_DIR}/gen1ou.tar" ]; then
    "${ROOT_DIR}/tools/prepare_strong_replays.sh"
fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/prepare_selected_strong_replays.py" \
    --source "${GOOD_REPLAYS_DIR}" \
    --output "${SELECTED_STRONG_REPLAYS_DIR}"

LOG_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
    LOG_ARGS=(--log)
fi

echo "Starting Monte Carlo advantage-weighted cloning"
echo "  Actor and value anchor: ${SOURCE_CHECKPOINT}"
echo "  Selected strong trajectories: ${SELECTED_STRONG_REPLAYS_DIR}"
echo "  New run: ${SAVE_DIR}/${RUN_NAME}"
echo "  AdamW updates per epoch: ${STEPS_PER_EPOCH}"
echo "  Early stopping: patience=${PATIENCE} min_delta=${MIN_DELTA}"

exec "${PYTHON_BIN}" -m metamon.rl.finetune \
    --run_name "${RUN_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --base_model TaurosV1 \
    --model_gin_config grouped_v2_medium_cuda.gin \
    --train_gin_config stage1_b1_actor_awr.gin \
    --dataset_config "${DATASET_CONFIG}" \
    --epochs "${EPOCHS}" \
    --steps_per_epoch "${STEPS_PER_EPOCH}" \
    --batch_size_per_gpu "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --mixed_precision bf16 \
    --dloader_workers "${DLOADER_WORKERS}" \
    --async_env_mp_context spawn \
    --ckpt_interval "${CKPT_INTERVAL}" \
    --validation_fraction 0.10 \
    --validation_batches "${VALIDATION_BATCHES}" \
    --validation_seed 42 \
    --early_stopping_min_delta "${MIN_DELTA}" \
    --stage1_b1_actor_refinement \
    --b1_anchor_checkpoint "${SOURCE_CHECKPOINT}" \
    --b1_phase2_checkpoint "${SOURCE_CHECKPOINT}" \
    --b1_actor_epochs "${EPOCHS}" \
    --b1_actor_steps_per_epoch "${STEPS_PER_EPOCH}" \
    --b1_actor_patience "${PATIENCE}" \
    --eval_gens \
    "${LOG_ARGS[@]}"
