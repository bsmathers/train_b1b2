#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export TAUROSV1_SAVE_DIR="${TAUROSV1_SAVE_DIR:-${ROOT_DIR}/checkpoints}"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-${ROOT_DIR}/cache}"
export GOOD_REPLAYS_DIR="${GOOD_REPLAYS_DIR:-${METAMON_CACHE_DIR}/strong_model_replays_parsed}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ "${ENABLE_TORCH_COMPILE:-0}" != "1" ]; then
    export TORCHDYNAMO_DISABLE=1
fi

SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-stage1_b1}"
RUN_NAME="${RUN_NAME:-stage1_b1_actor_bc}"
PHASE2_CHECKPOINT="${PHASE2_CHECKPOINT:-${SAVE_DIR}/${SOURCE_RUN_NAME}/ckpts/policy_weights/policy_value_backbone_best.pt}"
DATASET_CONFIG="${DATASET_CONFIG:-${ROOT_DIR}/metamon/rl/configs/datasets/stage1_b1_strong_only.yaml}"
EPOCHS="${EPOCHS:-1000}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
DLOADER_WORKERS="${DLOADER_WORKERS:-8}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-500}"
PATIENCE="${PATIENCE:-6}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1}"

mkdir -p "${METAMON_CACHE_DIR}" "${SAVE_DIR}"
if [ ! -f "${PHASE2_CHECKPOINT}" ]; then
    echo "ERROR: Phase-2 checkpoint not found: ${PHASE2_CHECKPOINT}" >&2
    exit 1
fi
if [ -e "${SAVE_DIR}/${RUN_NAME}" ]; then
    echo "ERROR: clean actor-refinement run already exists: ${SAVE_DIR}/${RUN_NAME}" >&2
    echo "Choose a new RUN_NAME or remove that exact run directory before retrying." >&2
    exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required to prevent concurrent Stage1 B1 runs." >&2
    exit 1
fi
exec 9>"${SAVE_DIR}/.${RUN_NAME}.lock"
if ! flock -n 9; then
    echo "ERROR: another ${RUN_NAME} launcher or trainer is already running." >&2
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
    echo "ERROR: flash-attn is not installed in ${PYTHON_BIN}. Run the main CUDA provisioning once." >&2
    exit 1
fi
if [ ! -f "${TAUROSV1_SAVE_DIR}/mediumg1online/ckpts/policy_weights/policy_epoch_475.pt" ]; then
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/assemble_epoch475.py" \
        --repo-root "${ROOT_DIR}" --output-root "${TAUROSV1_SAVE_DIR}"
fi
if [ ! -d "${GOOD_REPLAYS_DIR}/gen1ou" ] && [ ! -f "${GOOD_REPLAYS_DIR}/gen1ou.tar" ]; then
    "${ROOT_DIR}/tools/prepare_strong_replays.sh"
fi

LOG_ARG=()
if [ -n "${WANDB_API_KEY:-}" ]; then
    LOG_ARG=(--log)
fi

echo "Starting clean Epoch20-style actor refinement"
echo "  Phase-2 source: ${PHASE2_CHECKPOINT}"
echo "  Strong trajectories: ${GOOD_REPLAYS_DIR}"
echo "  New run: ${SAVE_DIR}/${RUN_NAME}"

exec "${PYTHON_BIN}" -m metamon.rl.finetune \
    --run_name "${RUN_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --base_model TaurosV1 \
    --base_checkpoint 475 \
    --model_gin_config grouped_v2_medium_cuda.gin \
    --train_gin_config stage1_b1_actor_bc.gin \
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
    --early_stopping_min_delta 0.001 \
    --stage1_b1_actor_refinement \
    --b1_phase2_checkpoint "${PHASE2_CHECKPOINT}" \
    --b1_actor_epochs "${EPOCHS}" \
    --b1_actor_steps_per_epoch "${STEPS_PER_EPOCH}" \
    --b1_actor_patience "${PATIENCE}" \
    --eval_gens \
    "${LOG_ARG[@]}"
