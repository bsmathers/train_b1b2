#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints}"
ANCHOR_RUN_NAME="${ANCHOR_RUN_NAME:-stage1_b1_actor_bc}"
ANCHOR_EPOCH="${ANCHOR_EPOCH:-51}"
ANCHOR_CHECKPOINT="${ANCHOR_CHECKPOINT:-${SAVE_DIR}/${ANCHOR_RUN_NAME}/ckpts/policy_weights/policy_epoch_${ANCHOR_EPOCH}.pt}"
VALUE_RUN_NAME="${VALUE_RUN_NAME:-stage1_b1_epoch51_value_refit}"
ACTOR_RUN_NAME="${ACTOR_RUN_NAME:-stage1_b1_epoch51_selected_actor}"

export TAUROSV1_SAVE_DIR="${TAUROSV1_SAVE_DIR:-${SAVE_DIR}}"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-${ROOT_DIR}/cache}"
export GOOD_REPLAYS_DIR="${GOOD_REPLAYS_DIR:-${METAMON_CACHE_DIR}/strong_model_replays_parsed}"
export SELECTED_STRONG_REPLAYS_DIR="${SELECTED_STRONG_REPLAYS_DIR:-${METAMON_CACHE_DIR}/strong_model_replays_selected}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ "${ENABLE_TORCH_COMPILE:-0}" != "1" ]; then
    export TORCHDYNAMO_DISABLE=1
fi

VALUE_EPOCHS="${VALUE_EPOCHS:-100}"
VALUE_STEPS_PER_EPOCH="${VALUE_STEPS_PER_EPOCH:-1000}"
VALUE_PATIENCE="${VALUE_PATIENCE:-3}"
ACTOR_EPOCHS="${ACTOR_EPOCHS:-1000}"
ACTOR_STEPS_PER_EPOCH="${ACTOR_STEPS_PER_EPOCH:-4000}"
ACTOR_PATIENCE="${ACTOR_PATIENCE:-6}"
BATCH_SIZE="${BATCH_SIZE:-12}"
ACTOR_GRAD_ACCUM="${ACTOR_GRAD_ACCUM:-4}"
DLOADER_WORKERS="${DLOADER_WORKERS:-8}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-500}"
CKPT_INTERVAL="${CKPT_INTERVAL:-1}"

VALUE_CHECKPOINT="${SAVE_DIR}/${VALUE_RUN_NAME}/ckpts/policy_weights/policy_value_head_refit_final.pt"
ACTOR_CHECKPOINT="${SAVE_DIR}/${ACTOR_RUN_NAME}/ckpts/policy_weights/policy_actor_bc_final.pt"
VALUE_DATASET="${ROOT_DIR}/metamon/rl/configs/datasets/stage1_b1_epoch51_value_refit.yaml"
ACTOR_DATASET="${ROOT_DIR}/metamon/rl/configs/datasets/stage1_b1_selected_strong_only.yaml"

mkdir -p "${METAMON_CACHE_DIR}" "${SAVE_DIR}"
if [ ! -f "${ANCHOR_CHECKPOINT}" ]; then
    echo "ERROR: epoch-${ANCHOR_EPOCH} anchor not found: ${ANCHOR_CHECKPOINT}" >&2
    exit 1
fi
if [ ! -d "${TAUROSV1_SAVE_DIR}/mediumg1online/ckpts" ]; then
    echo "ERROR: TaurosV1 architecture directory is missing: ${TAUROSV1_SAVE_DIR}/mediumg1online/ckpts" >&2
    echo "The old model supplies metadata only; its weights are not used as an anchor." >&2
    exit 1
fi
if [ -e "${SAVE_DIR}/${VALUE_RUN_NAME}" ] && [ ! -f "${VALUE_CHECKPOINT}" ]; then
    echo "ERROR: incomplete value-refit run exists: ${SAVE_DIR}/${VALUE_RUN_NAME}" >&2
    exit 1
fi
if [ -e "${SAVE_DIR}/${ACTOR_RUN_NAME}" ]; then
    echo "ERROR: actor-refinement run already exists: ${SAVE_DIR}/${ACTOR_RUN_NAME}" >&2
    exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required to prevent concurrent pipeline runs." >&2
    exit 1
fi
exec 9>"${SAVE_DIR}/.epoch51_anchored_pipeline.lock"
if ! flock -n 9; then
    echo "ERROR: another epoch-51 anchored pipeline is already running." >&2
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

echo "Preparing Gen1 OU human replays"
"${PYTHON_BIN}" -m metamon.data.download parsed-replays --formats gen1ou
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

if [ -f "${VALUE_CHECKPOINT}" ]; then
    echo "Reusing completed epoch-51 value refit: ${VALUE_CHECKPOINT}"
else
    echo "Starting epoch-51 value-head-only refit"
    echo "  Anchor: ${ANCHOR_CHECKPOINT}"
    echo "  Data: 70% strong-bot corpus, 30% human corpus"
    "${PYTHON_BIN}" -m metamon.rl.finetune \
        --run_name "${VALUE_RUN_NAME}" \
        --save_dir "${SAVE_DIR}" \
        --base_model TaurosV1 \
        --model_gin_config grouped_v2_medium_cuda.gin \
        --train_gin_config stage1_b1_imitation.gin \
        --dataset_config "${VALUE_DATASET}" \
        --epochs "${VALUE_EPOCHS}" \
        --steps_per_epoch "${VALUE_STEPS_PER_EPOCH}" \
        --batch_size_per_gpu "${BATCH_SIZE}" \
        --grad_accum 1 \
        --mixed_precision bf16 \
        --dloader_workers "${DLOADER_WORKERS}" \
        --async_env_mp_context spawn \
        --ckpt_interval "${CKPT_INTERVAL}" \
        --validation_fraction 0.10 \
        --validation_batches "${VALIDATION_BATCHES}" \
        --validation_seed 42 \
        --early_stopping_min_delta 0.001 \
        --stage1_b1_value_head_refit \
        --b1_anchor_checkpoint "${ANCHOR_CHECKPOINT}" \
        --b1_value_head_epochs "${VALUE_EPOCHS}" \
        --b1_value_head_steps_per_epoch "${VALUE_STEPS_PER_EPOCH}" \
        --b1_value_head_patience "${VALUE_PATIENCE}" \
        --eval_gens \
        "${LOG_ARGS[@]}"
fi

if [ ! -f "${VALUE_CHECKPOINT}" ]; then
    echo "ERROR: value refit did not publish ${VALUE_CHECKPOINT}" >&2
    exit 1
fi

echo "Starting selected-good-model actor refinement"
echo "  Immutable policy anchor: ${ANCHOR_CHECKPOINT}"
echo "  Value-fitted live model: ${VALUE_CHECKPOINT}"
echo "  Selected trajectories: ${SELECTED_STRONG_REPLAYS_DIR}"
exec "${PYTHON_BIN}" -m metamon.rl.finetune \
    --run_name "${ACTOR_RUN_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --base_model TaurosV1 \
    --model_gin_config grouped_v2_medium_cuda.gin \
    --train_gin_config stage1_b1_actor_bc.gin \
    --dataset_config "${ACTOR_DATASET}" \
    --epochs "${ACTOR_EPOCHS}" \
    --steps_per_epoch "${ACTOR_STEPS_PER_EPOCH}" \
    --batch_size_per_gpu "${BATCH_SIZE}" \
    --grad_accum "${ACTOR_GRAD_ACCUM}" \
    --mixed_precision bf16 \
    --dloader_workers "${DLOADER_WORKERS}" \
    --async_env_mp_context spawn \
    --ckpt_interval "${CKPT_INTERVAL}" \
    --validation_fraction 0.10 \
    --validation_batches "${VALIDATION_BATCHES}" \
    --validation_seed 42 \
    --early_stopping_min_delta 0.001 \
    --stage1_b1_actor_refinement \
    --b1_anchor_checkpoint "${ANCHOR_CHECKPOINT}" \
    --b1_phase2_checkpoint "${VALUE_CHECKPOINT}" \
    --b1_actor_epochs "${ACTOR_EPOCHS}" \
    --b1_actor_steps_per_epoch "${ACTOR_STEPS_PER_EPOCH}" \
    --b1_actor_patience "${ACTOR_PATIENCE}" \
    --eval_gens \
    "${LOG_ARGS[@]}"
