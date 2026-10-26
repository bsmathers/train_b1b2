#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export TAUROSV1_SAVE_DIR="${TAUROSV1_SAVE_DIR:-${ROOT_DIR}/checkpoints}"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-${ROOT_DIR}/cache}"
export GOOD_REPLAYS_DIR="${GOOD_REPLAYS_DIR:-${ROOT_DIR}/cache/strong_model_replays_parsed}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The installed AMAGO version decorates the transformer path with torch.compile,
# which has failed during local smoke tests. Eager execution still dispatches
# PyTorch's fused CUDA SDPA kernels. Opt back in only after a machine-specific
# smoke test.
if [ "${ENABLE_TORCH_COMPILE:-0}" != "1" ]; then
    export TORCHDYNAMO_DISABLE=1
fi

DATASET_CONFIG="${DATASET_CONFIG:-${ROOT_DIR}/metamon/rl/configs/datasets/stage1_b1_strong.yaml}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints}"
RUN_NAME="${RUN_NAME:-stage1_b1}"
EPOCHS="${EPOCHS:-30}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DLOADER_WORKERS="${DLOADER_WORKERS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

mkdir -p "${METAMON_CACHE_DIR}" "${SAVE_DIR}"

# Source checkouts do not install dependencies automatically. Install the
# project's declared dependencies together rather than chasing import errors
# one package at a time. Use the exact interpreter used for training.
if ! "${PYTHON_BIN}" -c 'import importlib.util; from importlib.metadata import version; version("poke-env"); assert importlib.util.find_spec("gin") and importlib.util.find_spec("amago")' >/dev/null 2>&1; then
    echo "Stage-1 dependencies are missing for ${PYTHON_BIN}; installing this project and its dependencies..."
    "${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}"
fi
# Fail early if installation did not make the required dependencies importable.
"${PYTHON_BIN}" -c 'import gin, amago; from importlib.metadata import version; version("poke-env")' || {
    echo "ERROR: Stage-1 dependencies are unavailable for ${PYTHON_BIN} after installation." >&2
    exit 1
}

if ! "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    if [ "${ALLOW_CPU:-0}" != "1" ]; then
        echo "ERROR: CUDA is unavailable. Set PYTHON_BIN to a CUDA-enabled Python."
        echo "Set ALLOW_CPU=1 only for a short smoke test; full CPU training is impractical."
        exit 1
    fi
    echo "WARNING: CUDA unavailable; running on CPU because ALLOW_CPU=1"
    MIXED_PRECISION="no"
elif [ "${MIXED_PRECISION}" = "bf16" ] && ! "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_bf16_supported() else 1)'; then
    echo "ERROR: this CUDA GPU does not support BF16; set MIXED_PRECISION=fp16."
    exit 1
fi

if [ ! -f "${TAUROSV1_SAVE_DIR}/mediumg1online/ckpts/policy_weights/policy_epoch_475.pt" ]; then
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/assemble_epoch475.py" --repo-root "${ROOT_DIR}"
fi

if [ ! -d "${GOOD_REPLAYS_DIR}/gen1ou" ] && [ ! -f "${GOOD_REPLAYS_DIR}/gen1ou.tar" ]; then
    "${ROOT_DIR}/tools/prepare_strong_replays.sh"
fi

LOG_ARG=""
if [ -n "${WANDB_API_KEY:-}" ]; then
    LOG_ARG="--log"
fi

cd "${ROOT_DIR}"
exec "${PYTHON_BIN}" -m metamon.rl.finetune \
    --run_name "${RUN_NAME}" \
    --save_dir "${SAVE_DIR}" \
    --base_model TaurosV1 \
    --base_checkpoint 475 \
    --model_gin_config grouped_v2_medium_cuda.gin \
    --train_gin_config stage1_b1_imitation.gin \
    --dataset_config "${DATASET_CONFIG}" \
    --epochs "${EPOCHS}" \
    --steps_per_epoch "${STEPS_PER_EPOCH}" \
    --batch_size_per_gpu "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --mixed_precision "${MIXED_PRECISION}" \
    --dloader_workers "${DLOADER_WORKERS}" \
    --async_env_mp_context spawn \
    --ckpt_interval 1 \
    --validation_fraction 0.10 \
    --validation_batches 50 \
    --validation_seed 42 \
    --early_stopping_patience 5 \
    --early_stopping_min_delta 0.001 \
    --eval_gens \
    ${LOG_ARG}
