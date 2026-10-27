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
# which has failed during local smoke tests. FlashAttention remains fused even
# when Dynamo is disabled. Opt back in only after a machine-specific smoke test.
if [ "${ENABLE_TORCH_COMPILE:-0}" != "1" ]; then
    export TORCHDYNAMO_DISABLE=1
fi

DATASET_CONFIG="${DATASET_CONFIG:-${ROOT_DIR}/metamon/rl/configs/datasets/stage1_b1_strong.yaml}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints}"
RUN_NAME="${RUN_NAME:-stage1_b1}"
EPOCHS="${EPOCHS:-1100}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
B1_VALUE_HEAD_STEPS_PER_EPOCH="${B1_VALUE_HEAD_STEPS_PER_EPOCH:-1000}"
B1_VALUE_BACKBONE_STEPS_PER_EPOCH="${B1_VALUE_BACKBONE_STEPS_PER_EPOCH:-1000}"
BATCH_SIZE="${BATCH_SIZE:-12}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
DLOADER_WORKERS="${DLOADER_WORKERS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PAC_TAUROS_TRAJECTORIES="${PAC_TAUROS_TRAJECTORIES:-250000}"
CKPT_INTERVAL="${CKPT_INTERVAL:-10}"

mkdir -p "${METAMON_CACHE_DIR}" "${SAVE_DIR}"

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

# The launcher can be run directly from a source checkout without installing
# metamon itself. In that case, ensure its required poke-env distribution is
# installed in the *same* interpreter used to start training. Importing the
# source package alone is not enough: metamon checks distribution metadata.
if ! "${PYTHON_BIN}" -c 'from importlib.metadata import version; version("poke-env"); version("amago")' >/dev/null 2>&1; then
    echo "Installing Metamon and its pinned Python dependencies"
    "${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}"
    if ! "${PYTHON_BIN}" -c 'from importlib.metadata import version; version("poke-env"); version("amago")' >/dev/null 2>&1; then
        echo "ERROR: poke-env or AMAGO is still unavailable for ${PYTHON_BIN}." >&2
        exit 1
    fi
fi

if ! "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "ERROR: CUDA is unavailable. Set PYTHON_BIN to a CUDA-enabled Python."
    exit 1
fi

# AMAGO v3.4's FlashAttention adapter explicitly converts QKV to BF16 even if
# the outer autocast mode is FP16, so reject GPUs that cannot execute BF16.
if ! "${PYTHON_BIN}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_bf16_supported() else 1)'; then
    echo "ERROR: AMAGO v3.4 FlashAttention requires a BF16-capable CUDA GPU."
    exit 1
fi

if ! "${PYTHON_BIN}" -c 'import flash_attn' >/dev/null 2>&1; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "ERROR: FlashAttention is missing and nvcc is unavailable. Use a CUDA development image."
        exit 1
    fi
    echo "Installing ${FLASH_ATTN_SPEC:-flash-attn}; CUDA kernels may compile once"
    "${PYTHON_BIN}" -m pip install -U packaging ninja
    MAX_JOBS="${MAX_JOBS:-4}" "${PYTHON_BIN}" -m pip install --no-build-isolation \
        "${FLASH_ATTN_SPEC:-flash-attn}"
fi

"${PYTHON_BIN}" - <<'PY'
import torch
from amago.nets.transformer import FlashAttention

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable during the FlashAttention smoke test")
qkv = torch.randn(
    1, 64, 3, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
)
attention = FlashAttention(causal=True, dropout=0.0, window_size=(32, 0)).cuda()
attention.train()
out = attention(qkv)
out.float().square().mean().backward()
torch.cuda.synchronize()
print("AMAGO FlashAttention training smoke test succeeded")
PY

if [ ! -f "${TAUROSV1_SAVE_DIR}/mediumg1online/ckpts/policy_weights/policy_epoch_475.pt" ]; then
    "${PYTHON_BIN}" "${ROOT_DIR}/tools/assemble_epoch475.py" \
        --repo-root "${ROOT_DIR}" \
        --output-root "${TAUROSV1_SAVE_DIR}"
fi

echo "Preparing bounded pac-tauros subset first (${PAC_TAUROS_TRAJECTORIES} trajectories)"
"${PYTHON_BIN}" -u "${ROOT_DIR}/tools/prepare_pac_tauros_subset.py" \
    --cache-dir "${METAMON_CACHE_DIR}" \
    --max-trajectories "${PAC_TAUROS_TRAJECTORIES}"

echo "Preparing Gen1 OU human replay dataset"
"${PYTHON_BIN}" -m metamon.data.download parsed-replays --formats gen1ou

if [ ! -d "${GOOD_REPLAYS_DIR}/gen1ou" ] && [ ! -f "${GOOD_REPLAYS_DIR}/gen1ou.tar" ]; then
    "${ROOT_DIR}/tools/prepare_strong_replays.sh"
fi

LOG_ARG=""
if [ -n "${WANDB_API_KEY:-}" ]; then
    LOG_ARG="--log"
fi

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
    --ckpt_interval "${CKPT_INTERVAL}" \
    --validation_fraction 0.10 \
    --validation_batches 50 \
    --validation_seed 42 \
    --early_stopping_patience 5 \
    --early_stopping_min_delta 0.001 \
    --stage1_b1_pipeline \
    --b1_value_head_epochs 100 \
    --b1_value_backbone_epochs 1000 \
    --b1_value_head_steps_per_epoch "${B1_VALUE_HEAD_STEPS_PER_EPOCH}" \
    --b1_value_backbone_steps_per_epoch "${B1_VALUE_BACKBONE_STEPS_PER_EPOCH}" \
    --b1_value_head_patience 3 \
    --b1_value_backbone_patience 5 \
    --eval_gens \
    ${LOG_ARG}
