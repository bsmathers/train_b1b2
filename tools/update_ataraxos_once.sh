#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ "${DEVICE:-cuda:0}" = "cpu" ]; then
    # The portable attention fallback is already eager-safe; avoiding Dynamo
    # here also prevents long compile startup in the local smoke test.
    export TORCHDYNAMO_DISABLE=1
fi

OVERRIDES=()
if [ -n "${ATARAXOS_LR:-}" ]; then
    OVERRIDES+=(--learning-rate "${ATARAXOS_LR}")
fi
if [ -n "${ATARAXOS_MAGNET_COEF:-}" ]; then
    OVERRIDES+=(--magnet-coef "${ATARAXOS_MAGNET_COEF}")
fi

exec "${PYTHON_BIN}" -m metamon.rl.ataraxos_update \
    --device "${DEVICE:-cuda:0}" \
    --mixed-precision "${MIXED_PRECISION:-bf16}" \
    --update-epochs "${ATARAXOS_UPDATE_EPOCHS:-1}" \
    --batch-size "${ATARAXOS_BATCH_SIZE:-4}" \
    --lr-coefficient "${ATARAXOS_LR_COEFFICIENT:-0.5}" \
    --lr-decay "${ATARAXOS_LR_DECAY:-1.1}" \
    --lr-floor "${ATARAXOS_LR_FLOOR:-5e-6}" \
    --lr-ceiling "${ATARAXOS_LR_CEILING:-1e-4}" \
    --magnet-coefficient "${ATARAXOS_MAGNET_COEFFICIENT:-0.05}" \
    --magnet-decay "${ATARAXOS_MAGNET_DECAY:-0.3}" \
    --magnet-floor "${ATARAXOS_MAGNET_FLOOR:-0.001}" \
    --magnet-ceiling "${ATARAXOS_MAGNET_CEILING:-0.1}" \
    --weight-decay "${ATARAXOS_WEIGHT_DECAY:-0}" \
    --clip-ratio "${ATARAXOS_CLIP_RATIO:-0.20}" \
    --value-coef "${ATARAXOS_VALUE_COEF:-1.0}" \
    --kl-coef "${ATARAXOS_KL_COEF:-0.10}" \
    --gae-lambda "${ATARAXOS_GAE_LAMBDA:-0.5}" \
    --td-lambda "${ATARAXOS_TD_LAMBDA:-0.8}" \
    --ema-decay "${ATARAXOS_EMA_DECAY:-0.999}" \
    --max-grad-norm "${ATARAXOS_MAX_GRAD_NORM:-0.267}" \
    --max-initial-kl "${ATARAXOS_MAX_INITIAL_KL:-0.02}" \
    "${OVERRIDES[@]}"
