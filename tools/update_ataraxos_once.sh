#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ "${DEVICE:-cuda:0}" = "cpu" ]; then
    # The portable attention fallback is already eager-safe; avoiding Dynamo
    # here also prevents long compile startup in the local smoke test.
    export TORCHDYNAMO_DISABLE=1
fi

exec "${PYTHON_BIN}" -m metamon.rl.ataraxos_update \
    --device "${DEVICE:-cuda:0}" \
    --mixed-precision "${MIXED_PRECISION:-bf16}" \
    --update-epochs "${ATARAXOS_UPDATE_EPOCHS:-2}" \
    --batch-size "${ATARAXOS_BATCH_SIZE:-4}" \
    --learning-rate "${ATARAXOS_LR:-3e-6}" \
    --clip-ratio "${ATARAXOS_CLIP_RATIO:-0.10}" \
    --value-coef "${ATARAXOS_VALUE_COEF:-0.5}" \
    --entropy-coef "${ATARAXOS_ENTROPY_COEF:-0.001}" \
    --kl-coef "${ATARAXOS_KL_COEF:-0.05}" \
    --max-grad-norm "${ATARAXOS_MAX_GRAD_NORM:-0.5}" \
    --max-initial-kl "${ATARAXOS_MAX_INITIAL_KL:-0.02}"
