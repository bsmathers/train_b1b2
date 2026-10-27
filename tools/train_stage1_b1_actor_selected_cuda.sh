#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-${ROOT_DIR}/cache}"
export GOOD_REPLAYS_DIR="${GOOD_REPLAYS_DIR:-${METAMON_CACHE_DIR}/strong_model_replays_parsed}"
export SELECTED_STRONG_REPLAYS_DIR="${SELECTED_STRONG_REPLAYS_DIR:-${METAMON_CACHE_DIR}/strong_model_replays_selected}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints}"
export TAUROSV1_SAVE_DIR="${TAUROSV1_SAVE_DIR:-${SAVE_DIR}}"
SOURCE_ACTOR_RUN_NAME="${SOURCE_ACTOR_RUN_NAME:-stage1_b1_actor_bc}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${SAVE_DIR}/${SOURCE_ACTOR_RUN_NAME}/ckpts/policy_weights/policy_actor_bc_best.pt}"

if [ ! -d "${GOOD_REPLAYS_DIR}/gen1ou" ]; then
    "${ROOT_DIR}/tools/prepare_strong_replays.sh"
fi
"${PYTHON_BIN}" "${ROOT_DIR}/tools/prepare_selected_strong_replays.py" \
    --source "${GOOD_REPLAYS_DIR}" \
    --output "${SELECTED_STRONG_REPLAYS_DIR}"

if [ ! -f "${SOURCE_CHECKPOINT}" ]; then
    echo "ERROR: source actor checkpoint not found: ${SOURCE_CHECKPOINT}" >&2
    echo "Finish the first actor refinement or set SOURCE_CHECKPOINT explicitly." >&2
    exit 1
fi

export SAVE_DIR
export RUN_NAME="${RUN_NAME:-stage1_b1_actor_selected_bc}"
export PHASE2_CHECKPOINT="${SOURCE_CHECKPOINT}"
export DATASET_CONFIG="${ROOT_DIR}/metamon/rl/configs/datasets/stage1_b1_selected_strong_only.yaml"

echo "Starting selected-policy actor refinement"
echo "  Source actor: ${SOURCE_CHECKPOINT}"
echo "  Selected trajectories: ${SELECTED_STRONG_REPLAYS_DIR}"
echo "  New run: ${SAVE_DIR}/${RUN_NAME}"

exec "${ROOT_DIR}/tools/train_stage1_b1_actor_refine_cuda.sh"
