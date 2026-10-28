#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SAVE_DIR="${SAVE_DIR:-/workspace/b1-data/checkpoints}"
CACHE_DIR="${METAMON_CACHE_DIR:-/workspace/b1-data/cache}"
RUN_NAME="${RUN_NAME:-stage1_b1_epoch51_selected_actor}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/b1-data/weight_average_screen_corrected_v2}"
BATTLES="${BATTLES:-500}"
LANES="${LANES:-32}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-20261028}"
WINDOWS="${WINDOWS:-2 4 8 16}"
SIGMA_POWER_ITERATIONS="${SIGMA_POWER_ITERATIONS:-32}"

cd "${ROOT_DIR}"
"${ROOT_DIR}/tools/setup_pkmn_engine.sh"

read -r -a WINDOW_ARGS <<< "${WINDOWS}"
EXTRA_ARGS=()
if [ -n "${END_EPOCH:-}" ]; then
    EXTRA_ARGS+=(--end-epoch "${END_EPOCH}")
fi

exec "${PYTHON_BIN}" "${ROOT_DIR}/tools/test_weight_averages.py" \
    --python "${PYTHON_BIN}" \
    --save-dir "${SAVE_DIR}" \
    --run-name "${RUN_NAME}" \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --windows "${WINDOW_ARGS[@]}" \
    --sigma-power-iterations "${SIGMA_POWER_ITERATIONS}" \
    --battles "${BATTLES}" \
    --lanes "${LANES}" \
    --workers "${WORKERS}" \
    --seed "${SEED}" \
    "${EXTRA_ARGS[@]}"
