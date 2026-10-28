#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -x /workspace/b1-venv/bin/python ]; then
    DEFAULT_PYTHON=/workspace/b1-venv/bin/python
else
    DEFAULT_PYTHON=python3
fi

PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
SAVE_DIR="${SAVE_DIR:-/workspace/b1-data/checkpoints}"
CACHE_DIR="${CACHE_DIR:-/workspace/b1-data/cache}"
RUN_NAME="${RUN_NAME:-stage1_b1_epoch51_awr_v3}"
POLICY_DIR="${SAVE_DIR}/${RUN_NAME}/ckpts/policy_weights"
BATTLES="${BATTLES:-500}"
LANES="${LANES:-32}"
SIM_WORKERS="${SIM_WORKERS:-4}"
SEED="${SEED:-20261028}"
TIMEOUT="${TIMEOUT:-7200}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/b1-data/tournaments/${RUN_NAME}_best_vs_taurosv0_${BATTLES}}"

# A clean early-stop publishes `final`. Ctrl-C intentionally does not. In an
# interrupted run, `best` is the atomically saved validation winner and is the
# checkpoint we want—not the newest chronological policy_epoch_N snapshot.
if [ -f "${POLICY_DIR}/policy_actor_awr_final.pt" ]; then
    CHECKPOINT="${POLICY_DIR}/policy_actor_awr_final.pt"
elif [ -f "${POLICY_DIR}/policy_actor_awr_best.pt" ]; then
    CHECKPOINT="${POLICY_DIR}/policy_actor_awr_best.pt"
else
    echo "ERROR: no completed AWR validation-best checkpoint found in ${POLICY_DIR}" >&2
    exit 1
fi

if [ ! -x "${PYTHON_BIN}" ] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [ ! -d "${SAVE_DIR}/mediumg1online/ckpts" ]; then
    echo "ERROR: Epoch475 architecture directory is missing: ${SAVE_DIR}/mediumg1online/ckpts" >&2
    exit 1
fi

export METAMON_CACHE_DIR="${CACHE_DIR}"
export B1_SAVE_DIR="${SAVE_DIR}"
export B1_RUN_NAME="${RUN_NAME}"
export TAUROSV1_SAVE_DIR="${SAVE_DIR}"
export TORCHDYNAMO_DISABLE=1

cd "${ROOT_DIR}"
"${ROOT_DIR}/tools/setup_pkmn_engine.sh"

echo "Starting exact native AWR-vs-TaurosV0 tournament"
echo "  Candidate: ${CHECKPOINT}"
echo "  Games: ${BATTLES}; action selection: argmax"
echo "  Teams: b1_good_807 for both players"
echo "  Output: ${OUTPUT_DIR}"
echo "  Live game results: ${OUTPUT_DIR}/games.log"

exec "${PYTHON_BIN}" -u "${ROOT_DIR}/tools/run_b1_vs_taurosv0_500.py" \
    --checkpoint-path "${CHECKPOINT}" \
    --battles "${BATTLES}" \
    --candidate-team b1_good_807 \
    --opponent-team b1_good_807 \
    --save-dir "${SAVE_DIR}" \
    --run-name "${RUN_NAME}" \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --lanes "${LANES}" \
    --workers "${SIM_WORKERS}" \
    --seed "${SEED}" \
    --temperature 0 \
    --timeout "${TIMEOUT}"
