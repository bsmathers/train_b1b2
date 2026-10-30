#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-${ROOT_DIR}/cache}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints}"
CHECKPOINT="${CHECKPOINT:-${SAVE_DIR}/stage1_b1_epoch51_selected_actor/ckpts/policy_weights/policy_actor_bc_final.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${SAVE_DIR}/ataraxos}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/ataraxos.log}"
MODE="${MODE:-collect}"
ROUND="${ROUND:-0}"
ROUNDS="${ROUNDS:-1}"
LANES="${LANES:-64}"
COLLECTOR_SHARDS="${COLLECTOR_SHARDS:-1}"
GAMES="${GAMES:-$((2 * LANES * COLLECTOR_SHARDS))}"
TARGET_DECISIONS="${TARGET_DECISIONS:-80000}"
SIM_WORKERS="${SIM_WORKERS:-4}"
WRITER_WORKERS="${WRITER_WORKERS:-2}"
VALIDATION_WORKERS="${VALIDATION_WORKERS:-0}"
TEAM_SET="${TEAM_SET:-b1_good_807,hl_05_26}"
SIDE="${SIDE:-alternate}"
TEMPERATURE="${TEMPERATURE:-1.0}"
DEVICE="${DEVICE:-cuda:0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export PYTHON_BIN DEVICE MIXED_PRECISION
PROFILE="${PROFILE:-0}"
JOINT_INFERENCE="${JOINT_INFERENCE:-1}"
TURN_LIMIT="${TURN_LIMIT:-200}"
SEED="${SEED:-20261027}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-20}"
MAX_VECTOR_STEPS="${MAX_VECTOR_STEPS:-1000000}"
EVALUATION_INTERVAL="${EVALUATION_INTERVAL:-10}"

if [ ! -x "${PYTHON_BIN}" ] && ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: behavior checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi
if [ "${MODE}" = "pipeline" ]; then
    UPDATE_COMMAND="${UPDATE_COMMAND:-${ROOT_DIR}/tools/update_ataraxos_once.sh}"
    EVALUATION_COMMAND="${EVALUATION_COMMAND:-${ROOT_DIR}/tools/evaluate_ataraxos_candidate.sh}"
fi

mkdir -p "${OUTPUT_DIR}"
"${ROOT_DIR}/tools/setup_pkmn_engine.sh"

echo "Starting native Ataraxos infrastructure"
echo "  Mode: ${MODE}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Round: ${ROUND}; rounds: ${ROUNDS}; target decisions: ${TARGET_DECISIONS} (0 = exact games quota ${GAMES})"
echo "  Collector: ${COLLECTOR_SHARDS} processes x ${LANES} lanes, ${SIM_WORKERS} simulator workers/process, ${WRITER_WORKERS} writer workers/process"
echo "  Device: ${DEVICE}; mixed precision: ${MIXED_PRECISION}"
echo "  Host logical CPUs: $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unknown)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

COMMAND=(
    "${PYTHON_BIN}" -m metamon.rl.ataraxos
    --mode "${MODE}"
    --checkpoint "${CHECKPOINT}"
    --output-dir "${OUTPUT_DIR}"
    --round "${ROUND}"
    --rounds "${ROUNDS}"
    --games "${GAMES}"
    --target-decisions "${TARGET_DECISIONS}"
    --lanes "${LANES}"
    --collector-shards "${COLLECTOR_SHARDS}"
    --workers "${SIM_WORKERS}"
    --writer-workers "${WRITER_WORKERS}"
    --validation-workers "${VALIDATION_WORKERS}"
    --team-set "${TEAM_SET}"
    --side "${SIDE}"
    --temperature "${TEMPERATURE}"
    --turn-limit "${TURN_LIMIT}"
    --seed "${SEED}"
    --progress-interval "${PROGRESS_INTERVAL}"
    --max-vector-steps "${MAX_VECTOR_STEPS}"
    --evaluation-interval "${EVALUATION_INTERVAL}"
    --device "${DEVICE}"
    --mixed-precision "${MIXED_PRECISION}"
)
if [ -n "${ARCHITECTURE_SAVE_ROOT:-}" ]; then
    COMMAND+=(--architecture-save-root "${ARCHITECTURE_SAVE_ROOT}")
fi
if [ -n "${ARCHITECTURE_RUN_NAME:-}" ]; then
    COMMAND+=(--architecture-run-name "${ARCHITECTURE_RUN_NAME}")
fi
if [ "${MODE}" = "pipeline" ]; then
    COMMAND+=(--update-command "${UPDATE_COMMAND}")
    COMMAND+=(--evaluation-command "${EVALUATION_COMMAND}")
fi
if [ "${PROFILE}" = "1" ]; then
    COMMAND+=(--profile)
fi
if [ "${JOINT_INFERENCE}" = "1" ]; then
    COMMAND+=(--joint-inference)
else
    COMMAND+=(--no-joint-inference)
fi

"${COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
