#!/usr/bin/env bash
set -euo pipefail

# One-shot collector benchmark. It records the two CPU-side ceilings first and
# can optionally run the real policy collector. No daemon or scheduler is used.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-${ROOT_DIR}/cache}"
export TAUROSV1_SAVE_DIR="${TAUROSV1_SAVE_DIR:-${ROOT_DIR}/checkpoints}"

TEAM_SET="${TEAM_SET:-b1_good_807,hl_05_26}"
LANES="${LANES:-64}"
SIM_WORKERS="${SIM_WORKERS:-4}"
ADAPTER_GAMES="${ADAPTER_GAMES:-2000}"
BINARY_GAMES="${BINARY_GAMES:-10000}"
BENCH_DIR="${BENCH_DIR:-${ROOT_DIR}/benchmarks/ataraxos-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${BENCH_DIR}"
"${ROOT_DIR}/tools/setup_pkmn_engine.sh"

echo "Ataraxos local benchmark"
echo "Output directory: ${BENCH_DIR}"
echo "Team set: ${TEAM_SET}"
echo "Lanes / simulator workers: ${LANES} / ${SIM_WORKERS}"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/benchmark_native_random_selfplay.py" \
    --games "${ADAPTER_GAMES}" \
    --lanes "${LANES}" \
    --workers "${SIM_WORKERS}" \
    --team-set "${TEAM_SET}" \
    --progress-every 500 \
    2>&1 | tee "${BENCH_DIR}/model-facing-adapter.log"

"${PYTHON_BIN}" "${ROOT_DIR}/tools/benchmark_native_binary_selfplay.py" \
    --games "${BINARY_GAMES}" \
    --lanes "${LANES}" \
    --workers "${SIM_WORKERS}" \
    --team-set "${TEAM_SET}" \
    --progress-every 2500 \
    2>&1 | tee "${BENCH_DIR}/binary-boundary.log"

if [ -n "${CHECKPOINT:-}" ]; then
    POLICY_GAMES="${POLICY_GAMES:-64}"
    POLICY_LANES="${POLICY_LANES:-8}"
    DEVICE="${DEVICE:-cpu}"
    MIXED_PRECISION="${MIXED_PRECISION:-no}"
    COMMAND=(
        "${PYTHON_BIN}" -m metamon.rl.ataraxos
        --mode collect
        --checkpoint "${CHECKPOINT}"
        --output-dir "${BENCH_DIR}/policy-collector"
        --games "${POLICY_GAMES}"
        --lanes "${POLICY_LANES}"
        --workers "${POLICY_SIM_WORKERS:-1}"
        --writer-workers "${WRITER_WORKERS:-2}"
        --team-set "${TEAM_SET}"
        --device "${DEVICE}"
        --mixed-precision "${MIXED_PRECISION}"
        --temperature "${TEMPERATURE:-0}"
        --profile
    )
    if [ -n "${ARCHITECTURE_SAVE_ROOT:-}" ]; then
        COMMAND+=(--architecture-save-root "${ARCHITECTURE_SAVE_ROOT}")
    fi
    if [ -n "${ARCHITECTURE_RUN_NAME:-}" ]; then
        COMMAND+=(--architecture-run-name "${ARCHITECTURE_RUN_NAME}")
    fi
    "${COMMAND[@]}" 2>&1 | tee "${BENCH_DIR}/policy-collector.log"
else
    echo "CHECKPOINT is unset; skipping real-policy collection benchmark."
fi

echo "Benchmark complete: ${BENCH_DIR}"
