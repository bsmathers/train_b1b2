#!/usr/bin/env bash
set -euo pipefail

# Build a transfer-sized archive of the policy-only checkpoints that are most
# useful for provenance, tournaments, and later iterate averaging. Optimizer
# and scheduler states are intentionally excluded.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAVE_DIR="${SAVE_DIR:-/workspace/b1-data/checkpoints}"
BACKUP_ROOT="${BACKUP_ROOT:-/workspace/b1-data/model-weight-backups}"
AVERAGE_DIR="${AVERAGE_DIR:-/workspace/b1-data/weight_average_screen_corrected_v2/averaged_checkpoints}"
MAX_INPUT_BYTES="${MAX_INPUT_BYTES:-9750000000}"
STAMP="${BACKUP_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
PACKAGE_NAME="actor-policy-weights-${STAMP}"
STAGING_DIR="${BACKUP_ROOT}/${PACKAGE_NAME}"
MANIFEST="${STAGING_DIR}/MANIFEST.txt"

if [ -e "${STAGING_DIR}" ]; then
    echo "ERROR: backup staging directory already exists: ${STAGING_DIR}" >&2
    exit 1
fi
mkdir -p "${STAGING_DIR}/weights"

if command -v sha256sum >/dev/null 2>&1; then
    checksum() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    checksum() { shasum -a 256 "$1" | awk '{print $1}'; }
else
    echo "ERROR: sha256sum or shasum is required." >&2
    exit 1
fi

file_bytes() {
    if stat -c '%s' "$1" >/dev/null 2>&1; then
        stat -c '%s' "$1"
    else
        stat -f '%z' "$1"
    fi
}

total_bytes=0
file_count=0

{
    echo "Metamon actor policy weight backup"
    echo "Created UTC: ${STAMP}"
    echo "Repository commit: $(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "Source checkpoint root: ${SAVE_DIR}"
    echo "Maximum uncompressed policy bytes: ${MAX_INPUT_BYTES}"
    echo
    echo "Each entry: archive path | bytes | SHA-256 | source path"
} >"${MANIFEST}"

copy_weight() {
    local source="$1"
    local archive_path="$2"
    local bytes destination digest
    [ -f "${source}" ] || return 0
    destination="${STAGING_DIR}/weights/${archive_path}"
    if [ -e "${destination}" ]; then
        return 0
    fi
    bytes="$(file_bytes "${source}")"
    if (( total_bytes + bytes > MAX_INPUT_BYTES )); then
        echo "ERROR: selected policy weights exceed the ${MAX_INPUT_BYTES}-byte budget." >&2
        echo "Last candidate: ${source}" >&2
        echo "Already selected: ${total_bytes} bytes" >&2
        exit 1
    fi
    mkdir -p "$(dirname "${destination}")"
    cp -p "${source}" "${destination}"
    digest="$(checksum "${destination}")"
    printf '%s | %s | %s | %s\n' \
        "weights/${archive_path}" "${bytes}" "${digest}" "${source}" >>"${MANIFEST}"
    total_bytes=$((total_bytes + bytes))
    file_count=$((file_count + 1))
    echo "Collected ${archive_path}"
}

copy_run_file() {
    local run_name="$1"
    local filename="$2"
    copy_weight \
        "${SAVE_DIR}/${run_name}/ckpts/policy_weights/${filename}" \
        "${run_name}/${filename}"
}

copy_final_or_best() {
    local run_name="$1"
    local final_name="$2"
    local best_name="$3"
    if [ -f "${SAVE_DIR}/${run_name}/ckpts/policy_weights/${final_name}" ]; then
        copy_run_file "${run_name}" "${final_name}"
    else
        copy_run_file "${run_name}" "${best_name}"
    fi
}

echo "Collecting lineage and phase-boundary policies"
if [ -n "${BASE_EPOCH475_CHECKPOINT:-}" ]; then
    copy_weight "${BASE_EPOCH475_CHECKPOINT}" "mediumg1online/policy_epoch_475.pt"
else
    copy_run_file "mediumg1online" "policy_epoch_475.pt"
    copy_weight \
        "${ROOT_DIR}/checkpoints/mediumg1online/ckpts/policy_weights/policy_epoch_475.pt" \
        "mediumg1online/policy_epoch_475.pt"
fi
copy_run_file "stage1_b1" "policy_phase2_final.pt"
copy_run_file "stage1_b1_epoch51_value_refit" "policy_value_head_refit_final.pt"

echo "Collecting original full-corpus actor-BC milestones"
for epoch in 0 1 2 4 8 16 32 48 51; do
    copy_run_file "stage1_b1_actor_bc" "policy_epoch_${epoch}.pt"
done
copy_final_or_best \
    "stage1_b1_actor_bc" "policy_actor_bc_final.pt" "policy_actor_bc_best.pt"

echo "Collecting every selected-strong actor checkpoint"
selected_dir="${SAVE_DIR}/stage1_b1_epoch51_selected_actor/ckpts/policy_weights"
if [ -d "${selected_dir}" ]; then
    while IFS= read -r checkpoint; do
        copy_weight "${checkpoint}" \
            "stage1_b1_epoch51_selected_actor/$(basename "${checkpoint}")"
    done < <(find "${selected_dir}" -maxdepth 1 -type f -name 'policy_epoch_*.pt' | sort -V)
fi
copy_final_or_best \
    "stage1_b1_epoch51_selected_actor" \
    "policy_actor_bc_final.pt" "policy_actor_bc_best.pt"

echo "Collecting dense anchored-AWR averaging window"
for epoch in $(seq 14 30); do
    copy_run_file "stage1_b1_epoch51_awr_v3" "policy_epoch_${epoch}.pt"
done
for epoch in $(seq 32 2 46); do
    copy_run_file "stage1_b1_epoch51_awr_v3" "policy_epoch_${epoch}.pt"
done
copy_final_or_best \
    "stage1_b1_epoch51_awr_v3" \
    "policy_actor_awr_final.pt" "policy_actor_awr_best.pt"

echo "Collecting unanchored-AWR update milestones when available"
for epoch in 0 1 3 7 15 23 31 39; do
    copy_run_file "stage1_b1_epoch51_awr_unanchored_e30" "policy_epoch_${epoch}.pt"
done
copy_final_or_best \
    "stage1_b1_epoch51_awr_unanchored_e30" \
    "policy_actor_awr_final.pt" "policy_actor_awr_best.pt"

echo "Collecting corrected weight-average candidates"
if [ -d "${AVERAGE_DIR}" ]; then
    while IFS= read -r checkpoint; do
        copy_weight "${checkpoint}" "weight_averages/$(basename "${checkpoint}")"
    done < <(find "${AVERAGE_DIR}" -maxdepth 1 -type f -name '*.pt' | sort -V)
fi

{
    echo
    echo "Policy files: ${file_count}"
    echo "Uncompressed policy bytes: ${total_bytes}"
    echo "Excluded: optimizer states, scheduler states, replay data, and duplicate latest aliases"
} >>"${MANIFEST}"

if (( file_count == 0 )); then
    echo "ERROR: no policy weights were found below ${SAVE_DIR}." >&2
    exit 1
fi

echo "Compressing ${file_count} policy files (${total_bytes} bytes)"
if command -v zstd >/dev/null 2>&1; then
    ARCHIVE="${BACKUP_ROOT}/${PACKAGE_NAME}.tar.zst"
    tar -C "${BACKUP_ROOT}" -cf - "${PACKAGE_NAME}" | zstd -T0 -3 -o "${ARCHIVE}"
elif command -v pigz >/dev/null 2>&1; then
    ARCHIVE="${BACKUP_ROOT}/${PACKAGE_NAME}.tar.gz"
    tar -C "${BACKUP_ROOT}" -cf - "${PACKAGE_NAME}" | pigz -1 >"${ARCHIVE}"
else
    ARCHIVE="${BACKUP_ROOT}/${PACKAGE_NAME}.tar.gz"
    tar -C "${BACKUP_ROOT}" -czf "${ARCHIVE}" "${PACKAGE_NAME}"
fi

archive_bytes="$(file_bytes "${ARCHIVE}")"
echo "Archive ready: ${ARCHIVE}"
echo "Archive bytes: ${archive_bytes}"
echo "Manifest: ${MANIFEST}"
if (( archive_bytes > 10000000000 )); then
    echo "ERROR: archive exceeds the 10 GB transfer limit; it has been left in place for inspection." >&2
    exit 1
fi
