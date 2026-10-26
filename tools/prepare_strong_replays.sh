#!/usr/bin/env bash
set -Eeuo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GOOD_REPLAYS_DIR="${GOOD_REPLAYS_DIR:-${ROOT_DIR}/cache/strong_model_replays_parsed}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
STRONG_REPLAYS_REVISION="${STRONG_REPLAYS_REVISION:-623ce1886eb93a2a893638ed413443dd7b4123aa}"
STRONG_REPLAYS_SHA256="${STRONG_REPLAYS_SHA256:-b43668b1a4342b2e27125ee5d52fb3ec14d7b8d3342d68ae92651d90348139e4}"
STRONG_REPLAYS_URL="${STRONG_REPLAYS_URL:-https://huggingface.co/datasets/bsmathers/scraped_replays_parsed/resolve/${STRONG_REPLAYS_REVISION}/gen1ou.zip?download=true}"
STRONG_REPLAYS_ARCHIVE="${STRONG_REPLAYS_ARCHIVE:-${GOOD_REPLAYS_DIR%/}.gen1ou.zip}"

if [[ -d "${GOOD_REPLAYS_DIR}/gen1ou" ]] \
    && find "${GOOD_REPLAYS_DIR}/gen1ou" -maxdepth 1 -type f \
      \( -name '*.json' -o -name '*.json.lz4' \) -print -quit | grep -q .; then
  rm -f "${STRONG_REPLAYS_ARCHIVE}" "${STRONG_REPLAYS_ARCHIVE}.partial"
  log "Reusing strong-model trajectories in ${GOOD_REPLAYS_DIR}/gen1ou"
  exit 0
fi

command -v curl >/dev/null || die "curl is required to download strong-model replays"
command -v "${PYTHON_BIN}" >/dev/null || die "Python is required to unpack strong-model replays"
mkdir -p "$(dirname "${GOOD_REPLAYS_DIR}")" "$(dirname "${STRONG_REPLAYS_ARCHIVE}")"

partial_archive="${STRONG_REPLAYS_ARCHIVE}.partial"
if [[ ! -f "${STRONG_REPLAYS_ARCHIVE}" ]]; then
  log "Downloading bsmathers/scraped_replays_parsed at ${STRONG_REPLAYS_REVISION}"
  curl_args=(--fail --location --retry 5 --retry-delay 5 --continue-at -)
  if [[ -n "${HF_TOKEN:-}" ]]; then
    curl_args+=(--header "Authorization: Bearer ${HF_TOKEN}")
  fi
  curl "${curl_args[@]}" --output "${partial_archive}" "${STRONG_REPLAYS_URL}"
  mv "${partial_archive}" "${STRONG_REPLAYS_ARCHIVE}"
fi

export GOOD_REPLAYS_DIR STRONG_REPLAYS_ARCHIVE STRONG_REPLAYS_REVISION STRONG_REPLAYS_SHA256
"${PYTHON_BIN}" - <<'PY'
import hashlib
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath

archive = Path(os.environ["STRONG_REPLAYS_ARCHIVE"])
output_root = Path(os.environ["GOOD_REPLAYS_DIR"])
expected = os.environ["STRONG_REPLAYS_SHA256"].lower()

digest = hashlib.sha256()
with archive.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
        digest.update(chunk)
actual = digest.hexdigest()
if actual != expected:
    raise SystemExit(f"Archive SHA-256 mismatch: {actual} != {expected}")

if output_root.exists() and any(path.is_file() for path in output_root.rglob("*")):
    raise SystemExit(f"Refusing to replace nonempty replay directory: {output_root}")

extract_root = output_root.with_name(output_root.name + ".extracting")
shutil.rmtree(extract_root, ignore_errors=True)
trajectory_dir = extract_root / "gen1ou"
trajectory_dir.mkdir(parents=True)
count = 0
try:
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"Unsafe archive member: {member.filename!r}")
            if member.is_dir() or "__MACOSX" in path.parts or path.name.startswith("._"):
                continue
            if not path.name.endswith((".json", ".json.lz4")):
                continue
            destination = trajectory_dir / path.name
            if destination.exists():
                raise SystemExit(f"Duplicate trajectory filename: {path.name}")
            with source.open(member) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            count += 1
    if count == 0:
        raise SystemExit(f"No trajectories found in {archive}")
    (extract_root / "SOURCE.txt").write_text(
        "Hugging Face dataset: bsmathers/scraped_replays_parsed\n"
        f"Revision: {os.environ['STRONG_REPLAYS_REVISION']}\n"
        f"Archive SHA-256: {actual}\n"
        f"Trajectory files: {count}\n",
        encoding="utf-8",
    )
    if output_root.exists():
        shutil.rmtree(output_root)
    extract_root.replace(output_root)
    archive.unlink(missing_ok=True)
except BaseException:
    shutil.rmtree(extract_root, ignore_errors=True)
    raise

print(f"Prepared {count:,} strong-model trajectories in {output_root / 'gen1ou'}")
PY
