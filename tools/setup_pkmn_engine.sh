#!/usr/bin/env bash
set -euo pipefail

# Build the pinned @pkmn/engine N-API addon used by the Gen 1 tournament.
# No Showdown server, apt dist-upgrade, git clone, or Zig tarball from ziglang.org.
# The npm lockfile pins the Zig source/JS ABI. Python's ziglang wheel supplies
# Zig 0.16.0 when it is not already installed.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_DIR="${ROOT_DIR}/metamon/env/vectorized"
ADDON="${HOST_DIR}/pkmn-showdown.node"
NPM_ENGINE="${HOST_DIR}/node_modules/@pkmn/engine"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "=== Native Gen 1 (@pkmn/engine Zig) setup ==="

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
    echo "ERROR: Node.js and npm are required. Install Node.js 20+ and rerun." >&2
    exit 1
fi
echo "Node: $(node --version); npm: $(npm --version)"

cd "${HOST_DIR}"

# Install from the exact repository lockfile without invoking the package's
# postinstall (which otherwise downloads Zig again from ziglang.org).
if [ ! -f "${NPM_ENGINE}/src/bin/install-pkmn-engine" ]; then
    echo "Installing pinned Node dependencies (skipping implicit Zig download)..."
    npm ci --ignore-scripts --no-audit --no-fund --legacy-peer-deps
fi

# The pinned npm package's driver resolves ./build/pkg/common.js. Retain the
# repo-specific adapter. npm ci may remove it, so repair before any require().
mkdir -p "${NPM_ENGINE}/build/pkg"
cp "${HOST_DIR}/pkmn_engine_common.js" "${NPM_ENGINE}/build/pkg/common.js"

verify_addon() {
    local path="$1"
    [ -s "$path" ] || return 1
    node - "$path" <<'NODE'
const path = require('path');
(async () => {
  const file = path.resolve(process.argv[2]);
  const native = require(file);
  if (!native?.engine?.options?.showdown || !native?.engine?.options?.log) {
    throw new Error('addon must be compiled with -Dshowdown -Dlog');
  }
  const engine = require('@pkmn/engine');
  if (typeof engine.initialize !== 'function') {
    throw new Error('pinned @pkmn/engine does not expose initialize()');
  }
  await engine.initialize(true, file);
  console.log('Verified native Zig addon (Showdown-compatible Gen 1, logging enabled)');
})().catch(error => {
  console.error('Native addon validation:', error.message);
  process.exitCode = 1;
});
NODE
}

# Most restarts should stop right here; neither Python nor Zig is installed.
if verify_addon "${ADDON}"; then
    echo "Native engine is ready; nothing to install or rebuild."
    exit 0
fi

# npm's installer can already have produced the correctly configured addon.
PKG_ADDON="${NPM_ENGINE}/build/lib/pkmn-showdown.node"
if verify_addon "${PKG_ADDON}"; then
    cp "${PKG_ADDON}" "${ADDON}"
    echo "Reused the existing Zig addon from the npm package."
    exit 0
fi

# Reuse an existing Zig 0.16.0 compiler before fetching anything.
if [ -x "${ROOT_DIR}/.zig/zig" ]; then
    export PATH="${ROOT_DIR}/.zig:${PATH}"
fi
if [ "$(zig version 2>/dev/null || true)" = "0.16.0" ]; then
    echo "Using existing Zig 0.16.0: $(command -v zig)"
else
    # The PyPI wheel is maintained by Zig's developers and uses the configured
    # pip index rather than the very slow ziglang.org tarball endpoint.
    if ! "${PYTHON_BIN}" -m ziglang version 2>/dev/null | grep -qx '0.16.0'; then
        echo "Installing Zig 0.16.0 from PyPI (one-time, via configured pip mirror)..."
        "${PYTHON_BIN}" -m pip install --disable-pip-version-check \
            --retries 2 --timeout 30 "ziglang==0.16.0"
    fi
    if ! "${PYTHON_BIN}" -m ziglang version 2>/dev/null | grep -qx '0.16.0'; then
        echo "ERROR: Zig 0.16.0 missing; try: ${PYTHON_BIN} -m pip install ziglang==0.16.0" >&2
        exit 1
    fi
    mkdir -p "${ROOT_DIR}/.zig"
    PYTHON_PATH="$(command -v "${PYTHON_BIN}")"
    printf '#!/bin/sh\nexec "%s" -m ziglang "$@"\n' "${PYTHON_PATH}" > "${ROOT_DIR}/.zig/zig"
    chmod +x "${ROOT_DIR}/.zig/zig"
    export PATH="${ROOT_DIR}/.zig:${PATH}"
fi
[ "$(zig version)" = "0.16.0" ] || {
    echo "ERROR: expected Zig 0.16.0; got $(zig version)" >&2
    exit 1
}

# Build directly from @pkmn/engine's *pinned npm source*. Its own installer
# obtains Node headers if necessary; there is no separate git checkout.
echo "Building pinned @pkmn/engine with -Dshowdown -Dlog..."
# Upstream install-pkmn-engine invokes Zig from the package root, but also
# renames the compiled addon to the relative path build/lib/pkmn-showdown.node.
# Invoking it from HOST_DIR makes that rename target the wrong directory and
# raises ENOENT even if Zig finished successfully.
(
    cd "${NPM_ENGINE}"
    node src/bin/install-pkmn-engine --options="-Dshowdown -Dlog"
)

if [ ! -s "${PKG_ADDON}" ]; then
    echo "ERROR: native build did not generate ${PKG_ADDON}" >&2
    exit 1
fi
cp "${PKG_ADDON}" "${ADDON}"
verify_addon "${ADDON}" || {
    rm -f "${ADDON}"
    echo "ERROR: generated addon failed native initialization checks." >&2
    exit 1
}
echo "Native Zig simulator ready. No Showdown websocket server needed."
