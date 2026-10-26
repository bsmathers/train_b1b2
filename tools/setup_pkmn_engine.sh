#!/usr/bin/env bash
set -euo pipefail

# Setup script for @pkmn/engine native Zig build (Linux/macOS)
# Builds pkmn-showdown.node with -Dshowdown -Dlog for 50x faster battle simulation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PKMN_DIR="${ROOT_DIR}/pkmn-engine"
PKMN_ENGINE_REF="${PKMN_ENGINE_REF:-78dc891c}"

echo "=== Metamon @pkmn/engine Setup ==="

# 0. Check or install Node.js / npm
if [ -d "${ROOT_DIR}/.node/bin" ]; then
    export PATH="${ROOT_DIR}/.node/bin:${PATH}"
fi

if command -v node &> /dev/null && command -v npm &> /dev/null; then
    echo "Found Node.js: $(node --version)"
    echo "Found npm:     $(npm --version)"
else
    echo "Node.js or npm not found in PATH."
    if command -v apt-get &> /dev/null && [ "$(id -u)" -eq 0 ]; then
        echo "Installing Node.js 20 LTS via apt..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
        apt-get install -y nodejs
    else
        echo "Downloading standalone Node.js Linux x86_64 binary..."
        NODE_VER="v20.18.0"
        NODE_TAR="node-${NODE_VER}-linux-x64.tar.xz"
        curl -L "https://nodejs.org/dist/${NODE_VER}/${NODE_TAR}" -o "/tmp/${NODE_TAR}"
        mkdir -p "${ROOT_DIR}/.node"
        tar -xf "/tmp/${NODE_TAR}" -C "${ROOT_DIR}/.node" --strip-components=1
        rm "/tmp/${NODE_TAR}"
        export PATH="${ROOT_DIR}/.node/bin:${PATH}"
    fi
    echo "Installed Node.js: $(node --version)"
fi

# 1. Check or install Zig (pkmn/engine requires Zig >= 0.16.0)
if [ -d "${ROOT_DIR}/.zig" ]; then
    export PATH="${ROOT_DIR}/.zig:${PATH}"
fi

NEED_ZIG=1
if command -v zig &> /dev/null; then
    CURRENT_ZIG=$(zig version)
    echo "Found Zig: ${CURRENT_ZIG}"
    MAJOR=$(echo "${CURRENT_ZIG}" | cut -d. -f1)
    MINOR=$(echo "${CURRENT_ZIG}" | cut -d. -f2)
    if [ "${MAJOR}" -gt 0 ] || [ "${MINOR}" -ge 16 ]; then
        NEED_ZIG=0
    else
        echo "Zig ${CURRENT_ZIG} is too old (pkmn/engine requires >= 0.16.0). Upgrading..."
    fi
fi

if [ "${NEED_ZIG}" -eq 1 ]; then
    echo "Downloading standalone Zig 0.16.0..."
    OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
    ARCH="$(uname -m)"
    if [ "${ARCH}" = "x86_64" ]; then
        ZIG_ARCH="x86_64"
    elif [ "${ARCH}" = "arm64" ] || [ "${ARCH}" = "aarch64" ]; then
        ZIG_ARCH="aarch64"
    else
        ZIG_ARCH="${ARCH}"
    fi

    if [ "${OS}" = "darwin" ]; then
        ZIG_OS="macos"
    else
        ZIG_OS="linux"
    fi

    ZIG_VERSION="0.16.0"
    ZIG_TAR="zig-${ZIG_ARCH}-${ZIG_OS}-${ZIG_VERSION}.tar.xz"
    ZIG_URL="https://ziglang.org/download/${ZIG_VERSION}/${ZIG_TAR}"

    rm -rf "${ROOT_DIR}/.zig"
    mkdir -p "${ROOT_DIR}/.zig"
    curl -L "${ZIG_URL}" -o "/tmp/${ZIG_TAR}"
    tar -xf "/tmp/${ZIG_TAR}" -C "${ROOT_DIR}/.zig" --strip-components=1
    rm "/tmp/${ZIG_TAR}"
    export PATH="${ROOT_DIR}/.zig:${PATH}"
    echo "Installed Zig: $(zig version)"
fi

# 2. Clone pkmn/engine if needed, then build the exact revision used by the
# pinned JavaScript package. Mixing a newer native ABI with an older wrapper
# can fail at startup or corrupt simulator results.
if [ ! -d "${PKMN_DIR}" ]; then
    echo "Cloning pkmn/engine..."
    git clone --no-checkout --filter=blob:none https://github.com/pkmn/engine.git "${PKMN_DIR}"
fi

cd "${PKMN_DIR}"
git fetch --filter=blob:none origin
git checkout --detach "${PKMN_ENGINE_REF}"

# 3. Build native addon with -Dshowdown -Dlog
echo "Building @pkmn/engine native addon (-Dshowdown -Dlog)..."
node src/bin/install-pkmn-engine --options="-Dshowdown -Dlog"

# 4. Install npm dependencies in vectorized env
echo "Setting up Node.js packages..."
cd "${ROOT_DIR}/metamon/env/vectorized"
npm ci --legacy-peer-deps

if [ -f "${ROOT_DIR}/metamon/env/vectorized/pkmn_engine_common.js" ]; then
    mkdir -p "${ROOT_DIR}/metamon/env/vectorized/node_modules/@pkmn/engine/build/pkg"
    cp "${ROOT_DIR}/metamon/env/vectorized/pkmn_engine_common.js" "${ROOT_DIR}/metamon/env/vectorized/node_modules/@pkmn/engine/build/pkg/common.js"
fi

# 5. Link built addon into vectorized env
if [ -f "${PKMN_DIR}/zig-out/lib/pkmn-showdown.node" ]; then
    cp "${PKMN_DIR}/zig-out/lib/pkmn-showdown.node" "${ROOT_DIR}/metamon/env/vectorized/"
    echo "Successfully built and copied pkmn-showdown.node to metamon/env/vectorized/"
elif [ -f "${PKMN_DIR}/build/lib/pkmn-showdown.node" ]; then
    cp "${PKMN_DIR}/build/lib/pkmn-showdown.node" "${ROOT_DIR}/metamon/env/vectorized/"
    echo "Successfully built and copied pkmn-showdown.node to metamon/env/vectorized/"
else
    echo "ERROR: the @pkmn/engine build completed without producing pkmn-showdown.node"
    exit 1
fi

echo "=== @pkmn/engine setup complete! ==="
