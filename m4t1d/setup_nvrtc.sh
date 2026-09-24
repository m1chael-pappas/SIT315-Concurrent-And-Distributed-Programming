#!/usr/bin/env bash
# SIT315 M4.T1D - installs NVRTC, the CUDA runtime compiler the cuda backend
# needs, into a project-local Python virtual environment. The CUDA driver
# itself comes from the Windows NVIDIA driver through /usr/lib/wsl/lib, so no
# CUDA toolkit or sudo is required.
#
#   ./setup_nvrtc.sh
#   source ./cuda_env.sh      # sets LD_LIBRARY_PATH for the current shell
#
# The NVRTC version matches the cudarc feature in Cargo.toml (cuda-13030) and
# must not be newer than the driver's CUDA version shown by nvidia-smi.

set -euo pipefail
cd "$(dirname "$0")"

NVRTC_VERSION=${NVRTC_VERSION:-13.3.33}
PYTHON=${PYTHON:-python3}

if [ ! -d .venv ]; then
    "$PYTHON" -m venv .venv
fi
.venv/bin/pip install --quiet --disable-pip-version-check "nvidia-cuda-nvrtc==$NVRTC_VERSION"

lib=$(find .venv -name 'libnvrtc.so.13' -printf '%h\n' | head -1)
if [ -z "$lib" ]; then
    echo "libnvrtc.so.13 not found under .venv" >&2
    exit 1
fi

cat > cuda_env.sh <<EOF
export LD_LIBRARY_PATH="$(pwd)/$lib:/usr/lib/wsl/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
echo "NVRTC $NVRTC_VERSION installed in $(pwd)/$lib"
echo "run: source ./cuda_env.sh"
