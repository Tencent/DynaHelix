#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="dyna2"
PYTHON_VERSION="3.11"
TORCH_VERSION="2.7.1"
TORCHVISION_VERSION="0.22.1"
TORCHAUDIO_VERSION="2.7.1"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
INSTALL_MODE="CUDA 12.6"

usage() {
    cat <<'EOF'
Usage: bash install.sh [--cpu | --cuda 12.6]

Options:
  --cpu        Install the CPU-only PyTorch build.
  --cuda 12.6  Install the CUDA 12.6 PyTorch build (default).
  -h, --help   Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            PYTORCH_INDEX="https://download.pytorch.org/whl/cpu"
            INSTALL_MODE="CPU"
            shift
            ;;
        --cuda)
            if [[ ${2:-} != "12.6" ]]; then
                echo "Error: only CUDA 12.6 is supported." >&2
                usage >&2
                exit 2
            fi
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
            INSTALL_MODE="CUDA 12.6"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Error: unknown option '$1'." >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda is required but was not found in PATH." >&2
    exit 1
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if conda run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
    echo "Using existing conda environment: $ENV_NAME"
else
    echo "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

pip_in_env() {
    PIP_ROOT_USER_ACTION=ignore conda run --no-capture-output -n "$ENV_NAME" \
        python -m pip "$@"
}

echo "Installing PyTorch $TORCH_VERSION ($INSTALL_MODE)"
pip_in_env install --upgrade pip
pip_in_env install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    --index-url "$PYTORCH_INDEX"

echo "Installing project dependencies"
pip_in_env install -r requirements.txt
pip_in_env install -e . --no-deps

echo "Installation complete. Activate the environment with: conda activate $ENV_NAME"
