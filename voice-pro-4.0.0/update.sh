#!/bin/bash
set -e

echo "========================================================================="
echo ""
echo "  ABUS Updater [Version 4.0 - uv]"
echo "  contact: abus.aikorea@gmail.com"
echo ""
echo "========================================================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Setup paths
INSTALL_DIR="$SCRIPT_DIR/installer_files"
INSTALL_ENV_DIR="$INSTALL_DIR/env"
UV_DIR="$INSTALL_DIR/uv"
UV_EXE="$UV_DIR/uv"

# Set temp directories
export TMP="$INSTALL_DIR"
export TEMP="$INSTALL_DIR"

# keep everything project-local (no global uv/python/cache pollution)
export UV_PYTHON_INSTALL_DIR="$INSTALL_DIR/python"
export UV_CACHE_DIR="$INSTALL_DIR/uv-cache"
export UV_PROJECT_ENVIRONMENT="$INSTALL_ENV_DIR"

# environment isolation
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONHOME

# Check if uv exists
if [ ! -f "$UV_EXE" ]; then
    echo "uv not found. Please run start.sh first to set up the environment."
    exit 1
fi

# figure out the GPU choice: GPU_CHOICE env var > saved choice > default CPU
SAVED_CHOICE_FILE="$INSTALL_DIR/gpu_choice.txt"
if [ -z "$GPU_CHOICE" ] && [ -f "$SAVED_CHOICE_FILE" ]; then
    GPU_CHOICE="$(cat "$SAVED_CHOICE_FILE")"
fi
GPU_CHOICE="$(echo "${GPU_CHOICE:-C}" | tr '[:lower:]' '[:upper:]')"
echo "$GPU_CHOICE" > "$SAVED_CHOICE_FILE"

if [ "$GPU_CHOICE" == "G" ]; then
    SYNC_EXTRA="gpu"
else
    SYNC_EXTRA="cpu"
fi
echo "Updating environment (extra: $SYNC_EXTRA) from uv.lock ..."

# update the environment to exactly match the committed lockfile
"$UV_EXE" sync --frozen --extra "$SYNC_EXTRA"

echo ""
echo "Update finished successfully."
