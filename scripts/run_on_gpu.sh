#!/usr/bin/env bash
# run_on_gpu.sh - Execute Python scripts on deepreasoning GPU server
# Usage: ./scripts/run_on_gpu.sh path/to/script.py [args...]

set -euo pipefail

REMOTE_HOST="deepreasoning"
REMOTE_USER="${USER}"
PROJECT_DIR="/home/cohenn1/NCE"
REMOTE_PROJECT_DIR="${PROJECT_DIR}"

SCRIPT_PATH="$1"
shift
SCRIPT_ARGS="$@"

# Ensure script exists locally
if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "Error: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

# Convert to absolute path if relative
if [[ "${SCRIPT_PATH}" != /* ]]; then
    SCRIPT_PATH="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)/$(basename "${SCRIPT_PATH}")"
fi

# Verify script is within project directory
if [[ "${SCRIPT_PATH}" != ${PROJECT_DIR}/* ]]; then
    echo "Error: Script must be within ${PROJECT_DIR}" >&2
    exit 1
fi

# Get relative path from project root
REL_SCRIPT_PATH="${SCRIPT_PATH#${PROJECT_DIR}/}"

echo "==> Syncing project files to ${REMOTE_HOST}..."
rsync -az --delete \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints' \
    --exclude='*.pkl' \
    --exclude='*.pth' \
    "${PROJECT_DIR}/" "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/"

echo "==> Running on ${REMOTE_HOST}: ${REL_SCRIPT_PATH} ${SCRIPT_ARGS}"
ssh -t "${REMOTE_HOST}" "cd ${REMOTE_PROJECT_DIR} && source venv/bin/activate && python3 ${REL_SCRIPT_PATH} ${SCRIPT_ARGS}"

echo "==> Syncing results back from ${REMOTE_HOST}..."
rsync -az \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/" "${PROJECT_DIR}/"

echo "==> Done"
