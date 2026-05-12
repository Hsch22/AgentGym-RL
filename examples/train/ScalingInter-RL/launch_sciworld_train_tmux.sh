#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${SESSION_NAME:-sciworld_scalinginter}"
SCRIPT_PATH="${SCRIPT_PATH:-${SCRIPT_DIR}/sciworld_train.sh}"
LOG_FILE="${LOG_FILE:-/tmp/sciworld_scalinginter_train.log}"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "bash ${SCRIPT_PATH} 2>&1 | tee ${LOG_FILE}"

echo "Started SciWorld training in tmux session ${SESSION_NAME}"
echo "Log: ${LOG_FILE}"
