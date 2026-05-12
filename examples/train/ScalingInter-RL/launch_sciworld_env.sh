#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-sciworld_env}"
PORT="${PORT:-36005}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCIWORLD_SERVER_BIN="${SCIWORLD_SERVER_BIN:-${REPO_ROOT}/.venv/bin/sciworld}"
JAVA_HOME_DEFAULT="${REPO_ROOT}/.venv"

# Local service should never use inherited proxies.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

export JAVA_HOME="${JAVA_HOME:-${JAVA_HOME_DEFAULT}}"
export PATH="${JAVA_HOME}/bin:${PATH}"

LOG_FILE="${LOG_FILE:-/tmp/sciworld_env_${PORT}.log}"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "\
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; \
  export no_proxy='127.0.0.1,localhost'; \
  export NO_PROXY='127.0.0.1,localhost'; \
  export JAVA_HOME='${JAVA_HOME}'; \
  export PATH='${JAVA_HOME}/bin:'\"\$PATH\"; \
  ${SCIWORLD_SERVER_BIN} --host 127.0.0.1 --port ${PORT} 2>&1 | tee ${LOG_FILE}"

echo "Started SciWorld env server in tmux session ${SESSION_NAME} on port ${PORT}"
echo "Log: ${LOG_FILE}"
