#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-babyai_env}"
PORT="${PORT:-36025}"
LOG_FILE="${LOG_FILE:-/tmp/babyai_env_${PORT}.log}"

# Local service should never use inherited proxies.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "\
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; \
  export no_proxy='127.0.0.1,localhost'; \
  export NO_PROXY='127.0.0.1,localhost'; \
  /share/project/husicheng/muhan/AgentGym-RL/.venv/bin/babyai --host 127.0.0.1 --port ${PORT} 2>&1 | tee ${LOG_FILE}"

echo "Started BabyAI env server in tmux session ${SESSION_NAME} on port ${PORT}"
echo "Log: ${LOG_FILE}"
