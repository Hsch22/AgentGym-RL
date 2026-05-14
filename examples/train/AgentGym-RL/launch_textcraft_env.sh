#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PORT="${PORT:-${TEXTCRAFT_ENV_PORT:-36005}}"
AGENTGYM_ROOT="${AGENTGYM_ROOT:-${REPO_ROOT}/AgentGym}"
TEXTCRAFT_SERVER_BIN="${TEXTCRAFT_SERVER_BIN:-${REPO_ROOT}/.venv-textcraft-server/bin/textcraft}"
TEXTCRAFT_ENV_SESSION_NAME="${TEXTCRAFT_ENV_SESSION_NAME:-textcraft_server_${PORT}_uv}"
TEXTCRAFT_ENV_LOG_FILE="${TEXTCRAFT_ENV_LOG_FILE:-/tmp/textcraft_server_${PORT}_uv.log}"
TEXTCRAFT_PREFLIGHT_CREATE="${TEXTCRAFT_PREFLIGHT_CREATE:-1}"

health_check() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/" >/dev/null
}

create_check() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    curl -fsS --max-time 15 \
      -X POST "http://127.0.0.1:${PORT}/create" \
      -H "Content-Type: application/json" \
      -d '{"minecraft_dir":"agentenv_textcraft/","commands":null,"goal":null}' >/dev/null
}

if [[ ! -x "${TEXTCRAFT_SERVER_BIN}" ]]; then
  echo "[textcraft-env] missing executable: ${TEXTCRAFT_SERVER_BIN}" >&2
  echo "[textcraft-env] expected uv env: ${REPO_ROOT}/.venv-textcraft-server" >&2
  exit 1
fi

if [[ ! -d "${AGENTGYM_ROOT}/agentenv-textcraft/agentenv_textcraft/recipes" ]]; then
  echo "[textcraft-env] missing TextCraft recipe directory under ${AGENTGYM_ROOT}/agentenv-textcraft" >&2
  exit 1
fi

if health_check; then
  echo "[textcraft-env] server already running on http://127.0.0.1:${PORT}"
  if [[ "${TEXTCRAFT_PREFLIGHT_CREATE}" == "1" ]]; then
    create_check
    echo "[textcraft-env] /create preflight ok"
  fi
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "[textcraft-env] tmux is required to start the background server" >&2
  exit 1
fi

tmux kill-session -t "${TEXTCRAFT_ENV_SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${TEXTCRAFT_ENV_SESSION_NAME}" "\
  set -euo pipefail; \
  cd '${AGENTGYM_ROOT}/agentenv-textcraft'; \
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; \
  export no_proxy='127.0.0.1,localhost'; \
  export NO_PROXY='127.0.0.1,localhost'; \
  echo '[textcraft-env] starting ${TEXTCRAFT_SERVER_BIN} on port ${PORT}'; \
  exec '${TEXTCRAFT_SERVER_BIN}' --host 0.0.0.0 --port '${PORT}' \
  2>&1 | tee '${TEXTCRAFT_ENV_LOG_FILE}'"

for _ in $(seq 1 30); do
  if health_check; then
    echo "[textcraft-env] server ready on http://127.0.0.1:${PORT}"
    if [[ "${TEXTCRAFT_PREFLIGHT_CREATE}" == "1" ]]; then
      create_check
      echo "[textcraft-env] /create preflight ok"
    fi
    echo "[textcraft-env] tmux session: ${TEXTCRAFT_ENV_SESSION_NAME}"
    echo "[textcraft-env] log: ${TEXTCRAFT_ENV_LOG_FILE}"
    exit 0
  fi
  sleep 1
done

echo "[textcraft-env] server did not become ready on http://127.0.0.1:${PORT}" >&2
tail -n 80 "${TEXTCRAFT_ENV_LOG_FILE}" >&2 || true
exit 1
