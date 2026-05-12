#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$ROOT/.venv/bin/python"
PY310="${PY310:-$(command -v python3.10 || true)}"
AGENTGYM_DIR="$ROOT/AgentGym"
AGENTGYM_RL_DIR="$ROOT/AgentGym-RL"
LMRLGYM_DIR="$AGENTGYM_DIR/agentenv-lmrlgym/lmrlgym"
LMRLGYM_COMMIT="83abeedb3a461c3d7d20572b73318a259d85f2ac"
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_WHEEL_NAME="${FLASH_URL##*/}"
FLASH_WHEEL_DIR="${FLASH_WHEEL_DIR:-${TMPDIR:-/tmp}}"

export PATH="$HOME/.local/bin:$PATH"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [[ -n "${AGENTGYM_HTTP_PROXY:-}" ]]; then
  export http_proxy="$AGENTGYM_HTTP_PROXY"
  export https_proxy="$AGENTGYM_HTTP_PROXY"
  export HTTP_PROXY="$AGENTGYM_HTTP_PROXY"
  export HTTPS_PROXY="$AGENTGYM_HTTP_PROXY"
fi
if [[ -n "${AGENTGYM_SOCKS_PROXY:-}" ]]; then
  export all_proxy="$AGENTGYM_SOCKS_PROXY"
  export ALL_PROXY="$AGENTGYM_SOCKS_PROXY"
fi
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

have_import() {
  local module="$1"
  "$VENV_PY" - <<PY >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("$module") else 1)
PY
}

ensure_agentgym_sources() {
  if [[ ! -f "$AGENTGYM_DIR/agentenv/pyproject.toml" ]]; then
    log "AgentGym sources missing, cloning again"
    rm -rf "$AGENTGYM_DIR"
    git clone https://github.com/WooooDyy/AgentGym "$AGENTGYM_DIR"
    git -C "$AGENTGYM_DIR" checkout d014732d9fe39b975c368c03749bfd50950067f6
  fi

  if [[ ! -f "$LMRLGYM_DIR/setup.py" ]]; then
    log "LMRL-Gym sources missing, cloning submodule payload manually"
    rm -rf "$LMRLGYM_DIR"
    git clone https://github.com/abdulhaim/LMRL-Gym.git "$LMRLGYM_DIR"
  fi

  if git -C "$LMRLGYM_DIR" rev-parse HEAD >/dev/null 2>&1; then
    local current
    current="$(git -C "$LMRLGYM_DIR" rev-parse HEAD)"
    if [[ "$current" != "$LMRLGYM_COMMIT" ]]; then
      log "Checking out LMRL-Gym commit $LMRLGYM_COMMIT"
      git -C "$LMRLGYM_DIR" checkout "$LMRLGYM_COMMIT"
    fi
  else
    log "LMRL-Gym git metadata is unavailable, using extracted sources"
  fi
}

wait_or_install_torch() {
  if have_import torch; then
    log "torch already importable"
    return
  fi

  while pgrep -af "uv .*${VENV_PY}.*torch==2.4.0" >/dev/null 2>&1; do
    log "Detected existing torch install process, waiting"
    sleep 60
    if have_import torch; then
      log "torch became importable while waiting"
      return
    fi
  done

  log "Installing torch==2.4.0 from PyTorch cu124 index"
  "$UV_BIN" pip install --python "$VENV_PY" torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
}

install_flash_attn() {
  if have_import flash_attn; then
    log "flash_attn already importable"
    return
  fi

  local tmp_whl
  mkdir -p "$FLASH_WHEEL_DIR"
  tmp_whl="$FLASH_WHEEL_DIR/$FLASH_WHEEL_NAME"
  if [[ -f "$tmp_whl" ]]; then
    log "Using existing flash-attn wheel $tmp_whl"
  else
    log "Downloading flash-attn wheel"
    curl -L "$FLASH_URL" -o "$tmp_whl"
  fi
  log "Installing flash-attn wheel"
  "$UV_BIN" pip install --python "$VENV_PY" "$tmp_whl"
}

main() {
  log "Starting background AgentGym-RL uv setup"

  if [[ -z "$PY310" ]]; then
    log "python3.10 not found in PATH"
    exit 1
  fi

  if [[ -z "$UV_BIN" ]]; then
    log "uv not found in PATH"
    exit 1
  fi

  log "ROOT=$ROOT"
  log "PY310=$PY310"
  log "UV_BIN=$UV_BIN"

  if [[ ! -x "$VENV_PY" ]]; then
    log "Creating uv venv with Python 3.10"
    "$UV_BIN" venv "$ROOT/.venv" --python "$PY310"
  fi

  ensure_agentgym_sources
  wait_or_install_torch
  install_flash_attn

  log "Installing AgentGym-RL editable package"
  "$UV_BIN" pip install --python "$VENV_PY" -e "$AGENTGYM_RL_DIR"

  log "Installing AgentGym agentenv editable package"
  "$UV_BIN" pip install --python "$VENV_PY" -e "$AGENTGYM_DIR/agentenv"

  log "Pinning transformers==4.51.3"
  "$UV_BIN" pip install --python "$VENV_PY" transformers==4.51.3

  log "Running import verification"
  "$VENV_PY" - <<'PY'
import torch
import transformers
import verl
import agentenv
import flash_attn

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("verl", getattr(verl, "__file__", "n/a"))
print("agentenv", getattr(agentenv, "__file__", "n/a"))
print("flash_attn", getattr(flash_attn, "__file__", "n/a"))
PY

  log "Background AgentGym-RL uv setup completed"
}

main "$@"
