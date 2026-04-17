#!/usr/bin/env bash
set -euo pipefail

ROOT="/share/project/husicheng/muhan/AgentGym-RL"
VENV_PY="$ROOT/.venv/bin/python"
PY310="/share/project/tanhuajie/miniconda3/envs/husicheng/bin/python3.10"
AGENTGYM_DIR="$ROOT/AgentGym"
AGENTGYM_RL_DIR="$ROOT/AgentGym-RL"
LMRLGYM_DIR="$AGENTGYM_DIR/agentenv-lmrlgym/lmrlgym"
LMRLGYM_COMMIT="83abeedb3a461c3d7d20572b73318a259d85f2ac"
FLASH_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

export https_proxy="http://10.8.36.23:2080"
export http_proxy="http://10.8.36.23:2080"

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

  if [[ ! -d "$LMRLGYM_DIR/.git" ]]; then
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
  fi
}

wait_or_install_torch() {
  if have_import torch; then
    log "torch already importable"
    return
  fi

  while pgrep -af "uv .*muhan/AgentGym-RL/.venv/bin/python.*torch==2.4.0" >/dev/null 2>&1; do
    log "Detected existing torch install process, waiting"
    sleep 60
    if have_import torch; then
      log "torch became importable while waiting"
      return
    fi
  done

  log "Installing torch==2.4.0 from PyTorch cu124 index"
  uv pip install --python "$VENV_PY" torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
}

install_flash_attn() {
  if have_import flash_attn; then
    log "flash_attn already importable"
    return
  fi

  local tmp_whl
  tmp_whl="$(mktemp "$ROOT/flash_attn.XXXXXX.whl")"
  trap 'rm -f "$tmp_whl"' RETURN
  log "Downloading flash-attn wheel"
  curl -L "$FLASH_URL" -o "$tmp_whl"
  log "Installing flash-attn wheel"
  uv pip install --python "$VENV_PY" "$tmp_whl"
  rm -f "$tmp_whl"
  trap - RETURN
}

main() {
  log "Starting background AgentGym-RL uv setup"

  if [[ ! -x "$VENV_PY" ]]; then
    log "Creating uv venv with Python 3.10"
    uv venv "$ROOT/.venv" --python "$PY310"
  fi

  ensure_agentgym_sources
  wait_or_install_torch
  install_flash_attn

  log "Installing AgentGym-RL editable package"
  uv pip install --python "$VENV_PY" -e "$AGENTGYM_RL_DIR"

  log "Installing AgentGym agentenv editable package"
  uv pip install --python "$VENV_PY" -e "$AGENTGYM_DIR/agentenv"

  log "Pinning transformers==4.51.3"
  uv pip install --python "$VENV_PY" transformers==4.51.3

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
