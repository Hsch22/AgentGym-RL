#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

QUEUE_SESSION="${SCIWORLD_LONG_QUEUE_SESSION:-sciworld_long_a3_b0_100step_20260530}"
QUEUE_LOG="${SCIWORLD_LONG_QUEUE_LOG:-${REPO_ROOT}/logs/sciworld_long_a3_b0_100step_20260530_queue.log}"
A3_RUN_DIR="${SCIWORLD_LONG_A3_RUN_DIR:-${REPO_ROOT}/results/sciworld/sciworld_A3_g2rl_normalized_action_gradient_3b_100step_continue_from25_20260530_long100_continue}"
B0_RUN_DIR="${SCIWORLD_LONG_B0_RUN_DIR:-${REPO_ROOT}/results/sciworld/sciworld_B0_strict_no_cluster_3b_100step_continue_from25_20260530_long100_continue}"
TARGET_STEPS="${SCIWORLD_LONG_TARGET_STEPS:-50 75 100}"

POLL_MIN_SECONDS="${POLL_MIN_SECONDS:-300}"
POLL_MAX_SECONDS="${POLL_MAX_SECONDS:-3600}"
POLL_NEAR_CHECKPOINT_SECONDS="${POLL_NEAR_CHECKPOINT_SECONDS:-900}"
POLL_EVAL_SECONDS="${POLL_EVAL_SECONDS:-600}"
STEP_SECONDS_FALLBACK="${STEP_SECONDS_FALLBACK:-200}"
WATCH_LOG="${WATCH_LOG:-}"
WATCH_ONCE="${WATCH_ONCE:-0}"

log() {
  local line
  line="[$(date '+%F %T')] $*"
  if [[ -n "${WATCH_LOG}" ]]; then
    echo "${line}" | tee -a "${WATCH_LOG}"
  else
    echo "${line}"
  fi
}

latest_executer_step() {
  local run_dir="$1"
  [[ -d "${run_dir}/executer_logs" ]] || return 0
  find "${run_dir}/executer_logs" -maxdepth 1 -type d -name 'step*' 2>/dev/null \
    | sed 's/.*step//' \
    | sort -n \
    | tail -1
}

latest_saved_checkpoint() {
  local run_dir="$1"
  [[ -d "${run_dir}" ]] || return 0
  find "${run_dir}" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null \
    | sed 's/.*global_step_//' \
    | sort -n \
    | tail -1
}

latest_step_seconds() {
  if [[ ! -f "${QUEUE_LOG}" ]]; then
    echo "${STEP_SECONDS_FALLBACK}"
    return
  fi
  "${REPO_ROOT}/.venv/bin/python" - "${QUEUE_LOG}" "${STEP_SECONDS_FALLBACK}" <<'PY'
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
fallback = float(sys.argv[2])
pattern = re.compile(r"timing_s/step:([+-]?(?:\d+(?:\.\d*)?|\.\d+))")
values = []
for line in log_path.read_text(errors="ignore").splitlines():
    match = pattern.search(line)
    if match:
        values.append(float(match.group(1)))
if not values:
    print(int(fallback))
else:
    recent = values[-5:]
    recent.sort()
    print(int(recent[len(recent) // 2]))
PY
}

log_has() {
  local pattern="$1"
  [[ -f "${QUEUE_LOG}" ]] && rg -q "${pattern}" "${QUEUE_LOG}"
}

current_phase() {
  local latest_a3="$1"
  local latest_b0="$2"

  if [[ -n "${latest_b0}" ]] || log_has "training B0"; then
    echo "B0"
  elif [[ -n "${latest_a3}" ]] || log_has "training A3"; then
    echo "A3"
  else
    echo "pending"
  fi
}

next_target_after() {
  local step="$1"
  local target
  for target in ${TARGET_STEPS}; do
    if (( step < target )); then
      echo "${target}"
      return
    fi
  done
}

max_target_step() {
  local target
  local max_target=0
  for target in ${TARGET_STEPS}; do
    if (( target > max_target )); then
      max_target="${target}"
    fi
  done
  echo "${max_target}"
}

clamp_sleep() {
  local seconds="$1"
  if (( seconds < POLL_MIN_SECONDS )); then
    seconds="${POLL_MIN_SECONDS}"
  fi
  if (( seconds > POLL_MAX_SECONDS )); then
    seconds="${POLL_MAX_SECONDS}"
  fi
  echo "${seconds}"
}

next_poll_seconds() {
  local phase="$1"
  local step="$2"
  local same_step_count="$3"
  local step_seconds
  local target
  local remaining_steps
  local remaining_seconds
  local next_sleep

  if [[ "${phase}" == "pending" || -z "${step}" ]]; then
    echo "${POLL_MIN_SECONDS}"
    return
  fi

  target="$(next_target_after "${step}")"
  if [[ -z "${target}" ]]; then
    echo "${POLL_EVAL_SECONDS}"
    return
  fi

  step_seconds="$(latest_step_seconds)"
  remaining_steps=$((target - step))
  remaining_seconds=$((remaining_steps * step_seconds))
  next_sleep=$((remaining_seconds - POLL_NEAR_CHECKPOINT_SECONDS))

  if (( same_step_count > 0 )); then
    next_sleep=$((next_sleep + same_step_count * POLL_MIN_SECONDS))
  fi

  clamp_sleep "${next_sleep}"
}

queue_running() {
  tmux has-session -t "${QUEUE_SESSION}" 2>/dev/null
}

check_for_failure() {
  [[ -f "${QUEUE_LOG}" ]] || return 0
  if rg -q "Traceback|RuntimeError|Cannot strictly resume|CUDA out of memory|Exception" "${QUEUE_LOG}"; then
    log "failure pattern found in ${QUEUE_LOG}"
    rg -n "Traceback|RuntimeError|Cannot strictly resume|CUDA out of memory|Exception" "${QUEUE_LOG}" | tail -n 20
    return 1
  fi
}

main() {
  local latest_a3=""
  local latest_b0=""
  local latest_a3_ckpt=""
  local latest_b0_ckpt=""
  local phase="pending"
  local active_step=""
  local last_phase=""
  local last_step=""
  local same_step_count=0
  local max_target
  local next_sleep

  max_target="$(max_target_step)"

  log "watcher started queue_session=${QUEUE_SESSION}"
  log "polling min=${POLL_MIN_SECONDS}s max=${POLL_MAX_SECONDS}s near_checkpoint=${POLL_NEAR_CHECKPOINT_SECONDS}s eval=${POLL_EVAL_SECONDS}s"

  while true; do
    check_for_failure

    latest_a3="$(latest_executer_step "${A3_RUN_DIR}")"
    latest_b0="$(latest_executer_step "${B0_RUN_DIR}")"
    latest_a3_ckpt="$(latest_saved_checkpoint "${A3_RUN_DIR}")"
    latest_b0_ckpt="$(latest_saved_checkpoint "${B0_RUN_DIR}")"
    phase="$(current_phase "${latest_a3}" "${latest_b0}")"

    if [[ "${phase}" == "B0" ]]; then
      active_step="${latest_b0}"
    elif [[ "${phase}" == "A3" ]]; then
      active_step="${latest_a3}"
    else
      active_step=""
    fi

    if [[ "${phase}:${active_step}" == "${last_phase}:${last_step}" ]]; then
      same_step_count=$((same_step_count + 1))
    else
      same_step_count=0
    fi
    last_phase="${phase}"
    last_step="${active_step}"

    log "phase=${phase} active_step=${active_step:-none} a3_ckpt=${latest_a3_ckpt:-none} b0_ckpt=${latest_b0_ckpt:-none}"

    if ! queue_running; then
      if log_has "SciWorld long A3/B0 queue completed"; then
        log "queue completed"
        return 0
      fi
      log "queue tmux ended before completion marker; inspect ${QUEUE_LOG}"
      return 2
    fi

    next_sleep="$(next_poll_seconds "${phase}" "${active_step}" "${same_step_count}")"
    if [[ "${phase}" == "A3" && -n "${active_step}" && "${active_step}" -ge "${max_target}" && "${latest_a3_ckpt:-0}" -lt "${max_target}" ]]; then
      next_sleep="${POLL_MIN_SECONDS}"
    fi
    if [[ "${phase}" == "B0" && -n "${active_step}" && "${active_step}" -ge "${max_target}" && "${latest_b0_ckpt:-0}" -lt "${max_target}" ]]; then
      next_sleep="${POLL_MIN_SECONDS}"
    fi
    if [[ "${WATCH_ONCE}" == "1" ]]; then
      log "next_poll=${next_sleep}s watch_once=1"
      return 0
    fi
    log "next_poll=${next_sleep}s"
    sleep "${next_sleep}"
  done
}

main "$@"
