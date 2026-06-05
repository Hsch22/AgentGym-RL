#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
SEEDS="${SCIWORLD_MULTI_SEEDS:-2 3}"
TOTAL_STEPS="${SCIWORLD_MULTI_TOTAL_STEPS:-100}"
SAVE_FREQ="${SCIWORLD_MULTI_SAVE_FREQ:-25}"
PORT="${SCIWORLD_MULTI_ENV_PORT:-36006}"
ENV_ADDR="${SCIWORLD_MULTI_ENV_ADDR:-http://127.0.0.1:${PORT}}"
TRAIN_GPUS="${SCIWORLD_MULTI_TRAIN_GPUS:-0,1}"
N_GPUS="${SCIWORLD_MULTI_N_GPUS:-2}"
MODEL_SAVE_ROOT="${SCIWORLD_MULTI_MODEL_SAVE_ROOT:-${REPO_ROOT}/results/sciworld_multiseed}"
LOG_ROOT="${SCIWORLD_MULTI_LOG_ROOT:-${REPO_ROOT}/logs}"
EVAL_CHECKPOINTS="${SCIWORLD_MULTI_EVAL_CHECKPOINTS:-50 75 100}"

mkdir -p "${MODEL_SAVE_ROOT}" "${LOG_ROOT}"

log() {
  echo "[$(date '+%F %T')] $*"
}

run_dir_for() {
  local label="$1"
  local seed="$2"
  local exp
  if [[ "${label}" == "A3" ]]; then
    exp="sciworld_A3_g2rl_normalized_action_gradient_3b_seed${seed}_100step_${RUN_TS}"
  else
    exp="sciworld_B0_strict_no_cluster_3b_seed${seed}_100step_${RUN_TS}"
  fi
  echo "${MODEL_SAVE_ROOT}/${exp}"
}

exp_name_for() {
  basename "$(run_dir_for "$1" "$2")"
}

require_fresh_or_complete() {
  local run_dir="$1"
  if [[ -d "${run_dir}" && ! -d "${run_dir}/global_step_${TOTAL_STEPS}" ]]; then
    echo "Refusing to overwrite incomplete run directory: ${run_dir}" >&2
    echo "Remove it, resume manually, or choose a new RUN_TS." >&2
    exit 1
  fi
}

run_train() {
  local label="$1"
  local seed="$2"
  local clustering_enabled="$3"
  local run_dir
  local exp_name
  local train_log

  run_dir="$(run_dir_for "${label}" "${seed}")"
  exp_name="$(exp_name_for "${label}" "${seed}")"
  train_log="${LOG_ROOT}/${exp_name}.log"
  require_fresh_or_complete "${run_dir}"

  if [[ -d "${run_dir}/global_step_${TOTAL_STEPS}" ]]; then
    log "skip training ${label} seed=${seed}: found ${run_dir}/global_step_${TOTAL_STEPS}"
    return
  fi

  log "training ${label} seed=${seed}: scratch -> ${run_dir} total_steps=${TOTAL_STEPS}"
  env \
    SCIWORLD_TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    SCIWORLD_N_GPUS_PER_NODE="${N_GPUS}" \
    SCIWORLD_ENV_SERVER_URL="${ENV_ADDR}" \
    SCIWORLD_MODEL_SAVE_PATH="${run_dir}" \
    SCIWORLD_EXP_NAME="${exp_name}" \
    SCIWORLD_TOTAL_TRAINING_STEPS="${TOTAL_STEPS}" \
    SCIWORLD_SAVE_FREQ="${SAVE_FREQ}" \
    SCIWORLD_RESUME_MODE=disable \
    SCIWORLD_DATA_SEED="${seed}" \
    SCIWORLD_ROLLOUT_SEED="${seed}" \
    SCIWORLD_CLUSTERING_ENABLED="${clustering_enabled}" \
    SCIWORLD_CLUSTERING_METHOD=g2rl_normalized_action_gradient \
    SCIWORLD_CLUSTERING_ACTION_NORMALIZER=sciworld \
    SCIWORLD_ROUND1_CANDIDATES=16 \
    SCIWORLD_ROUND1_CLUSTERS=8 \
    SCIWORLD_LATER_CANDIDATES=4 \
    SCIWORLD_LATER_CLUSTERS=1 \
    SCIWORLD_LATER_CLUSTER_EVERY=0 \
    bash "${REPO_ROOT}/examples/train/ScalingInter-RL/sciworld_train.sh" \
    2>&1 | tee "${train_log}"
  log "finished training ${label} seed=${seed}: log=${train_log}"
}

run_eval() {
  local label="$1"
  local seed="$2"
  local run_dir
  local exp_name
  local eval_log

  run_dir="$(run_dir_for "${label}" "${seed}")"
  exp_name="$(exp_name_for "${label}" "${seed}")"
  eval_log="${LOG_ROOT}/${exp_name}_eval.log"

  if [[ -f "${run_dir}/eval_sciworld_ckpt_sweep/results.csv" ]]; then
    local line_count
    line_count="$(wc -l < "${run_dir}/eval_sciworld_ckpt_sweep/results.csv")"
    if (( line_count >= 4 )); then
      log "skip eval ${label} seed=${seed}: found complete results.csv"
      return
    fi
  fi

  log "evaluating ${label} seed=${seed}: run_dir=${run_dir} checkpoints=${EVAL_CHECKPOINTS}"
  # shellcheck disable=SC2086
  env \
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    WANDB_MODE=offline \
    "${REPO_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/eval_sciworld_checkpoints.py" \
      --run-dir "${run_dir}" \
      --eval-data-dir "${PROJECT_ROOT}/AgentEval/sciworld/eval" \
      --env-addr "${ENV_ADDR}" \
      --checkpoints ${EVAL_CHECKPOINTS} \
      --n-gpus "${N_GPUS}" \
      --cuda-visible-devices "${TRAIN_GPUS}" \
      --wandb-name "${exp_name}_eval" \
    2>&1 | tee "${eval_log}"
  log "finished eval ${label} seed=${seed}: log=${eval_log}"
}

main() {
  log "SciWorld multiseed A3/B0 queue started"
  log "seeds=${SEEDS} total_steps=${TOTAL_STEPS} checkpoints=${EVAL_CHECKPOINTS}"
  log "env_addr=${ENV_ADDR} train_gpus=${TRAIN_GPUS} n_gpus=${N_GPUS}"
  log "model_save_root=${MODEL_SAVE_ROOT}"

  local seed
  for seed in ${SEEDS}; do
    log "seed ${seed} started"
    run_train A3 "${seed}" true
    run_train B0 "${seed}" false
    run_eval A3 "${seed}"
    run_eval B0 "${seed}"
    log "seed ${seed} completed"
  done

  log "SciWorld multiseed A3/B0 queue completed"
}

main "$@"
