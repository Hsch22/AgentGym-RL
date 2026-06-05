#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
TOTAL_STEPS="${SCIWORLD_LONG_TOTAL_STEPS:-100}"
SAVE_FREQ="${SCIWORLD_LONG_SAVE_FREQ:-25}"
PORT="${SCIWORLD_LONG_ENV_PORT:-36006}"
ENV_ADDR="${SCIWORLD_LONG_ENV_ADDR:-http://127.0.0.1:${PORT}}"
TRAIN_GPUS="${SCIWORLD_LONG_TRAIN_GPUS:-0,1}"
N_GPUS="${SCIWORLD_LONG_N_GPUS:-2}"
MODEL_SAVE_ROOT="${SCIWORLD_LONG_MODEL_SAVE_ROOT:-${REPO_ROOT}/results/sciworld}"
LOG_ROOT="${SCIWORLD_LONG_LOG_ROOT:-${REPO_ROOT}/logs}"
EVAL_CHECKPOINTS="${SCIWORLD_LONG_EVAL_CHECKPOINTS:-50 75 100}"

A3_SOURCE_RUN="${SCIWORLD_LONG_A3_SOURCE_RUN:-${REPO_ROOT}/results/sciworld/sciworld_A3_g2rl_normalized_action_gradient_3b_25step_20260530_20260529_2355}"
B0_SOURCE_RUN="${SCIWORLD_LONG_B0_SOURCE_RUN:-${REPO_ROOT}/results/sciworld/sciworld_B0_strict_no_cluster_3b_25step_20260530_20260530_0409}"

A3_EXP="${SCIWORLD_LONG_A3_EXP:-sciworld_A3_g2rl_normalized_action_gradient_3b_100step_continue_from25_${RUN_TS}}"
B0_EXP="${SCIWORLD_LONG_B0_EXP:-sciworld_B0_strict_no_cluster_3b_100step_continue_from25_${RUN_TS}}"
A3_RUN_DIR="${SCIWORLD_LONG_A3_RUN_DIR:-${MODEL_SAVE_ROOT}/${A3_EXP}}"
B0_RUN_DIR="${SCIWORLD_LONG_B0_RUN_DIR:-${MODEL_SAVE_ROOT}/${B0_EXP}}"

mkdir -p "${MODEL_SAVE_ROOT}" "${LOG_ROOT}"

log() {
  echo "[$(date '+%F %T')] $*"
}

require_checkpoint() {
  local run_dir="$1"
  local ckpt="${run_dir}/global_step_25"
  if [[ ! -f "${ckpt}/trainer_state.pt" || ! -d "${ckpt}/actor" ]]; then
    echo "Missing resumable checkpoint: ${ckpt}" >&2
    exit 1
  fi
}

run_train() {
  local label="$1"
  local source_run="$2"
  local target_run="$3"
  local exp_name="$4"
  local clustering_enabled="$5"
  local train_log="${LOG_ROOT}/${exp_name}.log"

  require_checkpoint "${source_run}"
  log "training ${label}: ${source_run}/global_step_25 -> ${target_run} total_steps=${TOTAL_STEPS}"
  env \
    SCIWORLD_TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    SCIWORLD_N_GPUS_PER_NODE="${N_GPUS}" \
    SCIWORLD_ENV_SERVER_URL="${ENV_ADDR}" \
    SCIWORLD_MODEL_SAVE_PATH="${target_run}" \
    SCIWORLD_EXP_NAME="${exp_name}" \
    SCIWORLD_TOTAL_TRAINING_STEPS="${TOTAL_STEPS}" \
    SCIWORLD_SAVE_FREQ="${SAVE_FREQ}" \
    SCIWORLD_RESUME_MODE="${source_run}/global_step_25" \
    SCIWORLD_RESUME_ALLOW_EXTEND_TOTAL_TRAINING_STEPS=true \
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
  log "finished ${label}: log=${train_log}"
}

run_eval() {
  local label="$1"
  local run_dir="$2"
  local exp_name="$3"
  local eval_log="${LOG_ROOT}/${exp_name}_eval.log"

  log "evaluating ${label}: run_dir=${run_dir} checkpoints=${EVAL_CHECKPOINTS}"
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
  log "finished eval ${label}: log=${eval_log}"
}

main() {
  log "SciWorld long A3/B0 queue started"
  log "env_addr=${ENV_ADDR} train_gpus=${TRAIN_GPUS} n_gpus=${N_GPUS}"
  log "a3_target=${A3_RUN_DIR}"
  log "b0_target=${B0_RUN_DIR}"

  run_train A3 "${A3_SOURCE_RUN}" "${A3_RUN_DIR}" "${A3_EXP}" true
  run_train B0 "${B0_SOURCE_RUN}" "${B0_RUN_DIR}" "${B0_EXP}" false

  run_eval A3 "${A3_RUN_DIR}" "${A3_EXP}"
  run_eval B0 "${B0_RUN_DIR}" "${B0_EXP}"

  log "SciWorld long A3/B0 queue completed"
}

main "$@"
