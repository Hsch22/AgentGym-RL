#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/AgentGym-RL}"
VENVPY="${VENVPY:-${REPO_ROOT}/.venv/bin/python}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
SESSION_NAME="${SESSION_NAME:-textcraft_qwen25_0p5b_train_eval_queue_${RUN_TS}}"
MODEL_PATH="${AGENT_MODEL_PATH:-${REPO_ROOT}/models/Qwen2.5-0.5B-Instruct}"
PORT="${TEXTCRAFT_ENV_PORT:-36005}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS="${TEXTCRAFT_N_GPUS_PER_NODE:-2}"
EVAL_GPUS="${TEXTCRAFT_EVAL_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
EVAL_N_GPUS="${TEXTCRAFT_EVAL_N_GPUS:-${N_GPUS}}"
CHECKPOINTS="${CHECKPOINTS:-auto}"
START_STAGE="${START_STAGE:-baseline}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/textcraft_qwen25_0p5b_train_eval_queue_${RUN_TS}.log}"
OFFLINE_LIST="${OFFLINE_LIST:-${REPO_ROOT}/logs/textcraft_qwen25_0p5b_train_eval_queue_offline_runs_${RUN_TS}.txt}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${REPO_ROOT}/results/wandb_agentgym_rl_eval_textcraft_qwen25_0p5b_${RUN_TS}}"

BASELINE_EXP="${BASELINE_EXP:-textcraft_scalinginter_baseline_qwen25_0p5b_2xh200_fastsettings_${RUN_TS}}"
G2RL_EXP="${G2RL_EXP:-textcraft_paper_g2rl_response_qwen25_0p5b_2xh200_fastsettings_${RUN_TS}}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-${PROJECT_ROOT}/saves/${BASELINE_EXP}}"
G2RL_RUN_DIR="${G2RL_RUN_DIR:-${PROJECT_ROOT}/saves/${G2RL_EXP}}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${PROJECT_ROOT}/AgentEval/textcraft/eval}"

mkdir -p "${REPO_ROOT}/logs" "$(dirname "${ARCHIVE_ROOT}")"

latest_offline_run() {
    local root="$1"
    find "${root}/wandb" -maxdepth 1 -type d -name 'offline-run-*' -printf '%T@ %p\n' \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2-
}

discover_checkpoints() {
    local run_dir="$1"
    if [[ "${CHECKPOINTS}" != "auto" ]]; then
        echo "${CHECKPOINTS}"
        return 0
    fi

    find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' \
        | sed 's/^global_step_//' \
        | sort -n \
        | xargs echo
}

ensure_textcraft_env() {
    echo "[queue] checking TextCraft env http://127.0.0.1:${PORT}"
    if "${VENVPY}" - <<PY; then
import requests
try:
    ok = requests.post(
        "http://127.0.0.1:${PORT}/create",
        json={"minecraft_dir":"agentenv_textcraft/","commands":None,"goal":None},
        timeout=10,
    ).status_code < 500
except requests.RequestException:
    ok = False
raise SystemExit(0 if ok else 1)
PY
        echo "[queue] TextCraft env ready"
        return 0
    fi

    echo "[queue] TextCraft env not ready; starting server on port ${PORT}"
    PORT="${PORT}" "${SCRIPT_DIR}/launch_textcraft_env.sh"
}

common_train_env() {
    export CUDA_VISIBLE_DEVICES
    export TEXTCRAFT_N_GPUS_PER_NODE="${N_GPUS}"
    export TEXTCRAFT_ENV_PORT="${PORT}"
    export TEXTCRAFT_AUTO_START_ENV=0
    export WANDB_MODE=offline
    export AGENT_MODEL_PATH="${MODEL_PATH}"
    export TEXTCRAFT_USE_SHM_MODEL=0

    export TEXTCRAFT_TRAIN_BATCH_SIZE=32
    export TEXTCRAFT_ROLLOUT_N=8
    export TEXTCRAFT_VAL_BATCH_SIZE=32
    export TEXTCRAFT_PPO_MINI_BATCH_SIZE=8
    export TEXTCRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU=1
    export TEXTCRAFT_PPO_EPOCHS=2
    export TEXTCRAFT_TOTAL_EPOCHS=30
    export TEXTCRAFT_MAX_PROMPT_LENGTH=512
    export TEXTCRAFT_MAX_RESPONSE_LENGTH=10240
    export TEXTCRAFT_USE_REMOVE_PADDING=true
    export TEXTCRAFT_ROLLOUT_GPU_MEMORY_UTILIZATION=0.8
    export TEXTCRAFT_ROLLOUT_MAX_MODEL_LEN=32768
    export TEXTCRAFT_ROLLOUT_MAX_NUM_BATCHED_TOKENS=16384
    export TEXTCRAFT_ROLLOUT_MAX_TOKENS=512
    export TEXTCRAFT_PPO_MAX_TOKEN_LEN_PER_GPU=32768
    export TEXTCRAFT_ENTROPY_CHUNK_SIZE=256
    export TEXTCRAFT_SAVE_FREQ=25
    export TEXTCRAFT_TEST_FREQ=35
    export TEXTCRAFT_TEST_BATCHES=1
    export TEXTCRAFT_EARLY_STOP_MIN_STEPS="${TEXTCRAFT_EARLY_STOP_MIN_STEPS:-100}"
    export TEXTCRAFT_ROUNDS_CTRL_TYPE=scaling_inter_stepwise
    export TEXTCRAFT_ROUNDS_SCALING_INTER=100
    export TEXTCRAFT_ROUNDS_SCHEDULE='[10,20,30]'
    export TEXTCRAFT_CLUSTERING_ENABLED=false
}

run_train() {
    local label="$1"
    local exp_name="$2"
    local g2rl_enabled="$3"
    local g2rl_scope="$4"
    local before_file
    local after_file
    local offline_run

    before_file="$(mktemp)"
    after_file="$(mktemp)"
    find "${PROJECT_ROOT}/wandb" -maxdepth 1 -type d -name 'offline-run-*' -print | sort > "${before_file}" || true

    echo "[queue] train ${label} start $(date '+%F %T')"
    common_train_env
    export TEXTCRAFT_EXP_NAME="${exp_name}"
    export TEXTCRAFT_G2RL_ENABLED="${g2rl_enabled}"
    export TEXTCRAFT_G2RL_FEATURE_SCOPE="${g2rl_scope}"
    export TEXTCRAFT_G2RL_LAMBDA_COEF=1.0
    export TEXTCRAFT_G2RL_REWARD_CLIP=3.0
    export TEXTCRAFT_G2RL_ZERO_ONE_TO_SIGNED=true
    export TEXTCRAFT_G2RL_NORMALIZE_NOVELTY=true
    export TEXTCRAFT_G2RL_FEATURE_TOPK=256
    export TEXTCRAFT_G2RL_TOKEN_CHUNK_SIZE=512

    "${SCRIPT_DIR}/textcraft_train.sh"

    find "${PROJECT_ROOT}/wandb" -maxdepth 1 -type d -name 'offline-run-*' -print | sort > "${after_file}" || true
    offline_run="$(comm -13 "${before_file}" "${after_file}" | tail -n 1)"
    if [[ -z "${offline_run}" ]]; then
        offline_run="$(latest_offline_run "${PROJECT_ROOT}")"
    fi
    rm -f "${before_file}" "${after_file}"
    echo "train_${label} ${offline_run}" | tee -a "${OFFLINE_LIST}"
    echo "[queue] train ${label} done $(date '+%F %T') offline_run=${offline_run}"
}

run_eval() {
    local ordinal="$1"
    local label="$2"
    local run_dir="$3"
    local wandb_name="$4"
    local before_file
    local after_file
    local offline_run
    local run_id
    local checkpoints

    before_file="$(mktemp)"
    after_file="$(mktemp)"
    find "${PROJECT_ROOT}/wandb" -maxdepth 1 -type d -name 'offline-run-*' -print | sort > "${before_file}" || true

    checkpoints="$(discover_checkpoints "${run_dir}")"
    if [[ -z "${checkpoints}" ]]; then
        echo "[queue] ERROR: no saved checkpoints found under ${run_dir}" >&2
        return 1
    fi
    echo "[queue] eval ${label} start $(date '+%F %T') run_dir=${run_dir}"
    echo "[queue] eval ${label} checkpoints=${checkpoints}"
    cd "${PROJECT_ROOT}"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    export no_proxy="127.0.0.1,localhost"
    export NO_PROXY="127.0.0.1,localhost"
    export CUDA_VISIBLE_DEVICES="${EVAL_GPUS}"
    export VLLM_USE_MODELSCOPE=0
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_ATTENTION_BACKEND=XFORMERS
    export HYDRA_FULL_ERROR=1
    export WANDB_MODE=offline

    "${VENVPY}" "${PROJECT_ROOT}/scripts/eval_textcraft_checkpoints.py" \
        --run-dir "${run_dir}" \
        --eval-data-dir "${EVAL_DATA_DIR}" \
        --env-addr "http://127.0.0.1:${PORT}" \
        --checkpoints ${checkpoints} \
        --n-gpus "${EVAL_N_GPUS}" \
        --wandb-project agentgym-rl-eval \
        --wandb-name "${wandb_name}" \
        --wandb-mode offline

    find "${PROJECT_ROOT}/wandb" -maxdepth 1 -type d -name 'offline-run-*' -print | sort > "${after_file}" || true
    offline_run="$(comm -13 "${before_file}" "${after_file}" | tail -n 1)"
    if [[ -z "${offline_run}" ]]; then
        offline_run="$(latest_offline_run "${PROJECT_ROOT}")"
    fi
    rm -f "${before_file}" "${after_file}"
    echo "eval_${label} ${offline_run}" | tee -a "${OFFLINE_LIST}"

    run_id="$(basename "${offline_run}" | awk -F- '{print $NF}')"
    "${VENVPY}" "${PROJECT_ROOT}/scripts/archive_eval_sweep_to_wandb_export.py" \
        --archive-root "${ARCHIVE_ROOT}" \
        --ordinal "${ordinal}" \
        --run-id "${run_id}" \
        --run-name "${wandb_name}" \
        --run-dir "${run_dir}" \
        --eval-data-dir "${EVAL_DATA_DIR}" \
        --env-addr "http://127.0.0.1:${PORT}" \
        --n-gpus "${EVAL_N_GPUS}" \
        --offline-run-dir "${offline_run}" \
        --results-csv "${run_dir}/eval_textcraft_ckpt_sweep/results.csv" \
        --results-json "${run_dir}/eval_textcraft_ckpt_sweep/results.json"

    echo "[queue] eval ${label} done $(date '+%F %T') offline_run=${offline_run}"
}

run_worker() {
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "[queue] start $(date '+%F %T')"
    echo "[queue] session=${SESSION_NAME}"
    echo "[queue] model=${MODEL_PATH}"
    echo "[queue] baseline_exp=${BASELINE_EXP}"
    echo "[queue] g2rl_exp=${G2RL_EXP}"
    echo "[queue] checkpoints=${CHECKPOINTS}"
    echo "[queue] start_stage=${START_STAGE}"
    echo "[queue] log=${LOG_FILE}"
    echo "[queue] offline_list=${OFFLINE_LIST}"
    echo "[queue] archive_root=${ARCHIVE_ROOT}"

    ensure_textcraft_env

    if [[ "${START_STAGE}" == "baseline" ]]; then
        cd "${REPO_ROOT}"
        run_train baseline "${BASELINE_EXP}" false action
        run_eval 1 baseline "${BASELINE_RUN_DIR}" "textcraft_eval_qwen25_0p5b_baseline_2xh200_fastsettings_${RUN_TS}"
    elif [[ "${START_STAGE}" != "g2rl_response" ]]; then
        echo "[queue] ERROR: START_STAGE must be baseline or g2rl_response, got ${START_STAGE}" >&2
        return 1
    fi

    cd "${REPO_ROOT}"
    run_train g2rl_response "${G2RL_EXP}" true response
    run_eval 2 g2rl_response "${G2RL_RUN_DIR}" "textcraft_eval_qwen25_0p5b_g2rl_response_2xh200_fastsettings_${RUN_TS}"

    echo "[queue] finished $(date '+%F %T')"
}

if [[ "${1:-}" == "--worker" ]]; then
    run_worker
    exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" \
    "RUN_TS='${RUN_TS}' SESSION_NAME='${SESSION_NAME}' LOG_FILE='${LOG_FILE}' OFFLINE_LIST='${OFFLINE_LIST}' ARCHIVE_ROOT='${ARCHIVE_ROOT}' PROJECT_ROOT='${PROJECT_ROOT}' VENVPY='${VENVPY}' AGENT_MODEL_PATH='${MODEL_PATH}' TEXTCRAFT_ENV_PORT='${PORT}' CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' TEXTCRAFT_N_GPUS_PER_NODE='${N_GPUS}' TEXTCRAFT_EVAL_CUDA_VISIBLE_DEVICES='${EVAL_GPUS}' TEXTCRAFT_EVAL_N_GPUS='${EVAL_N_GPUS}' CHECKPOINTS='${CHECKPOINTS}' START_STAGE='${START_STAGE}' BASELINE_EXP='${BASELINE_EXP}' G2RL_EXP='${G2RL_EXP}' BASELINE_RUN_DIR='${BASELINE_RUN_DIR}' G2RL_RUN_DIR='${G2RL_RUN_DIR}' EVAL_DATA_DIR='${EVAL_DATA_DIR}' bash '${SCRIPT_PATH}' --worker"

echo "Started tmux session: ${SESSION_NAME}"
echo "Queue log: ${LOG_FILE}"
echo "Offline run list: ${OFFLINE_LIST}"
echo "Eval archive root: ${ARCHIVE_ROOT}"
echo "Baseline save dir: ${BASELINE_RUN_DIR}"
echo "G2RL save dir: ${G2RL_RUN_DIR}"
