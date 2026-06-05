#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

BASELINE_SESSION="${BASELINE_SESSION:-textcraft_baseline_qwen25_1p5b_fastsettings_20260521_092959}"
BASELINE_EXP="${BASELINE_EXP:-textcraft_scalinginter_baseline_qwen25_1p5b_2xh200_fastsettings_20260521_092959}"
MODEL_PATH="${AGENT_MODEL_PATH:-${REPO_ROOT}/models/Qwen2.5-1.5B-Instruct}"
TEXTCRAFT_ENV_PORT="${TEXTCRAFT_ENV_PORT:-36005}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
TEXTCRAFT_N_GPUS_PER_NODE="${TEXTCRAFT_N_GPUS_PER_NODE:-2}"
POLL_SECONDS="${POLL_SECONDS:-60}"
POLL_LONG_WAIT_SECONDS="${POLL_LONG_WAIT_SECONDS:-1800}"
POLL_VERY_LONG_WAIT_SECONDS="${POLL_VERY_LONG_WAIT_SECONDS:-7200}"
POLL_MAX_SECONDS="${POLL_MAX_SECONDS:-600}"
GPU_WAIT_MAX_MEM_MB="${GPU_WAIT_MAX_MEM_MB:-2048}"
WATCH_LOG="${WATCH_LOG:-${REPO_ROOT}/logs/textcraft_g2rl_response_after_baseline_watch_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "${REPO_ROOT}/logs"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${WATCH_LOG}"
}

poll_sleep_seconds() {
    local started_at="$1"
    local now
    local elapsed
    local next_sleep

    now="$(date +%s)"
    elapsed=$((now - started_at))
    next_sleep="${POLL_SECONDS}"

    if (( elapsed >= POLL_LONG_WAIT_SECONDS )); then
        next_sleep=$((POLL_SECONDS * 2))
    fi
    if (( elapsed >= POLL_VERY_LONG_WAIT_SECONDS )); then
        next_sleep=$((POLL_SECONDS * 5))
    fi
    if (( next_sleep > POLL_MAX_SECONDS )); then
        next_sleep="${POLL_MAX_SECONDS}"
    fi
    if (( next_sleep < 1 )); then
        next_sleep=1
    fi

    echo "${next_sleep}"
}

gpu_max_mem_mb() {
    nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" \
        --query-gpu=memory.used \
        --format=csv,noheader,nounits \
        | awk 'BEGIN{max=0} {if ($1 > max) max=$1} END{print max+0}'
}

wait_for_gpu_idle() {
    local max_used
    local started_at
    local next_sleep
    started_at="$(date +%s)"
    while true; do
        max_used="$(gpu_max_mem_mb)"
        if (( max_used <= GPU_WAIT_MAX_MEM_MB )); then
            log "target GPUs idle enough: max_mem=${max_used} MiB <= ${GPU_WAIT_MAX_MEM_MB} MiB"
            return 0
        fi
        next_sleep="$(poll_sleep_seconds "${started_at}")"
        log "waiting for target GPUs to free: max_mem=${max_used} MiB > ${GPU_WAIT_MAX_MEM_MB} MiB next_poll=${next_sleep}s"
        sleep "${next_sleep}"
    done
}

wait_for_baseline_done() {
    local started_at
    local next_sleep
    started_at="$(date +%s)"
    while tmux has-session -t "${BASELINE_SESSION}" 2>/dev/null; do
        next_sleep="$(poll_sleep_seconds "${started_at}")"
        log "baseline tmux still running: ${BASELINE_SESSION} next_poll=${next_sleep}s"
        sleep "${next_sleep}"
    done
    log "baseline tmux ended: ${BASELINE_SESSION}"

    started_at="$(date +%s)"
    while ps -eo cmd | grep -F "${BASELINE_EXP}" | grep -v grep >/dev/null; do
        next_sleep="$(poll_sleep_seconds "${started_at}")"
        log "baseline process still present for exp=${BASELINE_EXP} next_poll=${next_sleep}s"
        sleep "${next_sleep}"
    done
    log "baseline process gone: ${BASELINE_EXP}"
}

ensure_textcraft_env() {
    if curl -fsS -X POST "http://127.0.0.1:${TEXTCRAFT_ENV_PORT}/create" \
        -H 'Content-Type: application/json' \
        -d '{"minecraft_dir":"agentenv_textcraft/","commands":null,"goal":null}' >/dev/null; then
        log "TextCraft env ready on port ${TEXTCRAFT_ENV_PORT}"
        return 0
    fi

    log "TextCraft env not ready; starting it on port ${TEXTCRAFT_ENV_PORT}"
    PORT="${TEXTCRAFT_ENV_PORT}" "${SCRIPT_DIR}/launch_textcraft_env.sh"
}

run_g2rl() {
    local run_log="$1"
    local exp_name="$2"

    mkdir -p "$(dirname "${run_log}")"
    exec > >(tee -a "${run_log}") 2>&1

    echo "[start] $(date '+%F %T')"
    echo "[exp] ${exp_name}"
    echo "[model] ${MODEL_PATH}"
    echo "[mode] textcraft paper-style g2rl response-scope"

    cd "${REPO_ROOT}"

    export CUDA_VISIBLE_DEVICES
    export TEXTCRAFT_N_GPUS_PER_NODE
    export TEXTCRAFT_ENV_PORT
    export TEXTCRAFT_AUTO_START_ENV=0
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export AGENT_MODEL_PATH="${MODEL_PATH}"
    export TEXTCRAFT_USE_SHM_MODEL=0
    export TEXTCRAFT_EXP_NAME="${exp_name}"

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
    export TEXTCRAFT_ROUNDS_CTRL_TYPE=scaling_inter_stepwise
    export TEXTCRAFT_ROUNDS_SCALING_INTER=100
    export TEXTCRAFT_ROUNDS_SCHEDULE='[10,20,30]'

    export TEXTCRAFT_G2RL_ENABLED=true
    export TEXTCRAFT_G2RL_FEATURE_SCOPE=response
    export TEXTCRAFT_G2RL_LAMBDA_COEF=1.0
    export TEXTCRAFT_G2RL_REWARD_CLIP=3.0
    export TEXTCRAFT_G2RL_ZERO_ONE_TO_SIGNED=true
    export TEXTCRAFT_G2RL_NORMALIZE_NOVELTY=true
    export TEXTCRAFT_G2RL_FEATURE_TOPK=256
    export TEXTCRAFT_G2RL_TOKEN_CHUNK_SIZE=512
    export TEXTCRAFT_CLUSTERING_ENABLED=false

    set +e
    "${SCRIPT_DIR}/textcraft_train.sh"
    local status=$?
    set -e
    echo "[exit] status=${status} time=$(date '+%F %T')"
    exit "${status}"
}

main() {
    if [[ "${1:-}" == "--run-g2rl" ]]; then
        run_g2rl "$2" "$3"
    fi

    log "watcher started"
    log "baseline_session=${BASELINE_SESSION}"
    log "baseline_exp=${BASELINE_EXP}"
    log "next_model=${MODEL_PATH}"
    log "next_g2rl_feature_scope=response"
    log "polling base=${POLL_SECONDS}s long_after=${POLL_LONG_WAIT_SECONDS}s very_long_after=${POLL_VERY_LONG_WAIT_SECONDS}s max=${POLL_MAX_SECONDS}s"

    wait_for_baseline_done
    wait_for_gpu_idle
    ensure_textcraft_env

    local run_ts
    local g2rl_session
    local g2rl_exp
    local g2rl_log
    run_ts="$(date +%Y%m%d_%H%M%S)"
    g2rl_session="${G2RL_SESSION:-textcraft_paper_g2rl_response_qwen25_1p5b_fastsettings_${run_ts}}"
    g2rl_exp="${G2RL_EXP:-textcraft_paper_g2rl_response_qwen25_1p5b_2xh200_fastsettings_${run_ts}}"
    g2rl_log="${G2RL_LOG:-${REPO_ROOT}/logs/textcraft_paper_g2rl_response_qwen25_1p5b_2xh200_fastsettings_${run_ts}.log}"

    log "launching next tmux session=${g2rl_session}"
    log "next_log=${g2rl_log}"
    log "next_exp=${g2rl_exp}"

    tmux new-session -d -s "${g2rl_session}" \
        "bash '${SCRIPT_PATH}' --run-g2rl '${g2rl_log}' '${g2rl_exp}'"

    log "launched next tmux session=${g2rl_session}"
}

main "$@"
