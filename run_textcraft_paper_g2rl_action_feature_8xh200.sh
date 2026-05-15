#!/usr/bin/env bash
set -euo pipefail

# Single-node 8xH200 launcher for the same TextCraft paper-style G2RL
# action-feature run that OOMed on 2xH200. Core rollout/training knobs are
# intentionally kept aligned with examples/train/AgentGym-RL/textcraft_train.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
TRAIN_SCRIPT="${REPO_ROOT}/examples/train/AgentGym-RL/textcraft_train.sh"

if [[ ! -x "${TRAIN_SCRIPT}" && ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[8xh200] ERROR: cannot find ${TRAIN_SCRIPT}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TEXTCRAFT_N_GPUS_PER_NODE="${TEXTCRAFT_N_GPUS_PER_NODE:-8}"

visible_gpu_count="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
if [[ "${visible_gpu_count}" != "${TEXTCRAFT_N_GPUS_PER_NODE}" ]]; then
    echo "[8xh200] ERROR: CUDA_VISIBLE_DEVICES has ${visible_gpu_count} GPUs, but TEXTCRAFT_N_GPUS_PER_NODE=${TEXTCRAFT_N_GPUS_PER_NODE}" >&2
    echo "[8xh200] Set both consistently, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TEXTCRAFT_N_GPUS_PER_NODE=8" >&2
    exit 1
fi

# NCCL defaults for a single H200 node. RDMA settings are best-effort and do not
# block local single-node training if show_gids or IB devices are unavailable.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -d /sys/class/infiniband ]]; then
    rdma_devices="$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ',' | sed 's/,$//')"
    if [[ -n "${rdma_devices}" ]]; then
        export NCCL_IB_HCA="${NCCL_IB_HCA:-${rdma_devices}}"
        export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
        if command -v show_gids >/dev/null 2>&1 && [[ -z "${NCCL_IB_GID_INDEX:-}" ]]; then
            gid_index="$(show_gids 2>/dev/null | awk '/v2/ && $5 ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$/ {print $3; exit}' || true)"
            if [[ -n "${gid_index}" ]]; then
                export NCCL_IB_GID_INDEX="${gid_index}"
            fi
        fi
    fi
fi

# Keep local TextCraft traffic off proxies while preserving any user-provided
# proxy settings for external services such as W&B.
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

# Defaults below mirror the failed 2xH200 run. Override explicitly only when
# running a new controlled experiment.
export AGENT_MODEL_PATH="${AGENT_MODEL_PATH:-${REPO_ROOT}/models/Qwen2.5-3B-Instruct}"
export TEXTCRAFT_ENV_PORT="${TEXTCRAFT_ENV_PORT:-36005}"
export TEXTCRAFT_AUTO_START_ENV="${TEXTCRAFT_AUTO_START_ENV:-1}"
export TEXTCRAFT_USE_SHM_MODEL="${TEXTCRAFT_USE_SHM_MODEL:-1}"

export TEXTCRAFT_G2RL_ENABLED="${TEXTCRAFT_G2RL_ENABLED:-true}"
export TEXTCRAFT_G2RL_FEATURE_SCOPE="${TEXTCRAFT_G2RL_FEATURE_SCOPE:-action}"
export TEXTCRAFT_G2RL_LAMBDA_COEF="${TEXTCRAFT_G2RL_LAMBDA_COEF:-1.0}"
export TEXTCRAFT_G2RL_REWARD_CLIP="${TEXTCRAFT_G2RL_REWARD_CLIP:-3.0}"
export TEXTCRAFT_G2RL_ZERO_ONE_TO_SIGNED="${TEXTCRAFT_G2RL_ZERO_ONE_TO_SIGNED:-true}"
export TEXTCRAFT_G2RL_NORMALIZE_NOVELTY="${TEXTCRAFT_G2RL_NORMALIZE_NOVELTY:-true}"
export TEXTCRAFT_G2RL_FEATURE_TOPK="${TEXTCRAFT_G2RL_FEATURE_TOPK:-256}"
export TEXTCRAFT_G2RL_TOKEN_CHUNK_SIZE="${TEXTCRAFT_G2RL_TOKEN_CHUNK_SIZE:-512}"

export TEXTCRAFT_CLUSTERING_ENABLED="${TEXTCRAFT_CLUSTERING_ENABLED:-false}"
export TEXTCRAFT_CLUSTERING_METHOD="${TEXTCRAFT_CLUSTERING_METHOD:-gradient_multiview}"
export TEXTCRAFT_ROUND1_CANDIDATES="${TEXTCRAFT_ROUND1_CANDIDATES:-64}"
export TEXTCRAFT_ROUND1_CLUSTERS="${TEXTCRAFT_ROUND1_CLUSTERS:-8}"
export TEXTCRAFT_LATER_CANDIDATES="${TEXTCRAFT_LATER_CANDIDATES:-16}"
export TEXTCRAFT_LATER_CLUSTERS="${TEXTCRAFT_LATER_CLUSTERS:-4}"
export TEXTCRAFT_LATER_CLUSTER_EVERY="${TEXTCRAFT_LATER_CLUSTER_EVERY:-2}"
export TEXTCRAFT_LATER_CLUSTER_START="${TEXTCRAFT_LATER_CLUSTER_START:-1}"
export TEXTCRAFT_LATER_CLUSTER_UNTIL="${TEXTCRAFT_LATER_CLUSTER_UNTIL:--1}"
export TEXTCRAFT_LATER_CLUSTER_HORIZON_MIN="${TEXTCRAFT_LATER_CLUSTER_HORIZON_MIN:-0.25}"
export TEXTCRAFT_GRADIENT_D_PROJ="${TEXTCRAFT_GRADIENT_D_PROJ:-512}"
export TEXTCRAFT_FEATURE_TOPK="${TEXTCRAFT_FEATURE_TOPK:-256}"
export TEXTCRAFT_FEATURE_CHUNK_SIZE="${TEXTCRAFT_FEATURE_CHUNK_SIZE:-4}"

export TEXTCRAFT_ROLLOUT_N="${TEXTCRAFT_ROLLOUT_N:-8}"
export TEXTCRAFT_TRAIN_BATCH_SIZE="${TEXTCRAFT_TRAIN_BATCH_SIZE:-32}"
export TEXTCRAFT_VAL_BATCH_SIZE="${TEXTCRAFT_VAL_BATCH_SIZE:-32}"
export TEXTCRAFT_PPO_MINI_BATCH_SIZE="${TEXTCRAFT_PPO_MINI_BATCH_SIZE:-8}"
export TEXTCRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU="${TEXTCRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
export TEXTCRAFT_PPO_EPOCHS="${TEXTCRAFT_PPO_EPOCHS:-2}"
export TEXTCRAFT_TOTAL_EPOCHS="${TEXTCRAFT_TOTAL_EPOCHS:-30}"

export TEXTCRAFT_MAX_PROMPT_LENGTH="${TEXTCRAFT_MAX_PROMPT_LENGTH:-512}"
export TEXTCRAFT_MAX_RESPONSE_LENGTH="${TEXTCRAFT_MAX_RESPONSE_LENGTH:-10240}"
export TEXTCRAFT_ROLLOUT_GPU_MEMORY_UTILIZATION="${TEXTCRAFT_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}"
export TEXTCRAFT_ROLLOUT_MAX_MODEL_LEN="${TEXTCRAFT_ROLLOUT_MAX_MODEL_LEN:-32768}"
export TEXTCRAFT_ROLLOUT_MAX_NUM_BATCHED_TOKENS="${TEXTCRAFT_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"
export TEXTCRAFT_ROLLOUT_MAX_TOKENS="${TEXTCRAFT_ROLLOUT_MAX_TOKENS:-512}"
export TEXTCRAFT_PPO_MAX_TOKEN_LEN_PER_GPU="${TEXTCRAFT_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
export TEXTCRAFT_ENTROPY_CHUNK_SIZE="${TEXTCRAFT_ENTROPY_CHUNK_SIZE:-256}"

export TEXTCRAFT_ROUNDS_CTRL_TYPE="${TEXTCRAFT_ROUNDS_CTRL_TYPE:-scaling_inter_stepwise}"
export TEXTCRAFT_ROUNDS_SCALING_INTER="${TEXTCRAFT_ROUNDS_SCALING_INTER:-100}"
export TEXTCRAFT_ROUNDS_SCHEDULE="${TEXTCRAFT_ROUNDS_SCHEDULE:-[10,20,30]}"
export TEXTCRAFT_SAVE_FREQ="${TEXTCRAFT_SAVE_FREQ:-25}"
export TEXTCRAFT_TEST_FREQ="${TEXTCRAFT_TEST_FREQ:-35}"
export TEXTCRAFT_TEST_BATCHES="${TEXTCRAFT_TEST_BATCHES:-1}"

timestamp="$(date +%Y%m%d_%H%M%S)"
export TEXTCRAFT_EXP_NAME="${TEXTCRAFT_EXP_NAME:-textcraft_paper_g2rl_action_feature_no_cluster_scalinginter_8xh200_${timestamp}}"

log_dir="${REPO_ROOT}/logs"
save_dir="${REPO_ROOT}/AgentGym-RL/saves/${TEXTCRAFT_EXP_NAME}"
mkdir -p "${log_dir}" "${save_dir}"
cp "$0" "${save_dir}/"
log_file="${log_dir}/${TEXTCRAFT_EXP_NAME}.log"

echo "[8xh200] repo=${REPO_ROOT}"
echo "[8xh200] exp=${TEXTCRAFT_EXP_NAME}"
echo "[8xh200] gpus=${CUDA_VISIBLE_DEVICES} n_gpus_per_node=${TEXTCRAFT_N_GPUS_PER_NODE}"
echo "[8xh200] batch=${TEXTCRAFT_TRAIN_BATCH_SIZE} rollout_n=${TEXTCRAFT_ROLLOUT_N} total_trajectories=$((TEXTCRAFT_TRAIN_BATCH_SIZE * TEXTCRAFT_ROLLOUT_N))"
echo "[8xh200] rollout_max_model_len=${TEXTCRAFT_ROLLOUT_MAX_MODEL_LEN} max_num_batched_tokens=${TEXTCRAFT_ROLLOUT_MAX_NUM_BATCHED_TOKENS} max_tokens=${TEXTCRAFT_ROLLOUT_MAX_TOKENS}"
echo "[8xh200] log=${log_file}"
echo "[8xh200] save_dir=${save_dir}"

cd "${REPO_ROOT}"
bash "${TRAIN_SCRIPT}" 2>&1 | tee -a "${log_file}"
