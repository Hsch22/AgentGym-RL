#!/usr/bin/env bash
set -euo pipefail
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-COLL}"
export WANDB_MODE="${WANDB_MODE:-offline}"

VENVPY="${VENVPY:-${REPO_ROOT}/.venv/bin/python}"
if [[ -n "${TEXTCRAFT_TMPDIR:-}" ]]; then
    export TMPDIR="${TEXTCRAFT_TMPDIR}"
else
    mkdir -p "${REPO_ROOT}/.tmp"
    ln -sfn "${REPO_ROOT}/.tmp" /tmp/agentgym_rl_tmp
    export TMPDIR="/tmp/agentgym_rl_tmp"
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${REPO_ROOT}/.cache/triton}"
mkdir -p "${TMPDIR}" "${TRITON_CACHE_DIR}"

TEXTCRAFT_ENV_PORT="${TEXTCRAFT_ENV_PORT:-36005}"
TEXTCRAFT_AUTO_START_ENV="${TEXTCRAFT_AUTO_START_ENV:-1}"
n_gpus_per_node="${TEXTCRAFT_N_GPUS_PER_NODE:-$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")}"

task_name="textcraft"

cd "${REPO_ROOT}/AgentGym-RL"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.bandw.top}"

env_server_url="${TEXTCRAFT_ENV_SERVER_URL:-http://127.0.0.1:${TEXTCRAFT_ENV_PORT}}"

if [[ "${TEXTCRAFT_AUTO_START_ENV}" == "1" ]]; then
    PORT="${TEXTCRAFT_ENV_PORT}" "${SCRIPT_DIR}/launch_textcraft_env.sh"
fi

if [[ "${WANDB_MODE}" != "offline" && -n "${WANDB_API_KEY:-}" ]]; then
    "${VENVPY}" -m wandb login "${WANDB_API_KEY}"
fi

pure_agent_model_name="Qwen2.5-3B-Instruct"
orig_model_path="${AGENT_MODEL_PATH:-/mnt/workspace/users/mws/husicheng/code/AgentGym-RL/models/${pure_agent_model_name}}"
use_shm_model="${TEXTCRAFT_USE_SHM_MODEL:-1}"
if [[ "${use_shm_model}" == "1" ]]; then
    shm_model_path="/dev/shm/${pure_agent_model_name}"
    if [[ ! -d "${shm_model_path}" || ! -f "${shm_model_path}/config.json" ]]; then
        echo "[textcraft-train] copying ${orig_model_path} -> ${shm_model_path}"
        rm -rf "${shm_model_path}"
        cp -r "${orig_model_path}" "${shm_model_path}"
    fi
    agent_model_path="${shm_model_path}"
else
    agent_model_path="${orig_model_path}"
fi

kl_coef=0.001
policy_learning_rate=1e-6
rollout_sample_num="${TEXTCRAFT_ROLLOUT_N:-8}"
train_batch_size="${TEXTCRAFT_TRAIN_BATCH_SIZE:-32}"
val_batch_size="${TEXTCRAFT_VAL_BATCH_SIZE:-32}"
ppo_mini_batch_size="${TEXTCRAFT_PPO_MINI_BATCH_SIZE:-8}"
ppo_micro_batch_size_per_gpu="${TEXTCRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ppo_inner_epochs="${TEXTCRAFT_PPO_EPOCHS:-2}"
total_epoches="${TEXTCRAFT_TOTAL_EPOCHS:-30}"
max_prompt_length="${TEXTCRAFT_MAX_PROMPT_LENGTH:-512}"
max_response_length="${TEXTCRAFT_MAX_RESPONSE_LENGTH:-10240}"
use_remove_padding="${TEXTCRAFT_USE_REMOVE_PADDING:-true}"
rollout_gpu_memory_utilization="${TEXTCRAFT_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}"
rollout_max_model_len="${TEXTCRAFT_ROLLOUT_MAX_MODEL_LEN:-32768}"
rollout_max_num_batched_tokens="${TEXTCRAFT_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"
rollout_max_tokens="${TEXTCRAFT_ROLLOUT_MAX_TOKENS:-512}"
ppo_max_token_len_per_gpu="${TEXTCRAFT_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
entropy_chunk_size="${TEXTCRAFT_ENTROPY_CHUNK_SIZE:-256}"
save_freq="${TEXTCRAFT_SAVE_FREQ:-25}"
test_freq="${TEXTCRAFT_TEST_FREQ:-35}"
test_batches="${TEXTCRAFT_TEST_BATCHES:-1}"
rounds_ctrl_type="${TEXTCRAFT_ROUNDS_CTRL_TYPE:-scaling_inter_stepwise}"
rounds_scaling_inter="${TEXTCRAFT_ROUNDS_SCALING_INTER:-100}"
rounds_schedule="${TEXTCRAFT_ROUNDS_SCHEDULE:-[10,20,30]}"
export VERL_ENTROPY_CHUNK_SIZE="${entropy_chunk_size}"

# ==== paper-style G2RL reward shaping ====
# Default run keeps rollout sampling identical to the ScalingInter baseline and
# applies trajectory-level G2RL only before GRPO advantage computation.
g2rl_enabled="${TEXTCRAFT_G2RL_ENABLED:-true}"
g2rl_lambda_coef="${TEXTCRAFT_G2RL_LAMBDA_COEF:-1.0}"
g2rl_reward_clip="${TEXTCRAFT_G2RL_REWARD_CLIP:-3.0}"
g2rl_zero_one_to_signed="${TEXTCRAFT_G2RL_ZERO_ONE_TO_SIGNED:-true}"
g2rl_normalize_novelty="${TEXTCRAFT_G2RL_NORMALIZE_NOVELTY:-true}"
g2rl_feature_topk="${TEXTCRAFT_G2RL_FEATURE_TOPK:-256}"
g2rl_token_chunk_size="${TEXTCRAFT_G2RL_TOKEN_CHUNK_SIZE:-512}"

# ==== optional rollout-time clustering ====
# Keep disabled for the paper-style G2RL run. Enable explicitly when testing
# selection-based exploration on top of reward shaping.
clustering_enabled="${TEXTCRAFT_CLUSTERING_ENABLED:-false}"
clustering_method="${TEXTCRAFT_CLUSTERING_METHOD:-gradient_multiview}"   # "gradient", "gradient_multiview", "semantic", "random_valid", or "random_raw"
round1_candidates="${TEXTCRAFT_ROUND1_CANDIDATES:-64}"
round1_clusters="${TEXTCRAFT_ROUND1_CLUSTERS:-8}"
later_candidates="${TEXTCRAFT_LATER_CANDIDATES:-16}"
later_clusters="${TEXTCRAFT_LATER_CLUSTERS:-4}"
later_cluster_every="${TEXTCRAFT_LATER_CLUSTER_EVERY:-2}"
later_cluster_start="${TEXTCRAFT_LATER_CLUSTER_START:-1}"
later_cluster_until="${TEXTCRAFT_LATER_CLUSTER_UNTIL:--1}"
later_cluster_horizon_min="${TEXTCRAFT_LATER_CLUSTER_HORIZON_MIN:-0.25}"
gradient_d_proj="${TEXTCRAFT_GRADIENT_D_PROJ:-512}"
feature_topk="${TEXTCRAFT_FEATURE_TOPK:-256}"
feature_chunk_size="${TEXTCRAFT_FEATURE_CHUNK_SIZE:-4}"
gradient_model_path=${agent_model_path}

model_save_dir="${TEXTCRAFT_MODEL_SAVE_DIR:-saves}"
mkdir -p ${model_save_dir}
exp_name="${TEXTCRAFT_EXP_NAME:-textcraft_paper_g2rl_no_cluster_scalinginter_2xh200_$(date +%Y%m%d_%H%M)}"
model_save_path=${model_save_dir}/${exp_name}

mkdir -p ${model_save_path}

echo "[full-run] task=${task_name} model=${agent_model_path}"
echo "[full-run] train_gpus=${CUDA_VISIBLE_DEVICES} n_gpus_per_node=${n_gpus_per_node}"
echo "[full-run] rounds_ctrl_type=${rounds_ctrl_type} steps_scaling_inter=${rounds_scaling_inter} rounds=${rounds_schedule}"
echo "[full-run] train_batch_size=${train_batch_size} val_batch_size=${val_batch_size} rollout_n=${rollout_sample_num}"
echo "[full-run] total_trajectories_per_step=$((train_batch_size * rollout_sample_num)) per_gpu_trajectories=$((train_batch_size * rollout_sample_num / n_gpus_per_node))"
echo "[full-run] ppo_mini_batch_size=${ppo_mini_batch_size} ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}"
echo "[full-run] max_prompt_length=${max_prompt_length} max_response_length=${max_response_length} rollout_max_model_len=${rollout_max_model_len} rollout_max_num_batched_tokens=${rollout_max_num_batched_tokens} rollout_max_tokens=${rollout_max_tokens}"
echo "[full-run] use_remove_padding=${use_remove_padding} entropy_chunk_size=${entropy_chunk_size} rollout_gpu_memory_utilization=${rollout_gpu_memory_utilization}"
echo "[full-run] g2rl_enabled=${g2rl_enabled} lambda=${g2rl_lambda_coef} reward_clip=${g2rl_reward_clip} zero_one_to_signed=${g2rl_zero_one_to_signed} feature_topk=${g2rl_feature_topk} token_chunk_size=${g2rl_token_chunk_size}"
echo "[full-run] clustering=${clustering_method} enabled=${clustering_enabled} round1=${round1_candidates}/${round1_clusters} later=${later_candidates}/${later_clusters}"
echo "[full-run] later_cluster_schedule=every:${later_cluster_every},start:${later_cluster_start},until:${later_cluster_until},horizon_min:${later_cluster_horizon_min}"
echo "[full-run] save_freq=${save_freq} test_freq=${test_freq} test_batches=${test_batches} tmpdir=${TMPDIR} logs=${model_save_path}"

"${VENVPY}" -m verl.agent_trainer.main_ppo  \
    algorithm.adv_estimator=grpo \
    "algorithm.rounds_ctrl.type=${rounds_ctrl_type}" \
    algorithm.rounds_ctrl.steps_scaling_inter=${rounds_scaling_inter} \
    "algorithm.rounds_ctrl.rounds=${rounds_schedule}" \
    algorithm.g2rl.enabled=${g2rl_enabled} \
    algorithm.g2rl.lambda_coef=${g2rl_lambda_coef} \
    algorithm.g2rl.reward_clip=${g2rl_reward_clip} \
    algorithm.g2rl.zero_one_to_signed=${g2rl_zero_one_to_signed} \
    algorithm.g2rl.normalize_novelty=${g2rl_normalize_novelty} \
    algorithm.g2rl.feature_topk=${g2rl_feature_topk} \
    algorithm.g2rl.token_chunk_size=${g2rl_token_chunk_size} \
    data.train_file=AgentItemId/${task_name}_train.json \
    data.val_files=AgentEval/textcraft/eval/textcraft_test.json \
    data.train_batch_size=${train_batch_size} \
    data.val_batch_size=${val_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization} \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_tokens=${rollout_max_tokens} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs} \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    actor_rollout_ref.rollout.clustering.enabled=${clustering_enabled} \
    actor_rollout_ref.rollout.clustering.method=${clustering_method} \
    actor_rollout_ref.rollout.clustering.round1_candidates=${round1_candidates} \
    actor_rollout_ref.rollout.clustering.round1_clusters=${round1_clusters} \
    actor_rollout_ref.rollout.clustering.later_candidates=${later_candidates} \
    actor_rollout_ref.rollout.clustering.later_clusters=${later_clusters} \
    actor_rollout_ref.rollout.clustering.later_cluster_every=${later_cluster_every} \
    actor_rollout_ref.rollout.clustering.later_cluster_start=${later_cluster_start} \
    actor_rollout_ref.rollout.clustering.later_cluster_until=${later_cluster_until} \
    actor_rollout_ref.rollout.clustering.later_cluster_horizon_min=${later_cluster_horizon_min} \
    actor_rollout_ref.rollout.clustering.gradient_d_proj=${gradient_d_proj} \
    actor_rollout_ref.rollout.clustering.feature_topk=${feature_topk} \
    actor_rollout_ref.rollout.clustering.feature_chunk_size=${feature_chunk_size} \
    actor_rollout_ref.rollout.clustering.gradient_model_path=${gradient_model_path} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=agentgym-rl-textcraft \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.test_batches=${test_batches} \
    trainer.total_epochs=${total_epoches} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    "trainer.logger=[console,wandb]"
status=$?
exit $status
