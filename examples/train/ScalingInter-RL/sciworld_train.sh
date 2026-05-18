#!/usr/bin/env bash
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

train_visible_devices="${SCIWORLD_TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${train_visible_devices}"
n_gpus_per_node="${SCIWORLD_N_GPUS_PER_NODE:-$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")}"

# Local env server must bypass inherited proxies.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=COLL
export WANDB_MODE=offline

task_name="sciworld"
VENVPY="${VENVPY:-${REPO_ROOT}/.venv/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/AgentGym-RL}"

cd "${PROJECT_ROOT}"

env_server_url="${SCIWORLD_ENV_SERVER_URL:-http://127.0.0.1:36005}"

pure_agent_model_name="Qwen2.5-3B-Instruct"
shm_model_path="/dev/shm/${pure_agent_model_name}"
orig_model_path="models/${pure_agent_model_name}"
if [ ! -d "${shm_model_path}" ] || [ ! -f "${shm_model_path}/config.json" ]; then
  echo "Copying ${orig_model_path} -> ${shm_model_path} (first run)"
  cp -r "${orig_model_path}" "${shm_model_path}"
fi
agent_model_path="${shm_model_path}"

kl_coef="${SCIWORLD_KL_COEF:-0.001}"
policy_learning_rate="${SCIWORLD_POLICY_LR:-1e-6}"
rollout_sample_num="${SCIWORLD_ROLLOUT_SAMPLE_NUM:-8}"
train_batch_size="${SCIWORLD_TRAIN_BATCH_SIZE:-16}"
ppo_mini_batch_size="${SCIWORLD_PPO_MINI_BATCH_SIZE:-8}"
ppo_micro_batch_size_per_gpu="${SCIWORLD_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ppo_inner_epochs="${SCIWORLD_PPO_EPOCHS:-1}"
use_remove_padding="${SCIWORLD_USE_REMOVE_PADDING:-true}"
rollout_max_model_len="${SCIWORLD_ROLLOUT_MAX_MODEL_LEN:-32768}"
rollout_max_num_batched_tokens="${SCIWORLD_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-}"
rollout_gpu_memory_utilization="${SCIWORLD_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
ppo_max_token_len_per_gpu="${SCIWORLD_PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
entropy_chunk_size="${SCIWORLD_ENTROPY_CHUNK_SIZE:-256}"
max_prompt_length="${SCIWORLD_MAX_PROMPT_LENGTH:-1024}"
max_response_length="${SCIWORLD_MAX_RESPONSE_LENGTH:-8192}"
max_tokens="${SCIWORLD_MAX_TOKENS:-200}"
rounds="${SCIWORLD_ROUNDS:-[10,20,30]}"
steps_scaling_inter="${SCIWORLD_STEPS_SCALING_INTER:-100}"

total_epoches="${SCIWORLD_TOTAL_EPOCHS:-10}"
save_freq="${SCIWORLD_SAVE_FREQ:-25}"
total_training_steps="${SCIWORLD_TOTAL_TRAINING_STEPS:-}"

model_save_dir="saves"
mkdir -p "${model_save_dir}"
exp_name_prefix="${SCIWORLD_EXP_PREFIX:-sciworld_scalinginter_baseline_3b}"
exp_name="${exp_name_prefix}_$(date +%Y%m%d_%H%M)"
model_save_path="${model_save_dir}/${exp_name}"

mkdir -p "${model_save_path}"

echo "[baseline-run] task=${task_name} model=${pure_agent_model_name}"
echo "[baseline-run] train_gpus=${CUDA_VISIBLE_DEVICES} n_gpus_per_node=${n_gpus_per_node}"
echo "[baseline-run] rounds=${rounds} train_batch_size=${train_batch_size} rollout_n=${rollout_sample_num}"
echo "[baseline-run] max_prompt_length=${max_prompt_length} max_response_length=${max_response_length}"
echo "[baseline-run] use_remove_padding=${use_remove_padding} entropy_chunk_size=${entropy_chunk_size}"
echo "[baseline-run] gpu_memory_utilization=${rollout_gpu_memory_utilization} max_model_len=${rollout_max_model_len} max_num_batched_tokens=${rollout_max_num_batched_tokens:-default} ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} max_tokens=${max_tokens} tensor_model_parallel_size=1"
echo "[baseline-run] env_server_url=${env_server_url}"
if [ -n "${total_training_steps}" ]; then
  echo "[baseline-run] total_training_steps=${total_training_steps}"
fi
echo "[baseline-run] logs=${model_save_path}"

export VERL_ENTROPY_CHUNK_SIZE=${entropy_chunk_size}

cmd=(
  "${VENVPY}" -m verl.agent_trainer.main_ppo
  algorithm.adv_estimator=grpo
  algorithm.rounds_ctrl.type=scaling_inter_stepwise
  "algorithm.rounds_ctrl.steps_scaling_inter=${steps_scaling_inter}"
  "algorithm.rounds_ctrl.rounds=${rounds}"
  "data.train_file=AgentItemId/${task_name}_train.json"
  "data.train_batch_size=${train_batch_size}"
  "data.max_prompt_length=${max_prompt_length}"
  "data.max_response_length=${max_response_length}"
  "actor_rollout_ref.agentgym.task_name=${task_name}"
  "actor_rollout_ref.agentgym.env_addr=${env_server_url}"
  actor_rollout_ref.agentgym.timeout=600
  "actor_rollout_ref.model.path=${agent_model_path}"
  "actor_rollout_ref.model.use_remove_padding=${use_remove_padding}"
  actor_rollout_ref.actor.use_kl_loss=True
  "actor_rollout_ref.actor.kl_loss_coef=${kl_coef}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization}"
  "actor_rollout_ref.rollout.n=${rollout_sample_num}"
  "actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}"
  "actor_rollout_ref.rollout.max_tokens=${max_tokens}"
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  "actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs}"
  "actor_rollout_ref.actor.optim.lr=${policy_learning_rate}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}"
  actor_rollout_ref.actor.use_dynamic_bsz=True
  "actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs"
  "algorithm.kl_ctrl.kl_coef=${kl_coef}"
  "trainer.default_local_dir=${model_save_path}"
  trainer.project_name=agentgym-rl-baseline
  "trainer.experiment_name=${exp_name}"
  "trainer.save_freq=${save_freq}"
  "trainer.total_epochs=${total_epoches}"
  trainer.nnodes=1
  "trainer.n_gpus_per_node=${n_gpus_per_node}"
  "trainer.logger=[console,wandb]"
)

if [ -n "${total_training_steps}" ]; then
  cmd+=("trainer.total_training_steps=${total_training_steps}")
fi

if [ -n "${rollout_max_num_batched_tokens}" ]; then
  cmd+=("actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}")
fi

"${cmd[@]}"
status=$?
exit $status
