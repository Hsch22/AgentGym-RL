#!/usr/bin/env bash
set -x
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# Local env server must bypass any inherited proxies.
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

task_name="textcraft"
VENVPY=/share/project/husicheng/muhan/AgentGym-RL/.venv/bin/python
PROJECT_ROOT=/share/project/husicheng/muhan/AgentGym-RL/AgentGym-RL

cd "${PROJECT_ROOT}"

env_server_url="http://127.0.0.1:36005"

# Use the locally available actor model so the baseline can run on this machine.
pure_agent_model_name="Qwen2.5-3B-Instruct"
shm_model_path="/dev/shm/${pure_agent_model_name}"
orig_model_path="models/${pure_agent_model_name}"
if [ ! -d "${shm_model_path}" ] || [ ! -f "${shm_model_path}/config.json" ]; then
  echo "Copying ${orig_model_path} -> ${shm_model_path} (first run)"
  cp -r "${orig_model_path}" "${shm_model_path}"
fi
agent_model_path="${shm_model_path}"

kl_coef=0.001
policy_learning_rate=1e-6
rollout_sample_num=8
train_batch_size=32
ppo_mini_batch_size=8
ppo_micro_batch_size_per_gpu=1
ppo_inner_epochs=2
use_remove_padding=true
rollout_max_model_len=32768
ppo_max_token_len_per_gpu=16384
entropy_chunk_size=256

total_epoches=30

model_save_dir="saves"
mkdir -p ${model_save_dir}
exp_name="textcraft_scalinginter_baseline_3b_$(date +%Y%m%d_%H%M)"
model_save_path=${model_save_dir}/${exp_name}

mkdir -p ${model_save_path}

echo "[baseline-run] task=${task_name} model=${pure_agent_model_name}"
echo "[baseline-run] rounds=[10,20,30] train_batch_size=${train_batch_size} rollout_n=${rollout_sample_num}"
echo "[baseline-run] max_prompt_length=512 max_response_length=10240"
echo "[baseline-run] use_remove_padding=${use_remove_padding} entropy_chunk_size=${entropy_chunk_size}"
echo "[baseline-run] gpu_memory_utilization=0.7 max_model_len=${rollout_max_model_len} ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} tensor_model_parallel_size=1"
echo "[baseline-run] logs=${model_save_path}"

export VERL_ENTROPY_CHUNK_SIZE=${entropy_chunk_size}

$VENVPY -m verl.agent_trainer.main_ppo  \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=scaling_inter_stepwise \
    algorithm.rounds_ctrl.steps_scaling_inter=100 \
    algorithm.rounds_ctrl.rounds=[10,20,30] \
    data.train_file=AgentItemId/${task_name}_train.json \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=512 \
    data.max_response_length=10240 \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len} \
    actor_rollout_ref.rollout.max_tokens=512 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs} \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=agentgym-rl-baseline \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=25 \
    trainer.total_epochs=${total_epoches} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    "trainer.logger=[console,wandb]"
status=$?
exit $status
