#!/usr/bin/env bash
# Level 2 smoke test: minimal single-GPU run to validate clustering rollout.
set -euo pipefail
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"

# Use the uv venv python (skip conda activate)
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

task_name="textcraft"

cd "${REPO_ROOT}/AgentGym-RL"

env_server_url="${TEXTCRAFT_ENV_SERVER_URL:-http://127.0.0.1:${TEXTCRAFT_ENV_PORT}}"

if [[ "${TEXTCRAFT_AUTO_START_ENV}" == "1" ]]; then
    PORT="${TEXTCRAFT_ENV_PORT}" "${SCRIPT_DIR}/launch_textcraft_env.sh"
fi

# Use local 3B model for smoke test.
agent_model_path="${AGENT_MODEL_PATH:-/mnt/workspace/users/mws/husicheng/code/AgentGym-RL/models/Qwen2.5-3B-Instruct}"

# Minimal hyperparams
kl_coef=0.001
policy_learning_rate=1e-6
rollout_sample_num=2         # parallel trajectories (== round1_clusters)
train_batch_size=2            # one prompt per local GPU
ppo_mini_batch_size=2
ppo_micro_batch_size_per_gpu=1
ppo_inner_epochs=1
total_epoches=1
total_training_steps="${TEXTCRAFT_TOTAL_TRAINING_STEPS:-1}"
save_freq="${TEXTCRAFT_SAVE_FREQ:-1}"
resume_mode="${TEXTCRAFT_RESUME_MODE:-disable}"

# Clustering smoke params: semantic (no extra gradient model)
clustering_enabled=true
clustering_method="semantic"
round1_candidates=8
round1_clusters=2            # must equal rollout_sample_num
later_candidates=4
later_clusters=2
gradient_d_proj=256
feature_topk=256
feature_chunk_size=4
gradient_model_path=${agent_model_path}

model_save_dir="saves"
exp_name="smoke"
model_save_path=${model_save_dir}/${exp_name}
mkdir -p ${model_save_path}

$VENVPY -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=3 \
    data.train_file=AgentItemId/${task_name}_train.json \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_tokens=256 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs} \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    actor_rollout_ref.rollout.clustering.enabled=${clustering_enabled} \
    actor_rollout_ref.rollout.clustering.method=${clustering_method} \
    actor_rollout_ref.rollout.clustering.round1_candidates=${round1_candidates} \
    actor_rollout_ref.rollout.clustering.round1_clusters=${round1_clusters} \
    actor_rollout_ref.rollout.clustering.later_candidates=${later_candidates} \
    actor_rollout_ref.rollout.clustering.later_clusters=${later_clusters} \
    actor_rollout_ref.rollout.clustering.gradient_d_proj=${gradient_d_proj} \
    actor_rollout_ref.rollout.clustering.feature_topk=${feature_topk} \
    actor_rollout_ref.rollout.clustering.feature_chunk_size=${feature_chunk_size} \
    actor_rollout_ref.rollout.clustering.gradient_model_path=${gradient_model_path} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=smoke \
    trainer.experiment_name=${exp_name} \
    trainer.resume_mode=${resume_mode} \
    trainer.save_freq=${save_freq} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.total_epochs=${total_epoches} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    trainer.logger='[console]'
status=$?
exit $status
