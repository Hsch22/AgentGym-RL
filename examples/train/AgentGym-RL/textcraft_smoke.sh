#!/usr/bin/env bash
# Level 2 smoke test: minimal single-GPU run to validate clustering rollout.
# Run from /share/project/husicheng/muhan/AgentGym-RL
set -x
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use the uv venv python (skip conda activate)
VENVPY=/share/project/husicheng/muhan/AgentGym-RL/.venv/bin/python

task_name="textcraft"

cd AgentGym-RL

env_server_url="http://127.0.0.1:36005"

# Use smaller 3B model for smoke test (only 3B available locally)
pure_agent_model_name="Qwen2.5-3B-Instruct"
agent_model_path="models/${pure_agent_model_name}"

# Minimal hyperparams
kl_coef=0.001
policy_learning_rate=1e-6
rollout_sample_num=2         # parallel trajectories (== round1_clusters)
train_batch_size=1            # minimum
ppo_mini_batch_size=1
ppo_micro_batch_size_per_gpu=1
ppo_inner_epochs=1
total_epoches=1

# Clustering smoke params: semantic (no extra gradient model)
clustering_enabled=true
clustering_method="semantic"
round1_candidates=8
round1_clusters=2            # must equal rollout_sample_num
later_candidates=4
later_clusters=2
gradient_d_proj=256
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
    actor_rollout_ref.rollout.clustering.gradient_model_path=${gradient_model_path} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=smoke \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=-1 \
    trainer.total_epochs=${total_epoches} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    trainer.logger='[console]'
status=$?
exit $status
