#!/usr/bin/env bash
# 8-GPU random raw rollout training for TextCraft.
# Keeps the expanded candidate rollout shape while replacing semantic clustering
# with random selection over raw generated responses.
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# Bypass any inherited http(s)_proxy so localhost env server calls don't route through a dead proxy
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Immediate stop-loss: NCCL timeout shortened to 30 min so hangs don't eat an hour,
# plus debug traces so SIGABRT reveals which collective died.
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=COLL

# Wandb - offline so the mirror / outbound HTTPS is not on the critical path;
# user can `wandb sync saves/${exp_name}/wandb/offline-run-*` later.
export WANDB_MODE=offline
# WANDB_API_KEY is expected to be set in the environment before running (still used if sync later)

VENVPY="${VENVPY:-${REPO_ROOT}/.venv/bin/python}"
task_name="textcraft"

cd "${REPO_ROOT}/AgentGym-RL"

env_server_url="http://127.0.0.1:36005"

pure_agent_model_name="Qwen2.5-3B-Instruct"
# Use /dev/shm (tmpfs) to avoid concurrent checkpoint-load contention on shared filesystems
# from 8 FSDP workers (one rank stuck at 50% caused an NCCL watchdog timeout last run).
shm_model_path="/dev/shm/${pure_agent_model_name}"
orig_model_path="models/${pure_agent_model_name}"
if [ ! -d "${shm_model_path}" ] || [ ! -f "${shm_model_path}/config.json" ]; then
  echo "Copying ${orig_model_path} -> ${shm_model_path} (first run)"
  cp -r "${orig_model_path}" "${shm_model_path}"
fi
agent_model_path="${shm_model_path}"

# training hyperparams
kl_coef=0.001
policy_learning_rate=1e-6
rollout_sample_num=8       # parallel trajectories per prompt (== round1_clusters)
train_batch_size=32        # ScalingInter baseline: 32 prompts/step (4 per GPU on 8-GPU)
val_batch_size=32
test_freq=35
test_batches=1
ppo_mini_batch_size=8
ppo_micro_batch_size_per_gpu=1
ppo_inner_epochs=2
total_epoches=30           # ScalingInter baseline
max_prompt_length=512
max_response_length=10240  # restore baseline trajectory budget
use_remove_padding=true
rollout_max_model_len=32768
ppo_max_token_len_per_gpu=16384
entropy_chunk_size=256
export VERL_ENTROPY_CHUNK_SIZE=${entropy_chunk_size}
rounds_ctrl_type="scaling_inter_stepwise"
rounds_scaling_inter=100
rounds_schedule='[10,20,30]'

# clustering (random raw debug mode)
clustering_enabled=true
clustering_method="random_raw"
round1_candidates=64       # 8× round1_clusters (kept ~8× ratio from plan)
round1_clusters=8          # must equal rollout_sample_num
later_candidates=9
later_clusters=1
gradient_d_proj=512
gradient_model_path=${agent_model_path}

model_save_dir="saves"
exp_name="tc_random_raw_rmpad_l9c1_$(date +%Y%m%d_%H%M)"
model_save_path=${model_save_dir}/${exp_name}
mkdir -p ${model_save_path}

echo "[run-config] task=${task_name} mode=random-raw baseline-aligned"
echo "[run-config] rounds_ctrl_type=${rounds_ctrl_type} rounds=${rounds_schedule} steps_scaling_inter=${rounds_scaling_inter}"
echo "[run-config] max_prompt_length=${max_prompt_length} max_response_length=${max_response_length}"
echo "[run-config] use_remove_padding=${use_remove_padding} rollout_max_model_len=${rollout_max_model_len} ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu} entropy_chunk_size=${entropy_chunk_size}"
echo "[run-config] train_batch_size=${train_batch_size} rollout_n=${rollout_sample_num} ppo_mini_batch_size=${ppo_mini_batch_size} ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}"
echo "[run-config] val_batch_size=${val_batch_size} test_freq=${test_freq} test_batches=${test_batches}"
echo "[run-config] clustering=${clustering_method} round1_candidates=${round1_candidates} round1_clusters=${round1_clusters} later_candidates=${later_candidates} later_clusters=${later_clusters}"
echo "[run-config] logs=${model_save_path}"

$VENVPY -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=${rounds_ctrl_type} \
    algorithm.rounds_ctrl.steps_scaling_inter=${rounds_scaling_inter} \
    algorithm.rounds_ctrl.rounds=${rounds_schedule} \
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
    trainer.project_name=agentgym-rl-random-raw \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=25 \
    trainer.test_freq=${test_freq} \
    trainer.test_batches=${test_batches} \
    trainer.total_epochs=${total_epoches} \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    "trainer.logger=[console,wandb]"
status=$?
exit $status
