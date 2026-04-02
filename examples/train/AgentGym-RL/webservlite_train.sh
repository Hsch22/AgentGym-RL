set -euo pipefail
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=1

task_name="webservlite"

cd AgentGym-RL
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentgym-rl-clean
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_BASE_URL=https://api.bandw.top

env_server_url="${ENV_SERVER_URL:-http://127.0.0.1:36006}"
pretty_log="${PRETTY_LOG:-1}"
debug_train_script="${DEBUG_TRAIN_SCRIPT:-0}"

if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "${WANDB_API_KEY}"
fi

pure_agent_model_name="${MODEL_NAME:-Qwen2.5-1.5B-Instruct}"
agent_model_path="${MODEL_PATH:-models/${pure_agent_model_name}}"

if [ ! -f "${agent_model_path}/config.json" ]; then
    echo "Model config not found: ${agent_model_path}/config.json"
    echo "Set MODEL_PATH to an existing local model directory, or place ${pure_agent_model_name} under AgentGym-RL/models/."
    exit 1
fi

model_type="$(python - <<'PY' "${agent_model_path}/config.json"
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    config = json.load(f)

print(config.get("model_type", "unknown"))
PY
)"

# Multi-GPU formal training defaults. Override CUDA_VISIBLE_DEVICES if you want
# to pin training to a different GPU set, e.g. "export CUDA_VISIBLE_DEVICES=0,5,6,7".
gpu_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${gpu_devices}"
IFS=',' read -r -a gpu_array <<< "${gpu_devices}"
visible_gpu_count="${#gpu_array[@]}"

kl_coef=0.001
policy_learning_rate=1e-6
rollout_sample_num="${ROLLOUT_SAMPLE_NUM:-2}"
max_rounds="${MAX_ROUNDS:-10}"
train_batch_size="${TRAIN_BATCH_SIZE:-4}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-${train_batch_size}}"
ppo_micro_batch_size_per_gpu=1
ppo_inner_epochs="${PPO_INNER_EPOCHS:-2}"
data_max_prompt_length="${DATA_MAX_PROMPT_LENGTH:-700}"
data_max_response_length="${DATA_MAX_RESPONSE_LENGTH:-512}"
if [ -n "${USE_REMOVE_PADDING:-}" ]; then
    use_remove_padding="${USE_REMOVE_PADDING}"
elif [ "${model_type}" = "qwen3" ]; then
    use_remove_padding=0
else
    use_remove_padding=1
fi
actor_param_offload="${ACTOR_PARAM_OFFLOAD:-0}"
actor_grad_offload="${ACTOR_GRAD_OFFLOAD:-1}"
actor_optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-1}"
ref_param_offload="${REF_PARAM_OFFLOAD:-1}"
critic_param_offload="${CRITIC_PARAM_OFFLOAD:-0}"
critic_grad_offload="${CRITIC_GRAD_OFFLOAD:-1}"
critic_optimizer_offload="${CRITIC_OPTIMIZER_OFFLOAD:-1}"

num_gpus="${NUM_GPUS:-}"
if [ -z "${num_gpus}" ]; then
    num_gpus="${visible_gpu_count}"
    if [ "${num_gpus}" -gt "${train_batch_size}" ]; then
        num_gpus="${train_batch_size}"
    fi
fi

if [ "${num_gpus}" -lt 2 ]; then
    echo "webservlite_train.sh expects at least 2 visible GPUs for formal multi-GPU PPO training."
    echo "Current CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    exit 1
fi

if [ "${num_gpus}" -gt "${visible_gpu_count}" ]; then
    echo "Requested NUM_GPUS=${num_gpus}, but only ${visible_gpu_count} GPUs are visible."
    exit 1
fi

if [ $((train_batch_size % num_gpus)) -ne 0 ]; then
    echo "train_batch_size=${train_batch_size} must be divisible by num_gpus=${num_gpus}."
    echo "Either reduce NUM_GPUS or increase train_batch_size."
    exit 1
fi

# Keep vLLM TP modest so actor/ref/critic can still share the node.
rollout_tp_size="${ROLLOUT_TP_SIZE:-2}"
if [ "${rollout_tp_size}" -gt "${num_gpus}" ]; then
    rollout_tp_size="${num_gpus}"
fi

# Conservative multi-GPU defaults to leave more headroom per GPU.
# These settings trade throughput for stability.
rollout_gpu_mem_util="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.35}"
rollout_max_model_len="${ROLLOUT_MAX_MODEL_LEN:-8192}"
rollout_max_tokens="${ROLLOUT_MAX_TOKENS:-256}"
rollout_max_batched_tokens="${ROLLOUT_MAX_BATCHED_TOKENS:-4096}"
rollout_max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-256}"
actor_default_token_budget=$((data_max_prompt_length + data_max_response_length))
critic_default_token_budget=$((data_max_prompt_length + (data_max_response_length * 2)))
actor_ppo_max_token_len="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-${actor_default_token_budget}}"
critic_ppo_max_token_len="${CRITIC_PPO_MAX_TOKEN_LEN_PER_GPU:-${critic_default_token_budget}}"
critic_micro_batch_size_per_gpu="${CRITIC_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
critic_forward_micro_batch_size_per_gpu="${CRITIC_FORWARD_MICRO_BATCH_SIZE_PER_GPU:-1}"

total_epoches="${TOTAL_EPOCHS:-10}"
save_freq="${SAVE_FREQ:-10}"

model_save_dir="saves"
mkdir -p ${model_save_dir}
exp_name="webservlite"
model_save_path=${model_save_dir}/${exp_name}

mkdir -p ${model_save_path}
mkdir -p "${model_save_path}/logs"

if [ "${debug_train_script}" = "1" ]; then
    set -x
fi

wandb_mode="${WANDB_MODE:-offline}"
project_name="${WANDB_PROJECT:-agentgym_rl}"
hydra_full_error="${HYDRA_FULL_ERROR:-0}"
raw_log="${model_save_path}/logs/train_$(date +%Y%m%d_%H%M%S).log"

print_section() {
    printf '\n========== %s ==========\n' "$1"
}

print_kv() {
    printf '%-20s %s\n' "$1" "$2"
}

print_section "WebServLite Multi-GPU Training"
print_kv "Task" "${task_name}"
print_kv "Env Server" "${env_server_url}"
print_kv "Model" "${agent_model_path}"
print_kv "Model Type" "${model_type}"
print_kv "Visible GPUs" "${CUDA_VISIBLE_DEVICES}"
print_kv "Trainer GPUs" "${num_gpus}"
print_kv "Rollout TP" "${rollout_tp_size}"
print_kv "Rollout Mem Util" "${rollout_gpu_mem_util}"
print_kv "Rollout Max Len" "${rollout_max_model_len}"
print_kv "Rollout Max Tokens" "${rollout_max_tokens}"
print_kv "Max Rounds" "${max_rounds}"
print_kv "Rollout Samples" "${rollout_sample_num}"
print_kv "Prompt Length" "${data_max_prompt_length}"
print_kv "Response Length" "${data_max_response_length}"
print_kv "Remove Padding" "${use_remove_padding}"
print_kv "Actor Offload" "param=${actor_param_offload} grad=${actor_grad_offload} optim=${actor_optimizer_offload}"
print_kv "Ref Offload" "param=${ref_param_offload}"
print_kv "Critic Offload" "param=${critic_param_offload} grad=${critic_grad_offload} optim=${critic_optimizer_offload}"
print_kv "Train Batch" "${train_batch_size}"
print_kv "PPO Mini Batch" "${ppo_mini_batch_size}"
print_kv "Epochs" "${total_epoches}"
print_kv "WandB Mode" "${wandb_mode}"
print_kv "Save Dir" "${model_save_path}"
print_kv "Raw Log" "${raw_log}"

train_cmd=(
    python3 -m verl.agent_trainer.main_ppo
    algorithm.adv_estimator=grpo
    algorithm.rounds_ctrl.type=fixed
    algorithm.rounds_ctrl.rounds=${max_rounds}
    data.train_file=AgentItemId/${task_name}_train.json
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${data_max_prompt_length}
    data.max_response_length=${data_max_response_length}
    actor_rollout_ref.agentgym.task_name=${task_name}
    actor_rollout_ref.agentgym.env_addr=${env_server_url}
    actor_rollout_ref.agentgym.timeout=600
    actor_rollout_ref.model.path=${agent_model_path}
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding}
    critic.model.path=${agent_model_path}
    critic.model.tokenizer_path=${agent_model_path}
    critic.model.use_remove_padding=${use_remove_padding}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.grad_offload=${actor_grad_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_sample_num}
    actor_rollout_ref.rollout.max_model_len=${rollout_max_model_len}
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_batched_tokens}
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.max_tokens=${rollout_max_tokens}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp_size}
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs}
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_param_offload}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${actor_ppo_max_token_len}
    critic.ppo_micro_batch_size_per_gpu=${critic_micro_batch_size_per_gpu}
    critic.forward_micro_batch_size_per_gpu=${critic_forward_micro_batch_size_per_gpu}
    critic.ppo_max_token_len_per_gpu=${critic_ppo_max_token_len}
    critic.forward_max_token_len_per_gpu=${critic_ppo_max_token_len}
    critic.model.fsdp_config.param_offload=${critic_param_offload}
    critic.model.fsdp_config.grad_offload=${critic_grad_offload}
    critic.model.fsdp_config.optimizer_offload=${critic_optimizer_offload}
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    trainer.default_local_dir=${model_save_path}
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${num_gpus}
    trainer.save_freq=${save_freq}
    trainer.total_epochs=${total_epoches}
)

pretty_filter() {
    awk '
    BEGIN {
        warned_ray_disk = 0
        warned_future = 0
        warned_vllm_version = 0
        warned_hydra_config = 0
        warned_model_config = 0
        in_hydra_config = 0
        in_model_config = 0
    }
    /^\(main_task pid=.*\) \{'\''actor_rollout_ref'\'': \{'\''actor'\'': \{'\''clip_ratio'\''/ {
        if (!warned_hydra_config) {
            print "[INFO] 已省略 Hydra 配置展开；完整内容见原始日志。"
            warned_hydra_config = 1
        }
        in_hydra_config = 1
        next
    }
    in_hydra_config {
        if (/All configuration checks passed successfully!/) {
            in_hydra_config = 0
            print $0
        }
        next
    }
    /Model config after override: Qwen2Config \{/ {
        if (!warned_model_config) {
            print "[INFO] 已省略模型配置展开；完整内容见原始日志。"
            warned_model_config = 1
        }
        in_model_config = 1
        next
    }
    in_model_config {
        if (/^\(WorkerDict pid=.*\) \}$/) {
            in_model_config = 0
        }
        next
    }
    /file_system_monitor\.cc:116/ {
        if (!warned_ray_disk) {
            print "[WARN] Ray 临时目录磁盘使用率超过 95%，如果发生 spilling 可能失败。"
            warned_ray_disk = 1
        }
        next
    }
    /FutureWarning/ {
        if (!warned_future) {
            print "[WARN] 已省略重复 FutureWarning；完整内容见原始日志。"
            warned_future = 1
        }
        next
    }
    /RuntimeWarning: Failed to read commit hash/ { next }
    /No module named '\''vllm\._version'\''/ {
        if (!warned_vllm_version) {
            print "[WARN] vLLM 版本元数据读取失败，但通常不影响训练。"
            warned_vllm_version = 1
        }
        next
    }
    /wrap_policy:/ { next }
    /Error executing job with overrides:/ { next }
    /wandb: Use W&B Weave/ { next }
    /wandb: Detected \[openai\] in use\./ { next }
    /Check[pP]oint tracker file does not exist/ { next }
    /Loading checkpoint shards:/ { print; next }
    /Started a local Ray instance/ { print "[RAY] " $0; next }
    /dataset len:/ { print "[DATA] " $0; next }
    /Size of train dataloader:/ { print "[DATA] " $0; next }
    /Total training steps:/ { print "[TRAIN] " $0; next }
    /Training from scratch/ { print "[TRAIN] " $0; next }
    /Run data is saved locally in/ { print "[WANDB] " $0; next }
    /Raw Log/ { print; next }
    { print }
    '
}

print_section "Launch"
set +e
if [ "${pretty_log}" = "1" ]; then
    HYDRA_FULL_ERROR="${hydra_full_error}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=${wandb_mode} \
        "${train_cmd[@]}" 2>&1 | tee "${raw_log}" | pretty_filter
    status=${PIPESTATUS[0]}
else
    HYDRA_FULL_ERROR="${hydra_full_error}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=${wandb_mode} \
        "${train_cmd[@]}" 2>&1 | tee "${raw_log}"
    status=${PIPESTATUS[0]}
fi
set -e

print_section "Finished"
print_kv "Exit Code" "${status}"
print_kv "Raw Log" "${raw_log}"
exit "${status}"
