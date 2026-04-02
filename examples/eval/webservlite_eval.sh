set -x
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

task_name="webservlite"

cd AgentGym-RL
source ~/miniconda3/etc/profile.d/conda.sh
conda activate agentgym-rl-clean
export VLLM_ATTENTION_BACKEND=XFORMERS

env_server_url="${ENV_SERVER_URL:-http://127.0.0.1:36006}"

sample_num=1
max_rounds=10

ckpt_path="${CKPT_PATH:-global_step_10/actor}"
model_path=${ckpt_path}/huggingface
base_model_name="${MODEL_NAME:-Qwen2.5-1.5B-Instruct}"
base_model_path="${MODEL_PATH:-models/${base_model_name}}"

if [ ! -f "${base_model_path}/config.json" ] && [ ! -f "${model_path}/config.json" ]; then
    echo "Neither merged checkpoint nor base model config was found."
    echo "Checked: ${model_path}/config.json and ${base_model_path}/config.json"
    exit 1
fi

cd AgentGym-RL/scripts
python model_merger.py \
    --local_dir ${ckpt_path}

HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_generation  \
    data.path=AgentEval/${task_name} \
    data.max_prompt_length=700 \
    data.max_response_length=4096 \
    data.n_samples=${sample_num} \
    data.batch_size=16 \
    agentgym.task_name=${task_name} \
    agentgym.env_addr=${env_server_url} \
    agentgym.max_rounds=${max_rounds} \
    agentgym.timeout=500 \
    model.path=${model_path} \
    rollout.gpu_memory_utilization=0.95 \
    rollout.temperature=1 \
    rollout.max_model_len=16384 \
    rollout.max_tokens=384 \
    rollout.tensor_model_parallel_size=1 \
    rollout.rollout_log_dir=executer_logs
status=$?
exit $status
