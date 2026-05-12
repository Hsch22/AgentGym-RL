#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-searchqa_eval_ckpts}"
ENV_SESSION_NAME="${ENV_SESSION_NAME:-searchqa_eval_env}"
PORT="${PORT:-36015}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/AgentGym-RL}"
VENVPY="${VENVPY:-${REPO_ROOT}/.venv/bin/python}"
SEARCHQA_ENV_CONDA_PREFIX="${SEARCHQA_ENV_CONDA_PREFIX:-${REPO_ROOT}/.venv}"
SEARCHQA_SERVER_BIN="${SEARCHQA_EVAL_SERVER_BIN:-${SEARCHQA_ENV_CONDA_PREFIX}/bin/searchqa}"
SEARCHQA_EVAL_FAISS_GPU="${SEARCHQA_EVAL_FAISS_GPU:-true}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/saves/searchqa_scalinginter_3b_2gpushard_faissgpu_20260422_1220}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${PROJECT_ROOT}/AgentEval/searchqa/eval}"
LAUNCH_SEARCHQA_ENV="${LAUNCH_SEARCHQA_ENV:-${SCRIPT_DIR}/launch_searchqa_env.sh}"

# Leave two GPUs for the FAISS env server by default; evaluate on the other six.
ENV_GPUS="${SEARCHQA_EVAL_ENV_GPUS:-2,3}"
EVAL_GPUS="${SEARCHQA_EVAL_CUDA_VISIBLE_DEVICES:-0,1,4,5,6,7}"
N_GPUS="${SEARCHQA_EVAL_N_GPUS:-$(awk -F',' '{print NF}' <<< "${EVAL_GPUS}")}"
CHECKPOINTS="${CHECKPOINTS:-25 50 75 100 125 150 175 200 225}"
BATCH_SIZE="${SEARCHQA_EVAL_BATCH_SIZE:-32}"
LOG_FILE="${LOG_FILE:-/tmp/searchqa_eval_ckpt_sweep.log}"
ENV_LOG_FILE="${ENV_LOG_FILE:-/tmp/searchqa_eval_env_${PORT}.log}"

WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-true}"
GPU_WAIT_MAX_MEM_MB="${GPU_WAIT_MAX_MEM_MB:-20000}"
GPU_WAIT_POLL_S="${GPU_WAIT_POLL_S:-300}"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "\
  set -euo pipefail; \
  cd ${PROJECT_ROOT}; \
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; \
  export no_proxy='127.0.0.1,localhost'; \
  export NO_PROXY='127.0.0.1,localhost'; \
  if [ '${WAIT_FOR_GPU_FREE}' = 'true' ]; then \
    echo '[searchqa-eval] waiting for GPUs ${EVAL_GPUS},${ENV_GPUS} max_mem < ${GPU_WAIT_MAX_MEM_MB} MiB'; \
    while true; do \
      max_used=\$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v ids='${EVAL_GPUS},${ENV_GPUS}' 'BEGIN{split(ids,a,\",\"); for(i in a){want[a[i]]=1}; m=0} {gsub(/[, ]/,\"\",\$1); gsub(/[, ]/,\"\",\$2); if((\$1 in want) && \$2>m){m=\$2}} END{print m+0}'); \
      echo \"[searchqa-eval] max target gpu memory used: \${max_used} MiB\"; \
      if [ \"\${max_used}\" -lt '${GPU_WAIT_MAX_MEM_MB}' ]; then break; fi; \
      sleep '${GPU_WAIT_POLL_S}'; \
    done; \
  fi; \
  SESSION_NAME=${ENV_SESSION_NAME} PORT=${PORT} GPU_ID='${ENV_GPUS}' SEARCHQA_SERVER_BIN='${SEARCHQA_SERVER_BIN}' SEARCHQA_FAISS_GPU=${SEARCHQA_EVAL_FAISS_GPU} SEARCHQA_RETRIEVAL_BATCH_SIZE=128 SEARCHQA_SEARCH_BATCH_SIZE=32 SEARCHQA_SEARCH_BATCH_MAX_WAIT_MS=20 LOG_FILE=${ENV_LOG_FILE} bash ${LAUNCH_SEARCHQA_ENV}; \
  echo '[searchqa-eval] waiting for env server http://127.0.0.1:${PORT}'; \
  until ${VENVPY} - <<'PY'; do sleep 10; done
import requests
raise SystemExit(0 if requests.get('http://127.0.0.1:${PORT}/', timeout=5).status_code == 200 else 1)
PY
  export WANDB_MODE=offline; \
  export VLLM_USE_MODELSCOPE=0; \
  export VLLM_WORKER_MULTIPROC_METHOD=spawn; \
  export VLLM_ATTENTION_BACKEND=XFORMERS; \
  ${VENVPY} ${PROJECT_ROOT}/scripts/eval_searchqa_checkpoints.py \
    --run-dir ${RUN_DIR} \
    --eval-data-dir ${EVAL_DATA_DIR} \
    --env-addr http://127.0.0.1:${PORT} \
    --checkpoints ${CHECKPOINTS} \
    --n-gpus ${N_GPUS} \
    --cuda-visible-devices '${EVAL_GPUS}' \
    --batch-size ${BATCH_SIZE} \
    --wandb-project agentgym-rl-eval \
    --wandb-name searchqa_eval_\$(basename ${RUN_DIR}) \
  2>&1 | tee ${LOG_FILE}"

echo "Started SearchQA checkpoint sweep in tmux session ${SESSION_NAME}"
echo "Eval log: ${LOG_FILE}"
echo "Env session: ${ENV_SESSION_NAME}"
echo "Env server bin: ${SEARCHQA_SERVER_BIN}"
echo "Env FAISS GPU: ${SEARCHQA_EVAL_FAISS_GPU}"
echo "Env log: ${ENV_LOG_FILE}"
