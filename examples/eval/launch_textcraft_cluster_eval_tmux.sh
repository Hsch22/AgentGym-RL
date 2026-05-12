#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-textcraft_cluster_eval_ckpts}"
ENV_SESSION_NAME="${ENV_SESSION_NAME:-textcraft_cluster_eval_env}"
PORT="${PORT:-36005}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/AgentGym-RL}"
AGENTGYM_ROOT="${AGENTGYM_ROOT:-${REPO_ROOT}/AgentGym}"
VENVPY="${VENVPY:-${REPO_ROOT}/.venv/bin/python}"
TEXTCRAFT_SERVER_BIN="${TEXTCRAFT_SERVER_BIN:-${REPO_ROOT}/.venv/bin/textcraft}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/saves/tc_sem_rmpad_chunked_l9c3_20260418_1336}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${PROJECT_ROOT}/AgentEval/textcraft/eval}"

EVAL_GPUS="${TEXTCRAFT_EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
N_GPUS="${TEXTCRAFT_EVAL_N_GPUS:-$(awk -F',' '{print NF}' <<< "${EVAL_GPUS}")}"
CHECKPOINTS="${CHECKPOINTS:-25 50 75 100}"
LOG_FILE="${LOG_FILE:-/tmp/textcraft_cluster_eval_ckpt_sweep.log}"
ENV_LOG_FILE="${ENV_LOG_FILE:-/tmp/textcraft_cluster_eval_env_${PORT}.log}"

WAIT_FOR_GPU_FREE="${WAIT_FOR_GPU_FREE:-true}"
GPU_WAIT_MAX_MEM_MB="${GPU_WAIT_MAX_MEM_MB:-20000}"
GPU_WAIT_POLL_S="${GPU_WAIT_POLL_S:-300}"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux kill-session -t "${ENV_SESSION_NAME}" 2>/dev/null || true

tmux new-session -d -s "${ENV_SESSION_NAME}" "\
  set -euo pipefail; \
  cd ${AGENTGYM_ROOT}/agentenv-textcraft; \
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; \
  export no_proxy='127.0.0.1,localhost'; \
  export NO_PROXY='127.0.0.1,localhost'; \
  echo '[textcraft-env] starting ${TEXTCRAFT_SERVER_BIN} on port ${PORT}'; \
  exec ${TEXTCRAFT_SERVER_BIN} --host 0.0.0.0 --port ${PORT} \
  2>&1 | tee ${ENV_LOG_FILE}"

tmux new-session -d -s "${SESSION_NAME}" "\
  set -euo pipefail; \
  cd ${PROJECT_ROOT}; \
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; \
  export no_proxy='127.0.0.1,localhost'; \
  export NO_PROXY='127.0.0.1,localhost'; \
  if [ '${WAIT_FOR_GPU_FREE}' = 'true' ]; then \
    echo '[textcraft-eval] waiting for GPUs ${EVAL_GPUS} max_mem < ${GPU_WAIT_MAX_MEM_MB} MiB'; \
    while true; do \
      max_used=\$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v ids='${EVAL_GPUS}' 'BEGIN{split(ids,a,\",\"); for(i in a){want[a[i]]=1}; m=0} {gsub(/[, ]/,\"\",\$1); gsub(/[, ]/,\"\",\$2); if((\$1 in want) && \$2>m){m=\$2}} END{print m+0}'); \
      echo \"[textcraft-eval] max target gpu memory used: \${max_used} MiB\"; \
      if [ \"\${max_used}\" -lt '${GPU_WAIT_MAX_MEM_MB}' ]; then break; fi; \
      sleep '${GPU_WAIT_POLL_S}'; \
    done; \
  fi; \
  echo '[textcraft-eval] waiting for env server http://127.0.0.1:${PORT}'; \
  until ${VENVPY} - <<'PY'; do sleep 10; done
import requests
try:
    ok = requests.get('http://127.0.0.1:${PORT}/', timeout=5).status_code == 200
except requests.RequestException:
    ok = False
raise SystemExit(0 if ok else 1)
PY
  export WANDB_MODE=offline; \
  export CUDA_VISIBLE_DEVICES='${EVAL_GPUS}'; \
  export VLLM_USE_MODELSCOPE=0; \
  export VLLM_WORKER_MULTIPROC_METHOD=spawn; \
  export VLLM_ATTENTION_BACKEND=XFORMERS; \
  export HYDRA_FULL_ERROR=1; \
  ${VENVPY} ${PROJECT_ROOT}/scripts/eval_textcraft_checkpoints.py \
    --run-dir ${RUN_DIR} \
    --eval-data-dir ${EVAL_DATA_DIR} \
    --env-addr http://127.0.0.1:${PORT} \
    --checkpoints ${CHECKPOINTS} \
    --n-gpus ${N_GPUS} \
    --wandb-project agentgym-rl-eval \
    --wandb-name textcraft_cluster_eval_\$(basename ${RUN_DIR}) \
    --wandb-mode offline \
  2>&1 | tee ${LOG_FILE}"

echo "Started TextCraft cluster checkpoint sweep in tmux session ${SESSION_NAME}"
echo "Eval log: ${LOG_FILE}"
echo "Env session: ${ENV_SESSION_NAME}"
echo "Env log: ${ENV_LOG_FILE}"
echo "Run dir: ${RUN_DIR}"
echo "Checkpoints: ${CHECKPOINTS}"
