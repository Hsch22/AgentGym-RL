#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-searchqa_env}"
PORT="${PORT:-36015}"
GPU_ID="${GPU_ID:-2}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Local service should never use inherited proxies.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

export SEARCHQA_RETRIEVAL_METHOD="${SEARCHQA_RETRIEVAL_METHOD:-e5}"
export SEARCHQA_FAISS_GPU="${SEARCHQA_FAISS_GPU:-false}"
export SEARCHQA_RETRIEVAL_USE_FP16="${SEARCHQA_RETRIEVAL_USE_FP16:-true}"
export SEARCHQA_RETRIEVAL_BATCH_SIZE="${SEARCHQA_RETRIEVAL_BATCH_SIZE:-128}"

LOG_FILE="${LOG_FILE:-/tmp/searchqa_env_${PORT}.log}"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "\
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; \
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; \
  export no_proxy='127.0.0.1,localhost'; \
  export NO_PROXY='127.0.0.1,localhost'; \
  export SEARCHQA_RETRIEVAL_METHOD=${SEARCHQA_RETRIEVAL_METHOD}; \
  export SEARCHQA_FAISS_GPU=${SEARCHQA_FAISS_GPU}; \
  export SEARCHQA_RETRIEVAL_USE_FP16=${SEARCHQA_RETRIEVAL_USE_FP16}; \
  export SEARCHQA_RETRIEVAL_BATCH_SIZE=${SEARCHQA_RETRIEVAL_BATCH_SIZE}; \
  /share/project/husicheng/muhan/AgentGym-RL/.venv/bin/searchqa --host 127.0.0.1 --port ${PORT} 2>&1 | tee ${LOG_FILE}"

echo "Started SearchQA env server in tmux session ${SESSION_NAME} on port ${PORT}"
echo "Log: ${LOG_FILE}"
