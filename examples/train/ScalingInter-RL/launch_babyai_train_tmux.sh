#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-babyai_scalinginter}"
SCRIPT_PATH="/share/project/husicheng/muhan/AgentGym-RL/examples/train/ScalingInter-RL/babyai_train.sh"
LOG_FILE="${LOG_FILE:-/tmp/babyai_scalinginter_train.log}"

tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
tmux new-session -d -s "${SESSION_NAME}" "bash ${SCRIPT_PATH} 2>&1 | tee ${LOG_FILE}"

echo "Started BabyAI training in tmux session ${SESSION_NAME}"
echo "Log: ${LOG_FILE}"
