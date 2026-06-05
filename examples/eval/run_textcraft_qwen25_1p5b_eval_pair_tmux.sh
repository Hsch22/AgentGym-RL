#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/AgentGym-RL}"
VENVPY="${VENVPY:-${REPO_ROOT}/.venv/bin/python}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
SESSION_NAME="${SESSION_NAME:-textcraft_qwen25_1p5b_eval_pair_${RUN_TS}}"
PORT="${PORT:-36005}"
EVAL_GPUS="${TEXTCRAFT_EVAL_CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS="${TEXTCRAFT_EVAL_N_GPUS:-2}"
CHECKPOINTS="${CHECKPOINTS:-25 50 75 100 125 150 175 200 225 250 275 300 325 330}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/textcraft_qwen25_1p5b_eval_pair_${RUN_TS}.log}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${REPO_ROOT}/results/wandb_agentgym_rl_eval_textcraft_qwen25_1p5b_${RUN_TS}}"
OFFLINE_LIST="${OFFLINE_LIST:-${REPO_ROOT}/logs/textcraft_qwen25_1p5b_eval_pair_offline_runs_${RUN_TS}.txt}"

BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-${PROJECT_ROOT}/saves/textcraft_scalinginter_baseline_qwen25_1p5b_2xh200_fastsettings_20260521_092959}"
G2RL_RUN_DIR="${G2RL_RUN_DIR:-${PROJECT_ROOT}/saves/textcraft_paper_g2rl_response_qwen25_1p5b_2xh200_fastsettings_20260521_171322}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-${PROJECT_ROOT}/AgentEval/textcraft/eval}"

mkdir -p "${REPO_ROOT}/logs" "$(dirname "${ARCHIVE_ROOT}")"

if [[ "${1:-}" == "--worker" ]]; then
    exec > >(tee -a "${LOG_FILE}") 2>&1

    echo "[eval-pair] start $(date '+%F %T')"
    echo "[eval-pair] project_root=${PROJECT_ROOT}"
    echo "[eval-pair] checkpoints=${CHECKPOINTS}"
    echo "[eval-pair] archive_root=${ARCHIVE_ROOT}"
    echo "[eval-pair] offline_list=${OFFLINE_LIST}"

    cd "${PROJECT_ROOT}"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    export no_proxy="127.0.0.1,localhost"
    export NO_PROXY="127.0.0.1,localhost"
    export CUDA_VISIBLE_DEVICES="${EVAL_GPUS}"
    export VLLM_USE_MODELSCOPE=0
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_ATTENTION_BACKEND=XFORMERS
    export HYDRA_FULL_ERROR=1
    export WANDB_MODE=offline

    echo "[eval-pair] checking TextCraft env http://127.0.0.1:${PORT}"
    until "${VENVPY}" - <<PY; do
import requests
try:
    ok = requests.post(
        "http://127.0.0.1:${PORT}/create",
        json={"minecraft_dir":"agentenv_textcraft/","commands":None,"goal":None},
        timeout=10,
    ).status_code < 500
except requests.RequestException:
    ok = False
raise SystemExit(0 if ok else 1)
PY
        echo "[eval-pair] TextCraft env not ready; sleep 10s"
        sleep 10
    done
    echo "[eval-pair] TextCraft env ready"

    run_one() {
        local ordinal="$1"
        local label="$2"
        local run_dir="$3"
        local wandb_name="$4"
        local before_file
        local after_file
        local offline_run
        before_file="$(mktemp)"
        after_file="$(mktemp)"
        find "${PROJECT_ROOT}/wandb" -maxdepth 1 -type d -name 'offline-run-*' -print | sort > "${before_file}" || true

        echo "[eval-pair] running ${label}: ${run_dir}"
        "${VENVPY}" "${PROJECT_ROOT}/scripts/eval_textcraft_checkpoints.py" \
            --run-dir "${run_dir}" \
            --eval-data-dir "${EVAL_DATA_DIR}" \
            --env-addr "http://127.0.0.1:${PORT}" \
            --checkpoints ${CHECKPOINTS} \
            --n-gpus "${N_GPUS}" \
            --wandb-project agentgym-rl-eval \
            --wandb-name "${wandb_name}" \
            --wandb-mode offline

        find "${PROJECT_ROOT}/wandb" -maxdepth 1 -type d -name 'offline-run-*' -print | sort > "${after_file}" || true
        offline_run="$(comm -13 "${before_file}" "${after_file}" | tail -n 1)"
        if [[ -z "${offline_run}" ]]; then
            offline_run="$(find "${PROJECT_ROOT}/wandb" -maxdepth 1 -type d -name 'offline-run-*' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
        fi
        rm -f "${before_file}" "${after_file}"

        echo "${label} ${offline_run}" | tee -a "${OFFLINE_LIST}"
        echo "[eval-pair] ${label} offline_run=${offline_run}"

        local run_id
        run_id="$(basename "${offline_run}" | awk -F- '{print $NF}')"
        "${VENVPY}" "${PROJECT_ROOT}/scripts/archive_eval_sweep_to_wandb_export.py" \
            --archive-root "${ARCHIVE_ROOT}" \
            --ordinal "${ordinal}" \
            --run-id "${run_id}" \
            --run-name "${wandb_name}" \
            --run-dir "${run_dir}" \
            --eval-data-dir "${EVAL_DATA_DIR}" \
            --env-addr "http://127.0.0.1:${PORT}" \
            --n-gpus "${N_GPUS}" \
            --offline-run-dir "${offline_run}" \
            --results-csv "${run_dir}/eval_textcraft_ckpt_sweep/results.csv" \
            --results-json "${run_dir}/eval_textcraft_ckpt_sweep/results.json"
    }

    run_one 1 baseline "${BASELINE_RUN_DIR}" "textcraft_eval_qwen25_1p5b_baseline_2xh200_fastsettings_${RUN_TS}"
    run_one 2 g2rl_response "${G2RL_RUN_DIR}" "textcraft_eval_qwen25_1p5b_g2rl_response_2xh200_fastsettings_${RUN_TS}"

    echo "[eval-pair] finished $(date '+%F %T')"
    echo "[eval-pair] archive_root=${ARCHIVE_ROOT}"
    echo "[eval-pair] offline_list=${OFFLINE_LIST}"
    exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" \
    "RUN_TS='${RUN_TS}' SESSION_NAME='${SESSION_NAME}' LOG_FILE='${LOG_FILE}' ARCHIVE_ROOT='${ARCHIVE_ROOT}' OFFLINE_LIST='${OFFLINE_LIST}' PROJECT_ROOT='${PROJECT_ROOT}' VENVPY='${VENVPY}' PORT='${PORT}' TEXTCRAFT_EVAL_CUDA_VISIBLE_DEVICES='${EVAL_GPUS}' TEXTCRAFT_EVAL_N_GPUS='${N_GPUS}' CHECKPOINTS='${CHECKPOINTS}' BASELINE_RUN_DIR='${BASELINE_RUN_DIR}' G2RL_RUN_DIR='${G2RL_RUN_DIR}' EVAL_DATA_DIR='${EVAL_DATA_DIR}' bash '${BASH_SOURCE[0]}' --worker"

echo "Started tmux session: ${SESSION_NAME}"
echo "Eval log: ${LOG_FILE}"
echo "Archive root: ${ARCHIVE_ROOT}"
echo "Offline eval run list: ${OFFLINE_LIST}"
