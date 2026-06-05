#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/AgentGym-RL}"
VENVPY="${VENVPY:-${REPO_ROOT}/.venv/bin/python}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
SESSION_NAME="${SESSION_NAME:-textcraft_qwen25_1p5b_cluster_ablation_100step_${RUN_TS}}"
MODEL_PATH="${AGENT_MODEL_PATH:-${REPO_ROOT}/models/Qwen2.5-1.5B-Instruct}"
PORT="${TEXTCRAFT_ENV_PORT:-36005}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
N_GPUS="${TEXTCRAFT_N_GPUS_PER_NODE:-2}"
VARIANTS="${VARIANTS:-A0 A1 A2}"

LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/textcraft_qwen25_1p5b_cluster_ablation_100step_${RUN_TS}.log}"
MANIFEST="${MANIFEST:-${REPO_ROOT}/logs/textcraft_qwen25_1p5b_cluster_ablation_100step_${RUN_TS}.manifest.tsv}"
SUMMARY_DOC="${SUMMARY_DOC:-${REPO_ROOT}/docs/textcraft_1p5b_clustering_100step_run_${RUN_TS}.md}"
AUDIT_ROOT="${AUDIT_ROOT:-${REPO_ROOT}/results/textcraft_qwen25_1p5b_cluster_ablation_100step_${RUN_TS}}"

EXP_A0="${EXP_A0:-textcraft_qwen25_1p5b_A0_no_cluster_no_g2rl_100step_${RUN_TS}}"
EXP_A1="${EXP_A1:-textcraft_qwen25_1p5b_A1_random_valid_no_g2rl_100step_${RUN_TS}}"
EXP_A2="${EXP_A2:-textcraft_qwen25_1p5b_A2_gradient_multiview_no_g2rl_100step_${RUN_TS}}"
EXP_A3="${EXP_A3:-textcraft_qwen25_1p5b_A3_g2rl_normalized_action_gradient_no_reward_100step_${RUN_TS}}"
EXP_A4="${EXP_A4:-textcraft_qwen25_1p5b_A4_quality_unique_action_no_g2rl_100step_${RUN_TS}}"

mkdir -p "${REPO_ROOT}/logs" "${AUDIT_ROOT}" "$(dirname "${SUMMARY_DOC}")"

ensure_textcraft_env() {
    echo "[cluster-100] checking TextCraft env http://127.0.0.1:${PORT}"
    if "${VENVPY}" - <<PY; then
import requests
try:
    response = requests.post(
        "http://127.0.0.1:${PORT}/create",
        json={"minecraft_dir": "agentenv_textcraft/", "commands": None, "goal": None},
        timeout=10,
    )
    ok = response.status_code < 500
except requests.RequestException:
    ok = False
raise SystemExit(0 if ok else 1)
PY
        echo "[cluster-100] TextCraft env ready"
        return 0
    fi

    echo "[cluster-100] TextCraft env not ready; starting server on port ${PORT}"
    PORT="${PORT}" "${SCRIPT_DIR}/launch_textcraft_env.sh"
}

common_train_env() {
    export CUDA_VISIBLE_DEVICES
    export TEXTCRAFT_N_GPUS_PER_NODE="${N_GPUS}"
    export TEXTCRAFT_ENV_PORT="${PORT}"
    export TEXTCRAFT_AUTO_START_ENV=0
    export WANDB_MODE=offline
    export AGENT_MODEL_PATH="${MODEL_PATH}"
    export TEXTCRAFT_USE_SHM_MODEL=0
    export TEXTCRAFT_RESUME_MODE="${TEXTCRAFT_RESUME_MODE:-disable}"

    export TEXTCRAFT_TRAIN_BATCH_SIZE="${TEXTCRAFT_TRAIN_BATCH_SIZE:-32}"
    export TEXTCRAFT_ROLLOUT_N="${TEXTCRAFT_ROLLOUT_N:-8}"
    export TEXTCRAFT_VAL_BATCH_SIZE="${TEXTCRAFT_VAL_BATCH_SIZE:-32}"
    export TEXTCRAFT_PPO_MINI_BATCH_SIZE="${TEXTCRAFT_PPO_MINI_BATCH_SIZE:-8}"
    export TEXTCRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU="${TEXTCRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
    export TEXTCRAFT_PPO_EPOCHS="${TEXTCRAFT_PPO_EPOCHS:-2}"
    export TEXTCRAFT_TOTAL_EPOCHS="${TEXTCRAFT_TOTAL_EPOCHS:-30}"
    export TEXTCRAFT_TOTAL_TRAINING_STEPS="${TEXTCRAFT_TOTAL_TRAINING_STEPS:-100}"

    export TEXTCRAFT_MAX_PROMPT_LENGTH="${TEXTCRAFT_MAX_PROMPT_LENGTH:-512}"
    export TEXTCRAFT_MAX_RESPONSE_LENGTH="${TEXTCRAFT_MAX_RESPONSE_LENGTH:-10240}"
    export TEXTCRAFT_USE_REMOVE_PADDING="${TEXTCRAFT_USE_REMOVE_PADDING:-true}"
    export TEXTCRAFT_ROLLOUT_GPU_MEMORY_UTILIZATION="${TEXTCRAFT_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}"
    export TEXTCRAFT_ROLLOUT_MAX_MODEL_LEN="${TEXTCRAFT_ROLLOUT_MAX_MODEL_LEN:-32768}"
    export TEXTCRAFT_ROLLOUT_MAX_NUM_BATCHED_TOKENS="${TEXTCRAFT_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"
    export TEXTCRAFT_ROLLOUT_MAX_TOKENS="${TEXTCRAFT_ROLLOUT_MAX_TOKENS:-512}"
    export TEXTCRAFT_PPO_MAX_TOKEN_LEN_PER_GPU="${TEXTCRAFT_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
    export TEXTCRAFT_ENTROPY_CHUNK_SIZE="${TEXTCRAFT_ENTROPY_CHUNK_SIZE:-256}"

    export TEXTCRAFT_SAVE_FREQ="${TEXTCRAFT_SAVE_FREQ:-25}"
    export TEXTCRAFT_TEST_FREQ="${TEXTCRAFT_TEST_FREQ:-35}"
    export TEXTCRAFT_TEST_BATCHES="${TEXTCRAFT_TEST_BATCHES:-1}"
    export TEXTCRAFT_EARLY_STOP_ENABLED="${TEXTCRAFT_EARLY_STOP_ENABLED:-true}"
    export TEXTCRAFT_EARLY_STOP_MIN_STEPS="${TEXTCRAFT_EARLY_STOP_MIN_STEPS:-100}"
    export TEXTCRAFT_EARLY_STOP_PATIENCE="${TEXTCRAFT_EARLY_STOP_PATIENCE:-5}"
    export TEXTCRAFT_EARLY_STOP_MIN_DELTA="${TEXTCRAFT_EARLY_STOP_MIN_DELTA:-0.005}"
    export TEXTCRAFT_EARLY_STOP_MAX_DROP="${TEXTCRAFT_EARLY_STOP_MAX_DROP:-0.08}"
    export TEXTCRAFT_EARLY_STOP_HARD_MIN_METRIC="${TEXTCRAFT_EARLY_STOP_HARD_MIN_METRIC:-0.20}"
    export TEXTCRAFT_EARLY_STOP_MIN_ROLLOUT_VALID_RATIO="${TEXTCRAFT_EARLY_STOP_MIN_ROLLOUT_VALID_RATIO:-0.20}"
    export TEXTCRAFT_EARLY_STOP_MAX_RESPONSE_LENGTH_MEAN="${TEXTCRAFT_EARLY_STOP_MAX_RESPONSE_LENGTH_MEAN:-1500}"
    export TEXTCRAFT_EARLY_STOP_MAX_KL_LOSS="${TEXTCRAFT_EARLY_STOP_MAX_KL_LOSS:-1.0}"
    export TEXTCRAFT_EARLY_STOP_MAX_GRAD_NORM="${TEXTCRAFT_EARLY_STOP_MAX_GRAD_NORM:-100000000}"

    export TEXTCRAFT_ROUNDS_CTRL_TYPE="${TEXTCRAFT_ROUNDS_CTRL_TYPE:-scaling_inter_stepwise}"
    export TEXTCRAFT_ROUNDS_SCALING_INTER="${TEXTCRAFT_ROUNDS_SCALING_INTER:-100}"
    export TEXTCRAFT_ROUNDS_SCHEDULE="${TEXTCRAFT_ROUNDS_SCHEDULE:-[10,20,30]}"

    export TEXTCRAFT_G2RL_ENABLED="${TEXTCRAFT_G2RL_ENABLED:-false}"
    export TEXTCRAFT_G2RL_FEATURE_SCOPE="${TEXTCRAFT_G2RL_FEATURE_SCOPE:-normalized_action}"

    # Realistic default schedule after the 64/16 schedule proved too costly:
    # round0 only uses 16 -> 8 (2x candidate overhead), and later clustering is off.
    export TEXTCRAFT_ROUND1_CANDIDATES="${TEXTCRAFT_ROUND1_CANDIDATES:-16}"
    export TEXTCRAFT_ROUND1_CLUSTERS="${TEXTCRAFT_ROUND1_CLUSTERS:-8}"
    export TEXTCRAFT_LATER_CANDIDATES="${TEXTCRAFT_LATER_CANDIDATES:-4}"
    export TEXTCRAFT_LATER_CLUSTERS="${TEXTCRAFT_LATER_CLUSTERS:-1}"
    export TEXTCRAFT_LATER_CLUSTER_EVERY="${TEXTCRAFT_LATER_CLUSTER_EVERY:-0}"
    export TEXTCRAFT_LATER_CLUSTER_START="${TEXTCRAFT_LATER_CLUSTER_START:-1}"
    export TEXTCRAFT_LATER_CLUSTER_UNTIL="${TEXTCRAFT_LATER_CLUSTER_UNTIL:--1}"
    export TEXTCRAFT_LATER_CLUSTER_HORIZON_MIN="${TEXTCRAFT_LATER_CLUSTER_HORIZON_MIN:-0.25}"
    export TEXTCRAFT_GRADIENT_D_PROJ="${TEXTCRAFT_GRADIENT_D_PROJ:-512}"
    export TEXTCRAFT_FEATURE_TOPK="${TEXTCRAFT_FEATURE_TOPK:-256}"
    export TEXTCRAFT_FEATURE_CHUNK_SIZE="${TEXTCRAFT_FEATURE_CHUNK_SIZE:-4}"
}

validate_rollout_budget() {
    local variant="$1"
    local clustering_enabled="$2"
    local round0_limit

    if [[ "${clustering_enabled}" != "true" ]]; then
        return 0
    fi
    if [[ "${TEXTCRAFT_ALLOW_HIGH_ROLLOUT:-0}" == "1" ]]; then
        return 0
    fi

    round0_limit=$((TEXTCRAFT_ROLLOUT_N * 2))
    if (( TEXTCRAFT_ROUND1_CANDIDATES > round0_limit )); then
        echo "[cluster-100] refusing ${variant}: round0 candidates ${TEXTCRAFT_ROUND1_CANDIDATES} exceed low-overhead cap ${round0_limit}" >&2
        echo "[cluster-100] set TEXTCRAFT_ALLOW_HIGH_ROLLOUT=1 only for an intentional high-cost diagnostic" >&2
        return 1
    fi
    if (( TEXTCRAFT_LATER_CLUSTER_EVERY > 0 && TEXTCRAFT_LATER_CANDIDATES > 4 )); then
        echo "[cluster-100] refusing ${variant}: later candidates ${TEXTCRAFT_LATER_CANDIDATES} exceed low-overhead cap 4" >&2
        echo "[cluster-100] keep TEXTCRAFT_LATER_CLUSTER_EVERY=0 for round0-only clustering, or set TEXTCRAFT_ALLOW_HIGH_ROLLOUT=1 intentionally" >&2
        return 1
    fi
}

variant_config() {
    case "$1" in
        A0)
            echo "${EXP_A0}|false|random_valid|baseline: no clustering, no G2RL reward"
            ;;
        A1)
            echo "${EXP_A1}|true|random_valid|control: random valid clustering, no G2RL reward"
            ;;
        A2)
            echo "${EXP_A2}|true|gradient_multiview|diagnostic: gradient multiview clustering, no G2RL reward"
            ;;
        A3)
            echo "${EXP_A3}|true|g2rl_normalized_action_gradient|main: G2RL normalized-action gradient clustering, no G2RL reward shaping"
            ;;
        A4)
            echo "${EXP_A4}|true|quality_unique_action|diagnostic: high-confidence unique-action clustering, no G2RL reward"
            ;;
        *)
            echo "[cluster-100] unknown variant: $1" >&2
            return 1
            ;;
    esac
}

latest_offline_run() {
    local root="$1"
    find "${root}/wandb" -maxdepth 1 -type d -name 'offline-run-*' -printf '%T@ %p\n' \
        | sort -n \
        | tail -n 1 \
        | cut -d' ' -f2-
}

discover_audit_steps() {
    local log_dir="$1"
    local wanted=(1 25 50 75 100)
    local found=()
    local step

    for step in "${wanted[@]}"; do
        if [[ -d "${log_dir}/step${step}" ]]; then
            found+=("${step}")
        fi
    done

    if [[ "${#found[@]}" -eq 0 ]]; then
        mapfile -t found < <(
            find "${log_dir}" -mindepth 1 -maxdepth 1 -type d -name 'step*' -printf '%f\n' \
                | sed 's/^step//' \
                | sort -n \
                | tail -n 5
        )
    fi

    printf '%s\n' "${found[@]}"
}

run_audit() {
    local variant="$1"
    local exp_name="$2"
    local run_dir="${PROJECT_ROOT}/saves/${exp_name}"
    local log_dir="${run_dir}/executer_logs"
    local output_dir="${AUDIT_ROOT}/${variant}_${exp_name}"
    local steps=()

    if [[ ! -d "${log_dir}" ]]; then
        echo "[cluster-100] skip audit ${variant}: missing ${log_dir}"
        return 0
    fi

    mapfile -t steps < <(discover_audit_steps "${log_dir}")
    if [[ "${#steps[@]}" -eq 0 ]]; then
        echo "[cluster-100] skip audit ${variant}: no step logs under ${log_dir}"
        return 0
    fi

    echo "[cluster-100] audit ${variant} steps=${steps[*]} output=${output_dir}"
    "${VENVPY}" "${PROJECT_ROOT}/scripts/audit_textcraft_rollout_similarity.py" \
        --log-dir "${log_dir}" \
        --output-dir "${output_dir}" \
        --steps "${steps[@]}" \
        --max-examples 10 \
        --max-trajectories-per-group 8 \
        --max-rounds-per-trajectory 12 \
        --max-raw-chars 3000

    echo -e "${variant}\taudit_dir\t${output_dir}" | tee -a "${MANIFEST}"
}

append_summary_header() {
    local later_schedule

    if [[ "${TEXTCRAFT_LATER_CLUSTER_EVERY}" == "0" ]]; then
        later_schedule="off (later_cluster_every=0; configured ${TEXTCRAFT_LATER_CANDIDATES} -> ${TEXTCRAFT_LATER_CLUSTERS} if re-enabled)"
    else
        later_schedule="${TEXTCRAFT_LATER_CANDIDATES} -> ${TEXTCRAFT_LATER_CLUSTERS}, every ${TEXTCRAFT_LATER_CLUSTER_EVERY} rounds, start ${TEXTCRAFT_LATER_CLUSTER_START}, horizon_min ${TEXTCRAFT_LATER_CLUSTER_HORIZON_MIN}"
    fi

    cat > "${SUMMARY_DOC}" <<EOF
# TextCraft 1.5B clustering 100-step run

Date: ${RUN_TS}

## Fixed configuration

| Field | Value |
|---|---|
| Model | \`${MODEL_PATH}\` |
| GPUs | \`${CUDA_VISIBLE_DEVICES}\` |
| TextCraft env | \`http://127.0.0.1:${PORT}\` |
| Total training steps | \`${TEXTCRAFT_TOTAL_TRAINING_STEPS}\` |
| G2RL reward shaping | \`${TEXTCRAFT_G2RL_ENABLED}\` |
| train_batch_size | \`${TEXTCRAFT_TRAIN_BATCH_SIZE}\` |
| rollout.n | \`${TEXTCRAFT_ROLLOUT_N}\` |
| ppo_mini_batch_size | \`${TEXTCRAFT_PPO_MINI_BATCH_SIZE}\` |
| ppo_micro_batch_size_per_gpu | \`${TEXTCRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU}\` |
| ppo_epochs | \`${TEXTCRAFT_PPO_EPOCHS}\` |
| lr | \`1e-6\` |
| kl_coef | \`0.001\` |
| rounds | \`${TEXTCRAFT_ROUNDS_SCHEDULE}\` |
| steps_scaling_inter | \`${TEXTCRAFT_ROUNDS_SCALING_INTER}\` |
| save_freq | \`${TEXTCRAFT_SAVE_FREQ}\` |
| test_freq | \`${TEXTCRAFT_TEST_FREQ}\` |
| test_batches | \`${TEXTCRAFT_TEST_BATCHES}\` |
| max_prompt_length | \`${TEXTCRAFT_MAX_PROMPT_LENGTH}\` |
| max_response_length | \`${TEXTCRAFT_MAX_RESPONSE_LENGTH}\` |
| rollout.max_tokens | \`${TEXTCRAFT_ROLLOUT_MAX_TOKENS}\` |
| round0 clustering schedule | \`${TEXTCRAFT_ROUND1_CANDIDATES} -> ${TEXTCRAFT_ROUND1_CLUSTERS}\` |
| later clustering schedule | \`${later_schedule}\` |
| high-cost rollout override | \`${TEXTCRAFT_ALLOW_HIGH_ROLLOUT:-0}\` |

## Variants

| ID | Experiment | Clustering | Method | Status | Save dir | Audit dir |
|---|---|---:|---|---|---|---|
EOF
}

append_summary_row() {
    local variant="$1"
    local exp_name="$2"
    local clustering="$3"
    local method="$4"
    local status="$5"
    local audit_dir="$6"
    local run_dir="${PROJECT_ROOT}/saves/${exp_name}"
    echo "| ${variant} | \`${exp_name}\` | \`${clustering}\` | \`${method}\` | ${status} | \`${run_dir}\` | \`${audit_dir}\` |" >> "${SUMMARY_DOC}"
}

run_variant() {
    local variant="$1"
    local config
    local exp_name
    local clustering_enabled
    local clustering_method
    local description
    local before_file
    local after_file
    local offline_run
    local run_dir

    config="$(variant_config "${variant}")"
    IFS='|' read -r exp_name clustering_enabled clustering_method description <<< "${config}"
    run_dir="${PROJECT_ROOT}/saves/${exp_name}"

    before_file="$(mktemp)"
    after_file="$(mktemp)"
    find "${PROJECT_ROOT}/wandb" -maxdepth 1 -type d -name 'offline-run-*' -print | sort > "${before_file}" || true

    echo "[cluster-100] train ${variant} start $(date '+%F %T')"
    echo "[cluster-100] ${description}"
    common_train_env
    export TEXTCRAFT_EXP_NAME="${exp_name}"
    export TEXTCRAFT_MODEL_SAVE_PATH="${run_dir}"
    export TEXTCRAFT_CLUSTERING_ENABLED="${clustering_enabled}"
    export TEXTCRAFT_CLUSTERING_METHOD="${clustering_method}"
    validate_rollout_budget "${variant}" "${clustering_enabled}"

    "${SCRIPT_DIR}/textcraft_train.sh"

    find "${PROJECT_ROOT}/wandb" -maxdepth 1 -type d -name 'offline-run-*' -print | sort > "${after_file}" || true
    offline_run="$(comm -13 "${before_file}" "${after_file}" | tail -n 1)"
    if [[ -z "${offline_run}" ]]; then
        offline_run="$(latest_offline_run "${PROJECT_ROOT}")"
    fi
    rm -f "${before_file}" "${after_file}"

    echo -e "${variant}\texp_name\t${exp_name}" | tee -a "${MANIFEST}"
    echo -e "${variant}\tsave_dir\t${run_dir}" | tee -a "${MANIFEST}"
    echo -e "${variant}\toffline_run\t${offline_run}" | tee -a "${MANIFEST}"
    echo "[cluster-100] train ${variant} done $(date '+%F %T') offline_run=${offline_run}"

    run_audit "${variant}" "${exp_name}"
    append_summary_row "${variant}" "${exp_name}" "${clustering_enabled}" "${clustering_method}" "done" "${AUDIT_ROOT}/${variant}_${exp_name}"
}

run_worker() {
    exec > >(tee -a "${LOG_FILE}") 2>&1
    echo "[cluster-100] start $(date '+%F %T')"
    echo "[cluster-100] session=${SESSION_NAME}"
    echo "[cluster-100] variants=${VARIANTS}"
    echo "[cluster-100] model=${MODEL_PATH}"
    echo "[cluster-100] log=${LOG_FILE}"
    echo "[cluster-100] manifest=${MANIFEST}"
    echo "[cluster-100] summary_doc=${SUMMARY_DOC}"
    echo "[cluster-100] audit_root=${AUDIT_ROOT}"

    : > "${MANIFEST}"
    common_train_env
    append_summary_header
    ensure_textcraft_env

    local variant
    for variant in ${VARIANTS}; do
        run_variant "${variant}"
    done

    cat >> "${SUMMARY_DOC}" <<EOF

## Artifacts

- Log: \`${LOG_FILE}\`
- Manifest: \`${MANIFEST}\`
- Audit root: \`${AUDIT_ROOT}\`

## Notes

- This run isolates rollout-time clustering. G2RL reward shaping is disabled in all variants.
- A1 controls for extra candidate sampling plus valid-action filtering.
- A2 is a gradient-multiview diagnostic; it is not the final G2RL clustering standard.
- A3 is the main G2RL rollout-time clustering test: k-center over the same normalized-action gradient feature formula used by G2RL.
- A4 is an optional quality-constrained diagnostic: unique normalized actions are preferred, but candidates are ranked by actor mean logprob.
- Default clustering is low-overhead: round0 \`${TEXTCRAFT_ROUND1_CANDIDATES} -> ${TEXTCRAFT_ROUND1_CLUSTERS}\`, later clustering disabled unless explicitly enabled.
- Existing old SciWorld pass@1 logs should not be reused as strict success evidence; see \`docs/sciworld_pass_at_1_audit_20260528.md\`.
EOF

    echo "[cluster-100] finished $(date '+%F %T')"
}

if [[ "${1:-}" == "--worker" ]]; then
    run_worker
    exit 0
fi

worker_env=(
    "RUN_TS=${RUN_TS}"
    "SESSION_NAME=${SESSION_NAME}"
    "LOG_FILE=${LOG_FILE}"
    "MANIFEST=${MANIFEST}"
    "SUMMARY_DOC=${SUMMARY_DOC}"
    "AUDIT_ROOT=${AUDIT_ROOT}"
    "PROJECT_ROOT=${PROJECT_ROOT}"
    "VENVPY=${VENVPY}"
    "AGENT_MODEL_PATH=${MODEL_PATH}"
    "TEXTCRAFT_ENV_PORT=${PORT}"
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    "TEXTCRAFT_N_GPUS_PER_NODE=${N_GPUS}"
    "VARIANTS=${VARIANTS}"
    "EXP_A0=${EXP_A0}"
    "EXP_A1=${EXP_A1}"
    "EXP_A2=${EXP_A2}"
    "EXP_A3=${EXP_A3}"
    "EXP_A4=${EXP_A4}"
    "TEXTCRAFT_TOTAL_TRAINING_STEPS=${TEXTCRAFT_TOTAL_TRAINING_STEPS:-100}"
    "TEXTCRAFT_TRAIN_BATCH_SIZE=${TEXTCRAFT_TRAIN_BATCH_SIZE:-32}"
    "TEXTCRAFT_ROLLOUT_N=${TEXTCRAFT_ROLLOUT_N:-8}"
    "TEXTCRAFT_VAL_BATCH_SIZE=${TEXTCRAFT_VAL_BATCH_SIZE:-32}"
    "TEXTCRAFT_PPO_MINI_BATCH_SIZE=${TEXTCRAFT_PPO_MINI_BATCH_SIZE:-8}"
    "TEXTCRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU=${TEXTCRAFT_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
    "TEXTCRAFT_PPO_EPOCHS=${TEXTCRAFT_PPO_EPOCHS:-2}"
    "TEXTCRAFT_TOTAL_EPOCHS=${TEXTCRAFT_TOTAL_EPOCHS:-30}"
    "TEXTCRAFT_MAX_PROMPT_LENGTH=${TEXTCRAFT_MAX_PROMPT_LENGTH:-512}"
    "TEXTCRAFT_MAX_RESPONSE_LENGTH=${TEXTCRAFT_MAX_RESPONSE_LENGTH:-10240}"
    "TEXTCRAFT_USE_REMOVE_PADDING=${TEXTCRAFT_USE_REMOVE_PADDING:-true}"
    "TEXTCRAFT_ROLLOUT_GPU_MEMORY_UTILIZATION=${TEXTCRAFT_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}"
    "TEXTCRAFT_ROLLOUT_MAX_MODEL_LEN=${TEXTCRAFT_ROLLOUT_MAX_MODEL_LEN:-32768}"
    "TEXTCRAFT_ROLLOUT_MAX_NUM_BATCHED_TOKENS=${TEXTCRAFT_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"
    "TEXTCRAFT_ROLLOUT_MAX_TOKENS=${TEXTCRAFT_ROLLOUT_MAX_TOKENS:-512}"
    "TEXTCRAFT_PPO_MAX_TOKEN_LEN_PER_GPU=${TEXTCRAFT_PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
    "TEXTCRAFT_ENTROPY_CHUNK_SIZE=${TEXTCRAFT_ENTROPY_CHUNK_SIZE:-256}"
    "TEXTCRAFT_EARLY_STOP_ENABLED=${TEXTCRAFT_EARLY_STOP_ENABLED:-true}"
    "TEXTCRAFT_EARLY_STOP_MIN_STEPS=${TEXTCRAFT_EARLY_STOP_MIN_STEPS:-100}"
    "TEXTCRAFT_EARLY_STOP_PATIENCE=${TEXTCRAFT_EARLY_STOP_PATIENCE:-5}"
    "TEXTCRAFT_EARLY_STOP_MIN_DELTA=${TEXTCRAFT_EARLY_STOP_MIN_DELTA:-0.005}"
    "TEXTCRAFT_EARLY_STOP_MAX_DROP=${TEXTCRAFT_EARLY_STOP_MAX_DROP:-0.08}"
    "TEXTCRAFT_EARLY_STOP_HARD_MIN_METRIC=${TEXTCRAFT_EARLY_STOP_HARD_MIN_METRIC:-0.20}"
    "TEXTCRAFT_EARLY_STOP_MIN_ROLLOUT_VALID_RATIO=${TEXTCRAFT_EARLY_STOP_MIN_ROLLOUT_VALID_RATIO:-0.20}"
    "TEXTCRAFT_EARLY_STOP_MAX_RESPONSE_LENGTH_MEAN=${TEXTCRAFT_EARLY_STOP_MAX_RESPONSE_LENGTH_MEAN:-1500}"
    "TEXTCRAFT_EARLY_STOP_MAX_KL_LOSS=${TEXTCRAFT_EARLY_STOP_MAX_KL_LOSS:-1.0}"
    "TEXTCRAFT_EARLY_STOP_MAX_GRAD_NORM=${TEXTCRAFT_EARLY_STOP_MAX_GRAD_NORM:-100000000}"
    "TEXTCRAFT_ROUNDS_CTRL_TYPE=${TEXTCRAFT_ROUNDS_CTRL_TYPE:-scaling_inter_stepwise}"
    "TEXTCRAFT_ROUNDS_SCHEDULE=${TEXTCRAFT_ROUNDS_SCHEDULE:-[10,20,30]}"
    "TEXTCRAFT_ROUNDS_SCALING_INTER=${TEXTCRAFT_ROUNDS_SCALING_INTER:-100}"
    "TEXTCRAFT_SAVE_FREQ=${TEXTCRAFT_SAVE_FREQ:-25}"
    "TEXTCRAFT_TEST_FREQ=${TEXTCRAFT_TEST_FREQ:-35}"
    "TEXTCRAFT_TEST_BATCHES=${TEXTCRAFT_TEST_BATCHES:-1}"
    "TEXTCRAFT_G2RL_ENABLED=${TEXTCRAFT_G2RL_ENABLED:-false}"
    "TEXTCRAFT_G2RL_FEATURE_SCOPE=${TEXTCRAFT_G2RL_FEATURE_SCOPE:-normalized_action}"
    "TEXTCRAFT_ROUND1_CANDIDATES=${TEXTCRAFT_ROUND1_CANDIDATES:-16}"
    "TEXTCRAFT_ROUND1_CLUSTERS=${TEXTCRAFT_ROUND1_CLUSTERS:-8}"
    "TEXTCRAFT_LATER_CANDIDATES=${TEXTCRAFT_LATER_CANDIDATES:-4}"
    "TEXTCRAFT_LATER_CLUSTERS=${TEXTCRAFT_LATER_CLUSTERS:-1}"
    "TEXTCRAFT_LATER_CLUSTER_EVERY=${TEXTCRAFT_LATER_CLUSTER_EVERY:-0}"
    "TEXTCRAFT_LATER_CLUSTER_START=${TEXTCRAFT_LATER_CLUSTER_START:-1}"
    "TEXTCRAFT_LATER_CLUSTER_UNTIL=${TEXTCRAFT_LATER_CLUSTER_UNTIL:--1}"
    "TEXTCRAFT_LATER_CLUSTER_HORIZON_MIN=${TEXTCRAFT_LATER_CLUSTER_HORIZON_MIN:-0.25}"
    "TEXTCRAFT_GRADIENT_D_PROJ=${TEXTCRAFT_GRADIENT_D_PROJ:-512}"
    "TEXTCRAFT_FEATURE_TOPK=${TEXTCRAFT_FEATURE_TOPK:-256}"
    "TEXTCRAFT_FEATURE_CHUNK_SIZE=${TEXTCRAFT_FEATURE_CHUNK_SIZE:-4}"
    "TEXTCRAFT_ALLOW_HIGH_ROLLOUT=${TEXTCRAFT_ALLOW_HIGH_ROLLOUT:-0}"
)

worker_cmd=""
for env_pair in "${worker_env[@]}"; do
    worker_cmd+="$(printf '%q' "${env_pair}") "
done
worker_cmd+="bash $(printf '%q' "${SCRIPT_PATH}") --worker"

tmux new-session -d -s "${SESSION_NAME}" "${worker_cmd}"

echo "Started tmux session: ${SESSION_NAME}"
echo "Queue log: ${LOG_FILE}"
echo "Manifest: ${MANIFEST}"
echo "Summary doc: ${SUMMARY_DOC}"
echo "Audit root: ${AUDIT_ROOT}"
echo "A0 save dir: ${PROJECT_ROOT}/saves/${EXP_A0}"
echo "A1 save dir: ${PROJECT_ROOT}/saves/${EXP_A1}"
echo "A2 save dir: ${PROJECT_ROOT}/saves/${EXP_A2}"
echo "A3 save dir: ${PROJECT_ROOT}/saves/${EXP_A3}"
echo "A4 save dir: ${PROJECT_ROOT}/saves/${EXP_A4}"
