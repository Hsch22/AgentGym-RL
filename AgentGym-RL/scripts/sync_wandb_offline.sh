#!/usr/bin/env bash
# Sync all offline wandb runs under wandb/ to the cloud, newest first.
# Progress is appended to wandb/sync_progress_<ts>.log.

if [[ -n "${WANDB_SYNC_HTTP_PROXY:-}" ]]; then
  export http_proxy="$WANDB_SYNC_HTTP_PROXY"
  export https_proxy="$WANDB_SYNC_HTTP_PROXY"
fi
unset WANDB_X_REQUIRE_LEGACY_SERVICE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

WB="${WB:-${REPO_ROOT}/.venv/bin/wandb}"
WANDB_DIR="${WANDB_DIR:-${PROJECT_ROOT}/wandb}"
LOG="$WANDB_DIR/sync_progress_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

ts() { date '+%F %T'; }

echo "[$(ts)] log file : $LOG"
echo "[$(ts)] wandb bin: $WB"
echo "[$(ts)] wandb dir: $WANDB_DIR"
echo

echo "[$(ts)] === wandb login ==="
if [ -n "${WANDB_API_KEY:-}" ]; then
  "$WB" login "${WANDB_API_KEY}"
else
  echo "[$(ts)] WANDB_API_KEY is not set; using existing wandb credentials"
fi
echo

echo "[$(ts)] === network test: curl -I www.google.com ==="
curl -I --max-time 15 www.google.com
echo

echo "[$(ts)] === enumerating offline runs (newest first) ==="
mapfile -t RUNS < <(ls -1d "$WANDB_DIR"/offline-run-*/ 2>/dev/null | sort -r)
printf '  %s\n' "${RUNS[@]}"
echo "  total: ${#RUNS[@]}"
echo

ok=0
fail=0
skip=0
for r in "${RUNS[@]}"; do
  name=$(basename "$r")
  echo
  echo "[$(ts)] ---- SYNC START: $name ----"
  if ls "$r"/files/*.synced >/dev/null 2>&1; then
    echo "[$(ts)] already synced (found .synced marker), skipping"
    skip=$((skip+1))
    continue
  fi
  t0=$SECONDS
  "$WB" sync "$r"
  rc=$?
  dt=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then
    echo "[$(ts)] ---- SYNC OK : $name (${dt}s) ----"
    ok=$((ok+1))
  else
    echo "[$(ts)] ---- SYNC FAIL(exit=$rc): $name (${dt}s) ----"
    fail=$((fail+1))
  fi
done

echo
echo "[$(ts)] === all done : ok=$ok fail=$fail skip=$skip total=${#RUNS[@]} ==="
