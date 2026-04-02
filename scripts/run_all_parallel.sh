#!/bin/bash
# Run ALL 5 perturbation types in parallel with disk monitoring
# Each with n_concurrent=2, total ~10 concurrent examples

set -e

cleanup_docker() {
    docker container prune -f >/dev/null 2>&1 || true
    USAGE=$(df / | tail -1 | awk '{print $5}' | tr -d '%')
    if [ "$USAGE" -gt 90 ]; then
        echo "[$(date)] CRITICAL ${USAGE}% — aggressive prune"
        docker image prune -a --filter "until=5m" -f >/dev/null 2>&1 || true
    elif [ "$USAGE" -gt 80 ]; then
        echo "[$(date)] WARNING ${USAGE}% — pruning images"
        docker image prune -a --filter "until=30m" -f >/dev/null 2>&1 || true
    elif [ "$USAGE" -gt 60 ]; then
        docker image prune -f --filter "until=1h" >/dev/null 2>&1 || true
    fi
    USAGE_AFTER=$(df / | tail -1 | awk '{print $5}' | tr -d '%')
    echo "[$(date)] Disk: ${USAGE_AFTER}%"
}

# Background disk monitor every 5 minutes (aggressive since we're running 5 in parallel)
(
    while true; do
        sleep 300
        cleanup_docker
    done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null; wait" EXIT

echo "[$(date)] Launching all 5 perturbation types in parallel"
cleanup_docker

# injection_swe (resume from 5 completed)
RESUME_DIR=$(ls -dt outputs/harbor-swebench-codex-gpt5mini-3pert-injection_swe/*/ 2>/dev/null | head -1)
python -m consistent_agents.eval_harbor \
    src/consistent_agents/config/gpt5mini-swebench-3pert/injection_swe.yaml \
    --resume "$RESUME_DIR" \
    > logs_3pert_injection_swe.log 2>&1 &
PID_INJ=$!
echo "[$(date)] Started injection_swe (PID $PID_INJ, resuming)"

# noise
python -m consistent_agents.eval_harbor \
    src/consistent_agents/config/gpt5mini-swebench-3pert/noise.yaml \
    > logs_3pert_noise.log 2>&1 &
PID_NOISE=$!
echo "[$(date)] Started noise (PID $PID_NOISE)"

# paraphrase
python -m consistent_agents.eval_harbor \
    src/consistent_agents/config/gpt5mini-swebench-3pert/paraphrase.yaml \
    > logs_3pert_paraphrase.log 2>&1 &
PID_PARA=$!
echo "[$(date)] Started paraphrase (PID $PID_PARA)"

# translation
python -m consistent_agents.eval_harbor \
    src/consistent_agents/config/gpt5mini-swebench-3pert/translation.yaml \
    > logs_3pert_translation.log 2>&1 &
PID_TRANS=$!
echo "[$(date)] Started translation (PID $PID_TRANS)"

# var_rename
python -m consistent_agents.eval_harbor \
    src/consistent_agents/config/gpt5mini-swebench-3pert/var_rename.yaml \
    > logs_3pert_var_rename.log 2>&1 &
PID_VAR=$!
echo "[$(date)] Started var_rename (PID $PID_VAR)"

echo ""
echo "[$(date)] All 5 launched. PIDs: $PID_INJ $PID_NOISE $PID_PARA $PID_TRANS $PID_VAR"
echo "[$(date)] Waiting for all to complete..."

# Wait and report as each finishes
PIDS=("$PID_INJ" "$PID_NOISE" "$PID_PARA" "$PID_TRANS" "$PID_VAR")
NAMES=("injection_swe" "noise" "paraphrase" "translation" "var_rename")
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" && echo "[$(date)] DONE: ${NAMES[$i]}" || echo "[$(date)] FAILED: ${NAMES[$i]} (exit $?)"
done

echo ""
echo "[$(date)] ALL EVALUATIONS COMPLETE!"
cleanup_docker
