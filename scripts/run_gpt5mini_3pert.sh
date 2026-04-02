#!/bin/bash
# Run GPT5mini with 3 perturbations per example, all perturbation types
# Aggressive concurrency, disk < 90%

set -e

CONFIGS=(
    "src/consistent_agents/config/gpt5mini-swebench-3pert/injection_swe.yaml"
    "src/consistent_agents/config/gpt5mini-swebench-3pert/noise.yaml"
    "src/consistent_agents/config/gpt5mini-swebench-3pert/paraphrase.yaml"
    "src/consistent_agents/config/gpt5mini-swebench-3pert/translation.yaml"
    "src/consistent_agents/config/gpt5mini-swebench-3pert/var_rename.yaml"
)

cleanup_docker() {
    echo "[$(date)] Docker cleanup..."
    docker container prune -f 2>/dev/null || true
    USAGE=$(df / | tail -1 | awk '{print $5}' | tr -d '%')
    echo "[$(date)] Disk: ${USAGE}%"
    if [ "$USAGE" -gt 90 ]; then
        echo "[$(date)] CRITICAL >90% — aggressive cleanup"
        docker image prune -a --filter "until=5m" -f 2>/dev/null || true
    elif [ "$USAGE" -gt 80 ]; then
        echo "[$(date)] WARNING >80%"
        docker image prune -a --filter "until=30m" -f 2>/dev/null || true
    elif [ "$USAGE" -gt 60 ]; then
        docker image prune -f --filter "until=1h" 2>/dev/null || true
    fi
}

# Background disk monitor every 10 minutes
(
    while true; do
        sleep 600
        cleanup_docker
    done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null" EXIT

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "=============================================="
    echo "[$(date)] Starting: $(basename $cfg)"
    echo "=============================================="
    cleanup_docker
    python -m consistent_agents.eval_harbor "$cfg" 2>&1 | tee "logs_3pert_$(basename "$cfg" .yaml).log"
    echo "[$(date)] Completed: $(basename $cfg)"
    cleanup_docker
done

echo ""
echo "[$(date)] All 3-perturbation evaluations complete!"
