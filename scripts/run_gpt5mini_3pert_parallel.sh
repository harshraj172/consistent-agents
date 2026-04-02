#!/bin/bash
# Run GPT5mini 3-pert evals in parallel pairs with disk monitoring
# Usage: bash scripts/run_gpt5mini_3pert_parallel.sh [--resume]

set -e

RESUME_FLAG=""
if [ "$1" = "--resume" ]; then
    RESUME_FLAG="--resume"
fi

cleanup_docker() {
    echo "[$(date)] Docker cleanup..."
    docker container prune -f 2>/dev/null || true
    USAGE=$(df / | tail -1 | awk '{print $5}' | tr -d '%')
    echo "[$(date)] Disk: ${USAGE}%"
    if [ "$USAGE" -gt 90 ]; then
        echo "[$(date)] CRITICAL >90%"
        docker image prune -a --filter "until=5m" -f 2>/dev/null || true
    elif [ "$USAGE" -gt 80 ]; then
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
trap "kill $MONITOR_PID 2>/dev/null; wait" EXIT

run_pair() {
    local cfg1=$1 cfg2=$2
    local resume1="" resume2=""

    # Find resume dirs if flag set
    if [ -n "$RESUME_FLAG" ]; then
        local out1=$(grep "path:" "$cfg1" | tail -1 | awk '{print $2}')
        local out2=$(grep "path:" "$cfg2" | tail -1 | awk '{print $2}')
        local rd1=$(ls -dt ${out1}/*/ 2>/dev/null | head -1)
        local rd2=$(ls -dt ${out2}/*/ 2>/dev/null | head -1)
        [ -n "$rd1" ] && resume1="--resume $rd1"
        [ -n "$rd2" ] && resume2="--resume $rd2"
    fi

    echo ""
    echo "=============================================="
    echo "[$(date)] PARALLEL: $(basename $cfg1) + $(basename $cfg2)"
    echo "=============================================="
    cleanup_docker

    python -m consistent_agents.eval_harbor "$cfg1" $resume1 2>&1 | tee "logs_3pert_$(basename $cfg1 .yaml).log" &
    PID1=$!
    python -m consistent_agents.eval_harbor "$cfg2" $resume2 2>&1 | tee "logs_3pert_$(basename $cfg2 .yaml).log" &
    PID2=$!

    wait $PID1
    echo "[$(date)] Completed: $(basename $cfg1)"
    wait $PID2
    echo "[$(date)] Completed: $(basename $cfg2)"
    cleanup_docker
}

run_single() {
    local cfg1=$1
    local resume1=""
    if [ -n "$RESUME_FLAG" ]; then
        local out1=$(grep "path:" "$cfg1" | tail -1 | awk '{print $2}')
        local rd1=$(ls -dt ${out1}/*/ 2>/dev/null | head -1)
        [ -n "$rd1" ] && resume1="--resume $rd1"
    fi

    echo ""
    echo "=============================================="
    echo "[$(date)] SINGLE: $(basename $cfg1)"
    echo "=============================================="
    cleanup_docker
    python -m consistent_agents.eval_harbor "$cfg1" $resume1 2>&1 | tee "logs_3pert_$(basename $cfg1 .yaml).log"
    echo "[$(date)] Completed: $(basename $cfg1)"
    cleanup_docker
}

# Pair 1: injection_swe + noise
run_pair \
    "src/consistent_agents/config/gpt5mini-swebench-3pert/injection_swe.yaml" \
    "src/consistent_agents/config/gpt5mini-swebench-3pert/noise.yaml"

# Pair 2: paraphrase + translation
run_pair \
    "src/consistent_agents/config/gpt5mini-swebench-3pert/paraphrase.yaml" \
    "src/consistent_agents/config/gpt5mini-swebench-3pert/translation.yaml"

# Last: var_rename
run_single \
    "src/consistent_agents/config/gpt5mini-swebench-3pert/var_rename.yaml"

echo ""
echo "[$(date)] All 3-perturbation evaluations complete!"
