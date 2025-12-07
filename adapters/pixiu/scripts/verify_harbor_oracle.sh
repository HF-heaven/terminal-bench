#!/bin/bash
# PIXIU Harbor Adapter - Oracle Verification Script
# Tests all generated Harbor tasks with oracle solution

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TASK_DIR="${1:-/tmp/pixiu_harbor_clean_test}"

echo -e "${BLUE}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  PIXIU Harbor Adapter - Oracle Verification  ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════╝${NC}"
echo ""

if [ ! -d "$TASK_DIR" ]; then
    echo -e "${RED}✗ Task directory not found: $TASK_DIR${NC}"
    echo "  Run verify_clean_generation.sh first"
    exit 1
fi

# Count tasks
total_tasks=$(find "$TASK_DIR" -mindepth 1 -maxdepth 1 -type d -name "pixiu-*" | wc -l)
echo "Testing $total_tasks Harbor tasks..."
echo "Task directory: $TASK_DIR"
echo ""

# Test each task
declare -A dataset_pass
declare -A dataset_total
pass_count=0
fail_count=0
failed_tasks=()

for task_path in "$TASK_DIR"/pixiu-*; do
    [ ! -d "$task_path" ] && continue
    
    task_name=$(basename "$task_path")
    dataset=$(echo "$task_name" | sed -E 's/^pixiu-([^-]+)-.*$/\1/')
    
    dataset_total[$dataset]=$((${dataset_total[$dataset]:-0} + 1))
    
    echo -n "Testing $task_name... "
    
    cd "$task_path"
    
    # Build Docker image
    if ! docker build -t "test-$task_name" . >/dev/null 2>&1; then
        echo -e "${RED}✗ BUILD FAILED${NC}"
        ((fail_count++))
        failed_tasks+=("$task_name (build failed)")
        cd - >/dev/null
        continue
    fi
    
    # Run oracle test
    test_output=$(docker run --rm "test-$task_name" bash -c '
        cd /app && \
        bash solution/solve.sh && \
        python3 -m pytest tests/ -v && \
        echo "1.0" > /logs/verifier/reward.txt && \
        cat /logs/verifier/reward.txt
    ' 2>&1 || true)
    
    # Check result
    if echo "$test_output" | tail -1 | grep -q "^1.0$"; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((pass_count++))
        dataset_pass[$dataset]=$((${dataset_pass[$dataset]:-0} + 1))
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((fail_count++))
        failed_tasks+=("$task_name")
        # Save failure log
        echo "$test_output" > "/tmp/pixiu_fail_${task_name}.log"
    fi
    
    # Cleanup
    docker rmi "test-$task_name" >/dev/null 2>&1 || true
    cd - >/dev/null
done

echo ""
echo "════════════════════════════════════════"
echo "Results by dataset:"
for dataset in $(echo "${!dataset_total[@]}" | tr ' ' '\n' | sort); do
    pass=${dataset_pass[$dataset]:-0}
    total=${dataset_total[$dataset]}
    pct=$((pass * 100 / total))
    
    if [ $pass -eq $total ]; then
        echo -e "  ${GREEN}$dataset: $pass/$total ($pct%)${NC}"
    else
        echo -e "  ${RED}$dataset: $pass/$total ($pct%)${NC}"
    fi
done

echo ""
echo "════════════════════════════════════════"
pct=$((pass_count * 100 / total_tasks))
echo -e "Overall: ${GREEN}$pass_count${NC}/${BLUE}$total_tasks${NC} (${BLUE}$pct%${NC})"

if [ $fail_count -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed tasks ($fail_count):${NC}"
    for task in "${failed_tasks[@]}"; do
        echo "  - $task"
    done
    echo ""
    echo "Failure logs saved to: /tmp/pixiu_fail_*.log"
    exit 1
else
    echo ""
    echo -e "${GREEN}✓ All tasks passed!${NC}"
fi
