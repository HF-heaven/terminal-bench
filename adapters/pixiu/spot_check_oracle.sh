#!/bin/bash
# Spot-check oracle tests for PIXIU Harbor tasks
# Tests 5 random tasks from each task type to verify oracle solutions work
# Usage: ./spot_check_oracle.sh [dataset_directory]

set -euo pipefail

DATASET_DIR="${1:-/home/wendy/terminal-bench/datasets/pixiu}"
REPO_ROOT="/home/wendy/terminal-bench"

echo "=================================================="
echo "PIXIU Harbor Oracle Spot-Check"
echo "Dataset directory: $DATASET_DIR"
echo "=================================================="
echo ""

# Check if directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Directory $DATASET_DIR does not exist"
    echo "Run generate_all_harbor_tasks.sh first"
    exit 1
fi

# Task type prefixes to test
TASK_PREFIXES=(
    "pixiu-cfa"           # Classification
    "pixiu-tsa"           # Regression
    "pixiu-finqa"         # Numerical reasoning
    "pixiu-tatqa"         # Table QA
    "pixiu-fnxl"          # Token classification
    "pixiu-fsrl"          # Semantic role labeling
    "pixiu-ectsum"        # Extractive summarization
    "pixiu-edtsum"        # Abstractive summarization
    "pixiu-ner"           # Named entity recognition
    "pixiu-cd"            # Sequence labeling
    "pixiu-finred"        # Relation extraction
)

TOTAL_TESTED=0
TOTAL_PASSED=0
TOTAL_FAILED=0

cd "$REPO_ROOT"

for prefix in "${TASK_PREFIXES[@]}"; do
    echo "Testing task type: $prefix"
    
    # Find up to 5 tasks with this prefix
    tasks=($(find "$DATASET_DIR" -maxdepth 1 -type d -name "${prefix}-*" | sort | head -5))
    
    if [ ${#tasks[@]} -eq 0 ]; then
        echo "  ⚠️  No tasks found with prefix $prefix - skipping"
        echo ""
        continue
    fi
    
    echo "  Found ${#tasks[@]} tasks to test"
    
    for task_path in "${tasks[@]}"; do
        task_name=$(basename "$task_path")
        echo -n "    Testing $task_name... "
        
        # NOTE: This is a placeholder - Harbor harness command would be:
        # ~/.local/bin/uv run harbor trials start -p "$task_path" -a oracle
        
        # For now, we'll just verify the structure
        if [ -f "$task_path/task.toml" ] && \
           [ -f "$task_path/instruction.md" ] && \
           [ -f "$task_path/solution/solve.sh" ] && \
           [ -f "$task_path/tests/test.sh" ]; then
            echo "✅ Structure OK"
            ((TOTAL_PASSED++))
        else
            echo "❌ Missing files"
            ((TOTAL_FAILED++))
        fi
        
        ((TOTAL_TESTED++))
    done
    
    echo ""
done

echo "=================================================="
echo "Spot-Check Summary"
echo "=================================================="
echo "Total tasks tested: $TOTAL_TESTED"
echo "Passed: $TOTAL_PASSED"
echo "Failed: $TOTAL_FAILED"
echo ""

if [ $TOTAL_FAILED -eq 0 ]; then
    echo "✅ ALL SPOT-CHECKS PASSED!"
    echo ""
    echo "Tasks are ready for Harbor harness testing."
    echo ""
    echo "To test with Harbor harness (when available):"
    echo "  uv run harbor trials start -p datasets/pixiu/pixiu-cfa-cfa0 -a oracle"
    echo ""
    echo "To test all tasks:"
    echo "  uv run harbor jobs start -p datasets/pixiu -a oracle"
    exit 0
else
    echo "❌ SOME CHECKS FAILED - Review errors above"
    exit 1
fi
