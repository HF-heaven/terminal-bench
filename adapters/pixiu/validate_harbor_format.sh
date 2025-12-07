#!/bin/bash
# Validate Harbor format structure for PIXIU tasks
# Usage: ./validate_harbor_format.sh <dataset_directory>

set -euo pipefail

DATASET_DIR="${1:-datasets/pixiu}"
ERRORS=0
WARNINGS=0

echo "Validating Harbor format in: $DATASET_DIR"
echo "=================================================="

# Check if directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Directory $DATASET_DIR does not exist"
    exit 1
fi

# Count total tasks
TOTAL_TASKS=$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "Found $TOTAL_TASKS tasks to validate"
echo ""

for task_dir in "$DATASET_DIR"/*/ ; do
    [ -d "$task_dir" ] || continue
    task_name=$(basename "$task_dir")
    
    # Required Harbor format files
    required_files=(
        "task.toml"
        "instruction.md"
        "environment/Dockerfile"
        "solution/solve.sh"
        "tests/test.sh"
        "tests/test_outputs.py"
    )
    
    # Check required files exist
    for file in "${required_files[@]}"; do
        if [ ! -f "$task_dir$file" ]; then
            echo "❌ ERROR: $task_name missing $file"
            ((ERRORS++))
        fi
    done
    
    # Check old Terminal-Bench files don't exist
    old_files=(
        "task.yaml"
        "Dockerfile"
        "docker-compose.yaml"
        "solution.sh"
        "run-tests.sh"
    )
    
    for file in "${old_files[@]}"; do
        if [ -f "$task_dir$file" ]; then
            echo "⚠️  WARNING: $task_name has old Terminal-Bench file: $file"
            ((WARNINGS++))
        fi
    done
    
    # Check test.sh contains reward writing
    if [ -f "$task_dir/tests/test.sh" ]; then
        if ! grep -q "/logs/verifier/reward.txt" "$task_dir/tests/test.sh"; then
            echo "❌ ERROR: $task_name test.sh missing reward.txt writing"
            ((ERRORS++))
        fi
    fi
    
    # Check task.toml is valid TOML (basic check)
    if [ -f "$task_dir/task.toml" ]; then
        if ! grep -q "^\[metadata\]" "$task_dir/task.toml"; then
            echo "⚠️  WARNING: $task_name task.toml may not have [metadata] section"
            ((WARNINGS++))
        fi
        if ! grep -q "^\[agent\]" "$task_dir/task.toml"; then
            echo "⚠️  WARNING: $task_name task.toml may not have [agent] section"
            ((WARNINGS++))
        fi
        if ! grep -q "^\[verifier\]" "$task_dir/task.toml"; then
            echo "⚠️  WARNING: $task_name task.toml may not have [verifier] section"
            ((WARNINGS++))
        fi
    fi
    
    # Check instruction.md is not empty
    if [ -f "$task_dir/instruction.md" ]; then
        if [ ! -s "$task_dir/instruction.md" ]; then
            echo "❌ ERROR: $task_name instruction.md is empty"
            ((ERRORS++))
        fi
    fi
done

echo ""
echo "=================================================="
echo "Validation complete!"
echo "Total tasks checked: $TOTAL_TASKS"
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"

if [ $ERRORS -gt 0 ]; then
    echo "❌ Validation FAILED with $ERRORS errors"
    exit 1
else
    echo "✅ Validation PASSED"
    if [ $WARNINGS -gt 0 ]; then
        echo "⚠️  Note: $WARNINGS warnings found (review recommended)"
    fi
    exit 0
fi
