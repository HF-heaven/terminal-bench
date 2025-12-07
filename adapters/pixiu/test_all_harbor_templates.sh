#!/bin/bash
# Test Harbor format generation for all PIXIU task types
# This script generates 5 sample tasks for each major task type to validate the migration

set -euo pipefail

OUTPUT_BASE="/tmp/harbor_validation_test"
ADAPTER_DIR="/home/wendy/terminal-bench/adapters/pixiu"

echo "=================================================="
echo "PIXIU Harbor Format Validation Test"
echo "=================================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Task types to test with their datasets
declare -A DATASETS=(
    ["classification"]="TheFinAI/flare-cfa"
    ["regression"]="TheFinAI/flare-tsa"
    ["finqa"]="TheFinAI/flare-finqa"
    ["tatqa"]="TheFinAI/flare-tatqa"
    ["fnxl"]="TheFinAI/flare-fnxl"
    ["fsrl"]="TheFinAI/flare-fsrl"
    ["ectsum"]="TheFinAI/flare-ectsum"
    ["edtsum"]="TheFinAI/flare-edtsum"
    ["ner"]="TheFinAI/flare-ner"
    ["seqlabel"]="TheFinAI/flare-cd"
    ["relext"]="TheFinAI/flare-finred"
)

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

for task_type in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$task_type]}"
    output_dir="$OUTPUT_BASE/test_$task_type"
    
    echo "Testing: $task_type ($dataset)"
    echo "  Generating 5 tasks..."
    
    if python "$ADAPTER_DIR/run_adapter.py" --harbor \
        --dataset-name "$dataset" \
        --limit 5 \
        --output-path "$output_dir" > /dev/null 2>&1; then
        
        echo "  ✅ Generation successful"
        
        # Validate structure
        echo "  Validating structure..."
        if "$ADAPTER_DIR/validate_harbor_format.sh" "$output_dir" > /dev/null 2>&1; then
            echo "  ✅ Validation passed"
            ((PASSED_TESTS++))
        else
            echo "  ❌ Validation failed"
            ((FAILED_TESTS++))
        fi
    else
        echo "  ❌ Generation failed"
        ((FAILED_TESTS++))
    fi
    
    ((TOTAL_TESTS++))
    echo ""
done

echo "=================================================="
echo "Test Summary"
echo "=================================================="
echo "Total task types tested: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $FAILED_TESTS"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo "✅ ALL TESTS PASSED!"
    echo ""
    echo "Next steps:"
    echo "1. Generate full datasets for each task type"
    echo "2. Run oracle tests on 5-10 tasks per type"
    echo "3. Submit to harbor-datasets repository"
    exit 0
else
    echo "❌ SOME TESTS FAILED - Review errors above"
    exit 1
fi
