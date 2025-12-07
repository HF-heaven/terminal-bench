#!/bin/bash
# Quick test of representative PIXIU datasets (one from each major category)
set -euo pipefail

OUTPUT_DIR="/tmp/pixiu_quick_test"
ADAPTER_DIR="/home/wendy/terminal-bench/adapters/pixiu"

echo "=========================================="
echo "Quick Test: Representative PIXIU Datasets"
echo "=========================================="
echo ""

# Clean and create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Representative datasets (one per major category)
declare -a DATASETS=(
    "TheFinAI/flare-cfa"           # Classification
    "TheFinAI/flare-tsa"           # Regression
    "TheFinAI/flare-finqa"         # Financial QA
    "TheFinAI/flare-tatqa"         # Table QA
    "TheFinAI/flare-ner"           # NER
    "TheFinAI/flare-fnxl"          # Token classification
    "TheFinAI/flare-fsrl"          # Semantic role labeling
    "TheFinAI/flare-cd"            # Sequence labeling
    "TheFinAI/flare-finred"        # Relation extraction
    "TheFinAI/flare-ectsum"        # Extractive summarization
    "TheFinAI/flare-edtsum"        # Abstractive summarization
)

SUCCESS=0
FAILED=0
TOTAL=${#DATASETS[@]}

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    num=$((i + 1))
    
    echo "[$num/$TOTAL] Testing: $dataset"
    
    ~/.local/bin/uv run python run_adapter.py --harbor \
        --dataset-name "$dataset" \
        --limit 1 \
        --output-path "$OUTPUT_DIR" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        ((SUCCESS++))
        echo "  ✅ PASS"
    else
        ((FAILED++))
        echo "  ❌ FAIL"
    fi
done

echo ""
echo "=========================================="
echo "Final Results"
echo "=========================================="
echo "Total categories tested: $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ ALL 11 TASK CATEGORIES WORKING!"
    echo ""
    echo "Validating structure..."
    if "$ADAPTER_DIR/validate_harbor_format.sh" "$OUTPUT_DIR" 2>&1 | tail -5; then
        echo ""
        echo "✅ Structure validation passed!"
    fi
    
    echo ""
    echo "Sample dynamic instructions from each category:"
    echo "=========================================="
    
    for task_dir in "$OUTPUT_DIR"/pixiu-*/; do
        [ -d "$task_dir" ] || continue
        task_name=$(basename "$task_dir")
        echo ""
        echo "📄 $task_name:"
        echo "--------------------"
        head -12 "$task_dir/instruction.md" | tail -8
    done
    
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    exit 1
fi
