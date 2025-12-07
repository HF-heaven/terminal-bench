#!/bin/bash
# Comprehensive test of ALL 29 PIXIU datasets with dynamic instruction generation
set -euo pipefail

OUTPUT_DIR="/tmp/pixiu_29_test"
ADAPTER_DIR="/home/wendy/terminal-bench/adapters/pixiu"

echo "=========================================="
echo "Testing ALL 29 PIXIU Dataset Types"
echo "=========================================="
echo ""

# Clean and create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# All 29 datasets (organized by category)
declare -a DATASETS=(
    # Classification (18)
    "TheFinAI/flare-headlines"
    "TheFinAI/en-fpb"
    "TheFinAI/flare-causal20-sc"
    "TheFinAI/flare-fiqasa"
    "TheFinAI/finben-fomc"
    "TheFinAI/flare-cfa"
    "TheFinAI/flare-german"
    "daishen/cra-ccfraud"
    "TheFinAI/flare-australian"
    "daishen/cra-ccf"
    "daishen/cra-taiwan"
    "TheFinAI/en-forecasting-travelinsurance"
    "TheFinAI/flare-mlesg"
    "TheFinAI/flare-ma"
    "TheFinAI/flare-multifin-en"
    "TheFinAI/flare-sm-acl"
    "TheFinAI/flare-sm-bigdata"
    "TheFinAI/flare-sm-cikm"
    # QA (2)
    "TheFinAI/flare-finqa"
    "TheFinAI/flare-tatqa"
    # Regression (1)
    "TheFinAI/flare-tsa"
    # Token/NER (6)
    "TheFinAI/flare-ner"
    "TheFinAI/flare-fnxl"
    "TheFinAI/flare-fsrl"
    "TheFinAI/finben-finer-ord"
    "TheFinAI/flare-cd"
    "TheFinAI/flare-finred"
    # Summarization (2)
    "TheFinAI/flare-ectsum"
    "TheFinAI/flare-edtsum"
)

SUCCESS=0
FAILED=0
TOTAL=${#DATASETS[@]}

for i in "${!DATASETS[@]}"; do
    dataset="${DATASETS[$i]}"
    num=$((i + 1))
    
    echo "[$num/$TOTAL] Testing: $dataset"
    
    if ~/.local/bin/uv run python run_adapter.py --harbor \
        --dataset-name "$dataset" \
        --limit 1 \
        --output-path "$OUTPUT_DIR" 2>&1 | grep -q "Generated.*PIXIU Harbor tasks"; then
        ((SUCCESS++))
        echo "  ✅ PASS"
    else
        ((FAILED++))
        echo "  ❌ FAIL"
    fi
    echo ""
done

echo "=========================================="
echo "Final Results"
echo "=========================================="
echo "Total datasets tested: $TOTAL"
echo "Successful: $SUCCESS"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ ALL 29 DATASETS WORKING!"
    
    # Show sample instructions from different categories
    echo ""
    echo "Sample dynamic instructions generated:"
    echo "=========================================="
    echo ""
    echo "1. Classification (CFA):"
    echo "--------------------"
    head -15 "$OUTPUT_DIR"/pixiu-cfa-*/instruction.md 2>/dev/null || echo "  (not found)"
    echo ""
    echo "2. Regression (TSA):"
    echo "--------------------"
    head -15 "$OUTPUT_DIR"/pixiu-tsa-*/instruction.md 2>/dev/null || echo "  (not found)"
    echo ""
    echo "3. Financial QA (FinQA):"
    echo "--------------------"
    head -15 "$OUTPUT_DIR"/pixiu-finqa-*/instruction.md 2>/dev/null || echo "  (not found)"
    echo ""
    echo "4. NER (flare-ner):"
    echo "--------------------"
    head -15 "$OUTPUT_DIR"/pixiu-ner-*/instruction.md 2>/dev/null || echo "  (not found)"
    echo ""
    echo "5. Extractive Summarization (ECTSUM):"
    echo "--------------------"
    head -15 "$OUTPUT_DIR"/pixiu-ectsum-*/instruction.md 2>/dev/null || echo "  (not found)"
    
    exit 0
else
    echo "❌ SOME TESTS FAILED"
    exit 1
fi
