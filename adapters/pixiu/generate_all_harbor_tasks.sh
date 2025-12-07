#!/bin/bash
# Generate Harbor format tasks for all PIXIU dataset types
# Usage: ./generate_all_harbor_tasks.sh [output_directory]

set -euo pipefail

OUTPUT_DIR="${1:-/home/wendy/terminal-bench/datasets/pixiu}"
ADAPTER_DIR="/home/wendy/terminal-bench/adapters/pixiu"

echo "=================================================="
echo "Generating PIXIU Harbor Tasks"
echo "Output directory: $OUTPUT_DIR"
echo "=================================================="
echo ""

# All 29 PIXIU datasets organized by category
declare -A DATASETS=(
    # Classification tasks (18 datasets)
    ["headlines"]="TheFinAI/flare-headlines:100"
    ["fpb"]="TheFinAI/en-fpb:100"
    ["causal20"]="TheFinAI/flare-causal20-sc:100"
    ["fiqasa"]="TheFinAI/flare-fiqasa:100"
    ["fomc"]="TheFinAI/finben-fomc:100"
    ["cfa"]="TheFinAI/flare-cfa:100"
    ["german"]="TheFinAI/flare-german:100"
    ["ccfraud"]="daishen/cra-ccfraud:100"
    ["australian"]="TheFinAI/flare-australian:100"
    ["ccf"]="daishen/cra-ccf:100"
    ["taiwan"]="daishen/cra-taiwan:100"
    ["travel"]="TheFinAI/en-forecasting-travelinsurance:100"
    ["mlesg"]="TheFinAI/flare-mlesg:100"
    ["ma"]="TheFinAI/flare-ma:100"
    ["multifin"]="TheFinAI/flare-multifin-en:100"
    ["smacl"]="TheFinAI/flare-sm-acl:100"
    ["smbigdata"]="TheFinAI/flare-sm-bigdata:100"
    ["smcikm"]="TheFinAI/flare-sm-cikm:100"
    
    # QA tasks (2 datasets)
    ["finqa"]="TheFinAI/flare-finqa:100"
    ["tatqa"]="TheFinAI/flare-tatqa:100"
    
    # Regression (1 dataset)
    ["tsa"]="TheFinAI/flare-tsa:100"
    
    # Token-level tasks (6 datasets)
    ["ner"]="TheFinAI/flare-ner:100"
    ["fnxl"]="TheFinAI/flare-fnxl:100"
    ["fsrl"]="TheFinAI/flare-fsrl:100"
    ["finerord"]="TheFinAI/finben-finer-ord:100"
    ["cd"]="TheFinAI/flare-cd:100"
    
    # Relation extraction (1 dataset)
    ["finred"]="TheFinAI/flare-finred:100"
    
    # Summarization (2 datasets)
    ["ectsum"]="TheFinAI/flare-ectsum:100"
    ["edtsum"]="TheFinAI/flare-edtsum:100"
)

TOTAL=0
SUCCESS=0
FAILED=0

cd "$ADAPTER_DIR"

for task_name in "${!DATASETS[@]}"; do
    IFS=':' read -r dataset limit <<< "${DATASETS[$task_name]}"
    
    echo "[$((TOTAL+1))/${#DATASETS[@]}] Generating: $task_name"
    echo "  Dataset: $dataset"
    echo "  Limit: $limit tasks"
    
    if python run_adapter.py --harbor \
        --dataset-name "$dataset" \
        --limit "$limit" \
        --output-path "$OUTPUT_DIR" 2>&1 | tail -1; then
        ((SUCCESS++))
    else
        echo "  ❌ Failed to generate $task_name"
        ((FAILED++))
    fi
    
    ((TOTAL++))
    echo ""
done

echo "=================================================="
echo "Generation Summary"
echo "=================================================="
echo "Total datasets: $TOTAL"
echo "Successfully generated: $SUCCESS"
echo "Failed: $FAILED"
echo ""
echo "Tasks generated in: $OUTPUT_DIR"
echo ""

# Validate structure
echo "Validating Harbor format structure..."
if "$ADAPTER_DIR/validate_harbor_format.sh" "$OUTPUT_DIR"; then
    echo ""
    echo "✅ All tasks validated successfully!"
    echo ""
    echo "Next: Run oracle spot-check with:"
    echo "  ./spot_check_oracle.sh $OUTPUT_DIR"
else
    echo ""
    echo "⚠️  Some validation issues found - review above"
fi
