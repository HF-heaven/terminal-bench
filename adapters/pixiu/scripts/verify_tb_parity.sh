#!/bin/bash
# PIXIU - Terminal-Bench Parity Verification
# Generates same tasks in TB format and verifies oracle pass rate

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

TB_OUTPUT="/tmp/pixiu_tb_parity_test"
LIMIT=5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  PIXIU - Terminal-Bench Parity Validation ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# Clean previous TB test
echo "Cleaning previous Terminal-Bench test data..."
rm -rf "$TB_OUTPUT"
mkdir -p "$TB_OUTPUT"

# Verify HF token
if [ -z "${HF_TOKEN:-}" ]; then
    echo -e "${RED}✗ ERROR: HF_TOKEN not set${NC}"
    exit 1
fi

# Generate Terminal-Bench format tasks
echo "Generating Terminal-Bench format tasks..."
echo ""

datasets=(
    "TheFinAI/flare-australian:test"
    "TheFinAI/flare-causal20-sc:test"
    "TheFinAI/flare-cfa:test"
    "TheFinAI/flare-ectsum:test"
    "TheFinAI/flare-edtsum:test"
    "TheFinAI/flare-finqa:test"
    "TheFinAI/flare-finred:test"
    "TheFinAI/flare-fomc:test"
    "TheFinAI/en-fpb:test"
    "TheFinAI/flare-german:test"
    "TheFinAI/flare-headlines:test"
    "TheFinAI/flare-ner:test"
    "TheFinAI/flare-tatqa:test"
    "TheFinAI/flare-tsa:test"
    "TheFinAI/flare-fiqasa:test"
    "TheFinAI/flare-fnxl:test"
    "TheFinAI/flare-fsrl:test"
    "TheFinAI/en-forecasting-travelinsurance:test"
)

success=0
for dataset_split in "${datasets[@]}"; do
    dataset=$(echo "$dataset_split" | cut -d: -f1)
    split=$(echo "$dataset_split" | cut -d: -f2)
    dataset_name=$(basename "$dataset")
    
    echo -n "  $dataset_name... "
    
    # Generate WITHOUT --harbor flag (Terminal-Bench format)
    if HF_TOKEN="$HF_TOKEN" ~/.local/bin/uv run python "$ADAPTER_DIR/run_adapter.py" \
        --dataset-name "$dataset" \
        --split "$split" \
        --limit "$LIMIT" \
        --output-path "$TB_OUTPUT" \
        > /tmp/pixiu_tb_gen_${dataset_name}.log 2>&1; then
        echo -e "${GREEN}✓${NC}"
        ((success++))
    else
        echo -e "${RED}✗${NC}"
    fi
done

total_tasks=$(find "$TB_OUTPUT" -mindepth 1 -maxdepth 1 -type d -name "pixiu-*" | wc -l)
echo ""
echo "Generated $total_tasks Terminal-Bench tasks"
echo ""

# Run Terminal-Bench oracle test
echo "Running Terminal-Bench oracle tests..."
cd "$ADAPTER_DIR/../.."

~/.local/bin/uv run tb run \
    --dataset-path "$TB_OUTPUT" \
    --agent oracle \
    --task-id "pixiu-*" \
    --output-path "/tmp/pixiu_tb_parity_results" \
    2>&1 | tee /tmp/pixiu_tb_parity.log

# Extract results
echo ""
echo "════════════════════════════════════════"
echo "Terminal-Bench Parity Results:"
echo "════════════════════════════════════════"

if grep -q "Resolved:" /tmp/pixiu_tb_parity.log; then
    resolved=$(grep "Resolved:" /tmp/pixiu_tb_parity.log | tail -1 | grep -oE "[0-9]+/[0-9]+" | head -1)
    echo -e "Oracle pass rate: ${GREEN}$resolved${NC}"
    
    # Check if 100%
    pass=$(echo "$resolved" | cut -d/ -f1)
    total=$(echo "$resolved" | cut -d/ -f2)
    
    if [ "$pass" -eq "$total" ]; then
        echo -e "${GREEN}✓ 100% parity achieved!${NC}"
    else
        echo -e "${YELLOW}⚠ Not all tasks passed${NC}"
        echo "Check logs: /tmp/pixiu_tb_parity.log"
    fi
else
    echo -e "${RED}✗ Could not extract results${NC}"
    echo "Check logs: /tmp/pixiu_tb_parity.log"
fi

echo ""
echo "Full log: /tmp/pixiu_tb_parity.log"
echo "Results: /tmp/pixiu_tb_parity_results"
