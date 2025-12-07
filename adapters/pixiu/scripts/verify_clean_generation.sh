#!/bin/bash
# PIXIU Harbor Adapter - Clean Generation Verification Script
# This script ensures completely fresh generation without cached results

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="/tmp/pixiu_harbor_clean_test"
LIMIT=5  # Tasks per dataset
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADAPTER_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  PIXIU Harbor Adapter - Clean Generation Verification ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Clean Environment
echo -e "${YELLOW}[Step 1/5]${NC} Cleaning environment..."
echo "  → Removing previous task directories..."
rm -rf /tmp/pixiu_* /tmp/test_pixiu_* "$OUTPUT_DIR"

echo "  → Removing PIXIU Docker images..."
docker images | grep -E "(pixiu|test-pixiu)" | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true

echo "  → Clearing Python cache..."
cd "$ADAPTER_DIR"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo -e "${GREEN}✓ Environment cleaned${NC}"
echo ""

# Step 2: Verify HF Token
echo -e "${YELLOW}[Step 2/5]${NC} Verifying HuggingFace token..."
if [ -z "${HF_TOKEN:-}" ]; then
    echo -e "${RED}✗ ERROR: HF_TOKEN environment variable not set${NC}"
    echo "  Please set it: export HF_TOKEN='your_token_here'"
    exit 1
fi
echo -e "${GREEN}✓ HF_TOKEN is set${NC}"
echo ""

# Step 3: Generate Tasks
echo -e "${YELLOW}[Step 3/5]${NC} Generating PIXIU Harbor tasks..."
echo "  Output directory: $OUTPUT_DIR"
echo "  Samples per dataset: $LIMIT"
echo ""

mkdir -p "$OUTPUT_DIR"

# List of working datasets
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

success_count=0
fail_count=0
failed_datasets=()

for dataset_split in "${datasets[@]}"; do
    dataset=$(echo "$dataset_split" | cut -d: -f1)
    split=$(echo "$dataset_split" | cut -d: -f2)
    dataset_name=$(basename "$dataset")
    
    echo -n "  Generating $dataset_name... "
    
    if HF_TOKEN="$HF_TOKEN" ~/.local/bin/uv run python "$ADAPTER_DIR/run_adapter.py" \
        --dataset-name "$dataset" \
        --split "$split" \
        --limit "$LIMIT" \
        --output-path "$OUTPUT_DIR" \
        --harbor \
        > /tmp/pixiu_gen_${dataset_name}.log 2>&1; then
        
        echo -e "${GREEN}✓${NC}"
        ((success_count++))
    else
        echo -e "${RED}✗${NC}"
        ((fail_count++))
        failed_datasets+=("$dataset_name")
    fi
done

echo ""
echo -e "Generation complete: ${GREEN}${success_count} succeeded${NC}, ${RED}${fail_count} failed${NC}"

if [ $fail_count -gt 0 ]; then
    echo -e "${RED}Failed datasets:${NC}"
    for ds in "${failed_datasets[@]}"; do
        echo "  - $ds"
        echo "    Log: /tmp/pixiu_gen_${ds}.log"
    done
    echo ""
fi

# Step 4: Validate Structure
echo -e "${YELLOW}[Step 4/5]${NC} Validating task structure..."

total_tasks=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name "pixiu-*" | wc -l)
echo "  Total tasks generated: $total_tasks"
echo "  Expected: $((success_count * LIMIT))"

if [ "$total_tasks" -ne "$((success_count * LIMIT))" ]; then
    echo -e "${RED}✗ Task count mismatch!${NC}"
    exit 1
fi

# Check structure of first task
sample_task=$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name "pixiu-*" | head -1)
echo "  Sample task: $(basename "$sample_task")"

required_files=(
    "task.toml"
    "instruction.md"
    "Dockerfile"
    "environment/Dockerfile"
    "solution/solve.sh"
    "tests/test_outputs.py"
)

all_files_exist=true
for file in "${required_files[@]}"; do
    if [ -f "$sample_task/$file" ]; then
        echo -e "    ${GREEN}✓${NC} $file"
    else
        echo -e "    ${RED}✗${NC} $file (missing)"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    echo -e "${RED}✗ Structure validation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Structure validation passed${NC}"
echo ""

# Step 5: Summary
echo -e "${YELLOW}[Step 5/5]${NC} Summary"
echo "════════════════════════════════════════"
echo -e "Tasks generated:     ${GREEN}$total_tasks${NC}"
echo -e "Datasets processed:  ${GREEN}$success_count${NC} / ${BLUE}${#datasets[@]}${NC}"
echo -e "Output location:     ${BLUE}$OUTPUT_DIR${NC}"
echo ""
echo -e "${GREEN}✓ Clean generation verification complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run oracle tests: ./scripts/verify_harbor_oracle.sh"
echo "  2. Validate parity: ./scripts/verify_tb_parity.sh"
echo "  3. Review: cat VERIFICATION_PLAN.md"
echo ""
