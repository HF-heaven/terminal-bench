#!/bin/bash

# Test all 29 PIXIU datasets with their correct splits
# Based on the user's analysis of available data

cd /home/wendy/terminal-bench/adapters/pixiu
rm -rf /tmp/pixiu_all_29_correct_splits
mkdir -p /tmp/pixiu_all_29_correct_splits

echo "Testing all 29 PIXIU datasets with correct splits..."
echo ""

passed=0
failed=0

# Array of datasets with their correct splits
# Format: "dataset_name|split"
declare -a datasets=(
  "TheFinAI/flare-australian|test"
  "TheFinAI/flare-causal20-sc|test"
  "daishen/cra-ccf|validation"
  "daishen/cra-ccfraud|train"
  "TheFinAI/flare-cd|test"
  "TheFinAI/flare-cfa|test"
  "daishen/cra-taiwan|validation"
  "TheFinAI/flare-ectsum|test"
  "TheFinAI/flare-edtsum|test"
  "TheFinAI/finben-finer-ord|test"
  "TheFinAI/flare-finqa|test"
  "TheFinAI/flare-finred|test"
  "TheFinAI/flare-fiqasa|test"
  "TheFinAI/flare-fnxl|test"
  "TheFinAI/finben-fomc|test"
  "TheFinAI/en-fpb|test"
  "TheFinAI/flare-fsrl|test"
  "TheFinAI/flare-german|test"
  "TheFinAI/flare-headlines|test"
  "TheFinAI/flare-ma|test"
  "TheFinAI/flare-mlesg|test"
  "TheFinAI/flare-multifin-en|test"
  "TheFinAI/flare-ner|test"
  "TheFinAI/flare-sm-acl|test"
  "TheFinAI/flare-sm-bigdata|test"
  "TheFinAI/flare-sm-cikm|test"
  "TheFinAI/en-forecasting-travelinsurance|validation"
  "TheFinAI/flare-tatqa|test"
  "TheFinAI/flare-tsa|test"
)

for entry in "${datasets[@]}"; do
  IFS='|' read -r dataset split <<< "$entry"
  echo "Testing: $dataset (split: $split)"
  
  if ~/.local/bin/uv run python run_adapter.py --harbor --dataset-name "$dataset" --split "$split" --limit 1 --output-path /tmp/pixiu_all_29_correct_splits 2>&1 | grep -q "Generated.*PIXIU Harbor tasks"; then
    echo "  ✅ PASS"
    ((passed++))
  else
    echo "  ❌ FAIL"
    ((failed++))
  fi
done

echo ""
echo "========================================"
echo "Summary: $passed passed, $failed failed out of 29 datasets"
echo "========================================"

if [ $failed -eq 0 ]; then
  echo "✅ All 29 datasets work correctly!"
else
  echo "⚠️  Some datasets failed"
fi
