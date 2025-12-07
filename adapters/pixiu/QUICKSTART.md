# PIXIU Harbor Migration - Quick Start Guide

## Overview

This guide shows you how to generate PIXIU tasks in Harbor format and validate them with oracle spot-checks.

## Prerequisites

```bash
cd /home/wendy/terminal-bench/adapters/pixiu
```

## Step 1: Generate Harbor Tasks

### Option A: Generate All Task Types (Recommended)

Generate 100 tasks per dataset type (29 datasets total):

```bash
./generate_all_harbor_tasks.sh
```

This will:
- Generate tasks in `datasets/pixiu/` by default
- Process all 29 PIXIU datasets
- Validate the structure automatically
- Show progress for each dataset

**Custom output directory:**
```bash
./generate_all_harbor_tasks.sh /path/to/custom/output
```

### Option B: Generate Specific Task Types

For testing or development, generate specific datasets:

```bash
# Classification tasks (CFA)
python run_adapter.py --harbor --dataset-name "TheFinAI/flare-cfa" --limit 100

# Regression tasks (TSA)  
python run_adapter.py --harbor --dataset-name "TheFinAI/flare-tsa" --limit 100

# Numerical reasoning (FinQA)
python run_adapter.py --harbor --dataset-name "TheFinAI/flare-finqa" --limit 100

# Token classification (FNXL)
python run_adapter.py --harbor --dataset-name "TheFinAI/flare-fnxl" --limit 100

# Summarization (ECTSUM)
python run_adapter.py --harbor --dataset-name "TheFinAI/flare-ectsum" --limit 100
```

## Step 2: Validate Structure

Check that all Harbor format requirements are met:

```bash
./validate_harbor_format.sh /home/wendy/terminal-bench/datasets/pixiu
```

This verifies:
- ✅ Required files exist (task.toml, instruction.md, environment/Dockerfile, etc.)
- ✅ Old Terminal-Bench files removed
- ✅ Reward writing in test.sh
- ✅ Proper TOML structure

## Step 3: Spot-Check Oracle

Test 5 random tasks from each task type to verify oracle solutions:

```bash
./spot_check_oracle.sh /home/wendy/terminal-bench/datasets/pixiu
```

This will test tasks from:
- Classification (CFA)
- Regression (TSA)
- Numerical reasoning (FinQA)
- Table QA (TAT-QA)
- Token classification (FNXL)
- Semantic role labeling (FSRL)
- Extractive summarization (ECTSUM)
- Abstractive summarization (EDTSUM)
- NER (flare-ner)
- Sequence labeling (CD)
- Relation extraction (FinRED)

## Step 4: Test with Harbor Harness (When Available)

### Test Single Task

```bash
cd /home/wendy/terminal-bench
uv run harbor trials start -p datasets/pixiu/pixiu-cfa-cfa0 -a oracle
```

### Test Specific Task Type

```bash
# Test 5 CFA tasks
for i in {0..4}; do
  uv run harbor trials start -p datasets/pixiu/pixiu-cfa-cfa$i -a oracle
done
```

### Test All Tasks

```bash
uv run harbor jobs start -p datasets/pixiu -a oracle
```

## Complete All 29 Dataset Types

The adapter supports all 29 PIXIU datasets:

**Classification (18):**
1. flare-headlines - Financial news headlines
2. en-fpb - Financial PhraseBank sentiment
3. flare-causal20-sc - Causal relationships
4. flare-fiqasa - FiQA sentiment analysis
5. finben-fomc - FOMC hawkish/dovish
6. flare-cfa - CFA exam questions
7. flare-german - German credit scoring
8. cra-ccfraud - Credit card fraud
9. flare-australian - Australian credit
10. cra-ccf - Credit card fraud (binary)
11. cra-taiwan - Taiwan bankruptcy
12. en-forecasting-travelinsurance - Travel insurance
13. flare-mlesg - ESG multi-label
14. flare-ma - M&A deals
15. flare-multifin-en - Multi-class headlines
16. flare-sm-acl - Stock movement (ACL)
17. flare-sm-bigdata - Stock movement (BigData)
18. flare-sm-cikm - Stock movement (CIKM)

**QA & Reasoning (2):**
19. flare-finqa - Financial numerical reasoning
20. flare-tatqa - Table-based QA

**Regression (1):**
21. flare-tsa - Targeted sentiment analysis

**Token-Level (6):**
22. flare-ner - Named entity recognition
23. flare-fnxl - Financial token classification
24. flare-fsrl - Semantic role labeling
25. finben-finer-ord - Financial NER
26. flare-cd - Causal detection

**Relation Extraction (1):**
27. flare-finred - Financial relation extraction

**Summarization (2):**
28. flare-ectsum - Extractive summarization
29. flare-edtsum - Abstractive summarization

## Example Workflows

### Quick Test (3 datasets, 10 tasks each)

```bash
# Generate small test set
python run_adapter.py --harbor --dataset-name "TheFinAI/flare-cfa" --limit 10
python run_adapter.py --harbor --dataset-name "TheFinAI/flare-tsa" --limit 10
python run_adapter.py --harbor --dataset-name "TheFinAI/flare-finqa" --limit 10

# Validate
./validate_harbor_format.sh datasets/pixiu

# Spot-check
./spot_check_oracle.sh datasets/pixiu
```

### Full Production Run (All 29 datasets)

```bash
# Generate all tasks (takes ~10-15 minutes)
./generate_all_harbor_tasks.sh

# This will automatically:
# 1. Generate 100 tasks per dataset (2900 total tasks)
# 2. Validate structure
# 3. Report any issues
```

### Custom Limits per Dataset

Edit `generate_all_harbor_tasks.sh` and change the limits:

```bash
declare -A DATASETS=(
    ["cfa"]="TheFinAI/flare-cfa:50"      # Change from 100 to 50
    ["tsa"]="TheFinAI/flare-tsa:200"     # Change from 100 to 200
    # ... etc
)
```

## Troubleshooting

### Issue: "Dataset not found" or "Access denied"

Some datasets are gated. You need to:
1. Visit the dataset page on HuggingFace
2. Request access
3. Login: `huggingface-cli login`

### Issue: "Template not found"

Make sure you're in the correct directory:
```bash
cd /home/wendy/terminal-bench/adapters/pixiu
```

### Issue: Validation fails

Check the specific error message. Common issues:
- Missing files: Re-run generation
- Old files exist: Clean up old Terminal-Bench tasks first

## Files Generated

For each task, Harbor format creates:
```
pixiu-cfa-cfa0/
├── task.toml                    # Metadata (TOML format)
├── instruction.md               # Task instructions
├── environment/
│   └── Dockerfile              # Container definition
├── solution/
│   └── solve.sh                # Oracle solution
└── tests/
    ├── test.sh                 # Test wrapper with reward
    ├── test_outputs.py         # Pytest tests
    └── data/
        └── item.json           # Task data
```

## Next Steps After Generation

1. ✅ Generate all tasks: `./generate_all_harbor_tasks.sh`
2. ✅ Validate structure: `./validate_harbor_format.sh datasets/pixiu`
3. ✅ Spot-check oracle: `./spot_check_oracle.sh datasets/pixiu`
4. 🚀 Test with Harbor: `uv run harbor jobs start -p datasets/pixiu -a oracle`
5. 📦 Submit to harbor-datasets repository
6. 📝 Update Harbor registry.json
7. 🧪 Run parity experiments

## Support

For issues or questions:
- Check `MIGRATION.md` for detailed migration documentation
- Review task-specific templates in `template_harbor_*/`
- Run validation scripts for debugging
