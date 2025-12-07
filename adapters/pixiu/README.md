# PIXIU → Harbor Adapter

## Overview

This adapter converts [PIXIU FinBen](https://github.com/The-FinAI/PIXIU) benchmark tasks into **Harbor-compatible tasks**, enabling evaluation of reasoning agents on financial NLP tasks within isolated, reproducible environments.

PIXIU evaluates a model's ability to understand and analyze financial data across multiple tasks including sentiment analysis, classification, question answering, named entity recognition, relation extraction, and text summarization.

- **Benchmark type:** Financial NLP and analysis
- **Languages:** English
- **Dataset size:** 18 accessible datasets from PIXIU FinBen
- **Source:** [PIXIU GitHub](https://github.com/The-FinAI/PIXIU) & [HuggingFace](https://huggingface.co/TheFinAI)
- **Licensing:** Mixed (see dataset-specific licenses)
- **Adapter scope:** All publicly accessible PIXIU datasets from TheFinAI organization

This Harbor adapter provides a standardized interface for PIXIU evaluation, with support for both Terminal-Bench and Harbor formats.

---

## What is PIXIU?

PIXIU (Platform for fInancial eXplainable Intelligence Understanding) is a comprehensive benchmark designed to evaluate LLMs on financial tasks. The FinBen 2.0 benchmark includes:

- **Sentiment Analysis:** FPB, FiQA-SA, TSA, FOMC
- **Classification:** Headlines, CFA, German credit scoring, Australian credit scoring
- **Knowledge Extraction:** NER, FinRED, Causal20-SC
- **Number Understanding:** FinQA, TatQA, FNXL, FSRL
- **Text Summarization:** ECTSUM, EDTSUM
- **Forecasting:** Travel Insurance prediction

**Metrics:** Task-specific (Accuracy, F1, ROUGE, Exact Match, etc.)

---

## Adapter Features

- ✅ Dynamic instruction generation (11 task-specific templates)
- ✅ Template consolidation (2 templates vs. original 11)
- ✅ Support for both Terminal-Bench (.yaml) and Harbor (.toml) formats
- ✅ Automatic handling of multiline answers (ECTSUM, EDTSUM)
- ✅ Shell variable escaping for financial amounts ($1,496.5, etc.)
- ✅ Format-specific tests (comma-separated NER, semicolon-separated FinRED)
- ✅ Reproducible Docker environments
- ✅ Complete parity validation with Terminal-Bench format

---

## Available Datasets

### Working Datasets (18 total)

| Dataset | Task Type | Samples | Split |
|---------|-----------|---------|-------|
| `flare-australian` | Credit Scoring | 690 | test |
| `flare-causal20-sc` | Causal Classification | 8,630 | test |
| `flare-cfa` | Classification (CFA Exam) | varies | test |
| `flare-ectsum` | Extractive Summarization | 495 | test |
| `flare-edtsum` | Abstractive Summarization | 2,000 | test |
| `flare-finqa` | Question Answering | 8,281 | test |
| `flare-finred` | Relation Extraction | 1,070 | test |
| `flare-fomc` | Sentiment Analysis | 496 | test |
| `en-fpb` | Sentiment Analysis | 4,845 | test |
| `flare-german` | Credit Scoring | 1,000 | test |
| `flare-headlines` | Classification | 11,412 | test |
| `flare-ner` | Named Entity Recognition | 1,366 | test |
| `flare-tatqa` | Question Answering | 1,670 | test |
| `flare-tsa` | Sentiment Analysis | 561 | test |
| `flare-fiqasa` | Sentiment Analysis | 1,173 | test |
| `flare-fnxl` | Numeric Labeling | 318 | test |
| `flare-fsrl` | Token Classification | 97 | test |
| `en-forecasting-travelinsurance` | Forecasting | 12,665 | test |

### Known Unavailable Datasets

The following datasets are referenced in PIXIU documentation but are not currently accessible:

- `flare-ccf`, `flare-ccfraud` - Not found on HuggingFace Hub
- `flare-convfinqa` - Data format issues (missing 'text' field)
- `flare-fpb` - Gated dataset (requires access request)
- Several others missing from official documentation

---

## Generated Task Structure

```
datasets/pixiu/
├── pixiu-{dataset}-{id}/
│   ├── task.toml              # Harbor configuration
│   ├── instruction.md         # Task-specific instruction
│   ├── Dockerfile             # Main container
│   ├── environment/
│   │   └── Dockerfile         # Base Python 3.11 environment
│   ├── solution/
│   │   └── solve.sh           # Oracle solution
│   └── tests/
│       ├── test_outputs.py    # pytest validation
│       └── data/
│           └── item.json      # Test data
```

---

## Usage: Generate Tasks

### Harbor Format (TOML-based)

```bash
cd /home/wendy/terminal-bench/adapters/pixiu
export HF_TOKEN="your_huggingface_token"

# Generate all 18 datasets (5 samples each = 90 tasks)
./scripts/verify_clean_generation.sh

# Or generate specific dataset
uv run python run_adapter.py \
  --dataset-name "TheFinAI/flare-cfa" \
  --split test \
  --limit 10 \
  --output-path ../../datasets/pixiu \
  --harbor
```

### Terminal-Bench Format (YAML-based)

```bash
# Generate without --harbor flag
uv run python run_adapter.py \
  --dataset-name "TheFinAI/flare-cfa" \
  --split test \
  --limit 10 \
  --output-path ../../tasks
```

---

## Run Evaluation

### Using Terminal-Bench Harness (YAML format)

```bash
cd /home/wendy/terminal-bench

# Oracle test
uv run tb run \
  --dataset-path tasks \
  --agent oracle \
  --task-id "pixiu-cfa-*" \
  --output-path runs/pixiu-oracle-test

# With agent
uv run tb run \
  --dataset-path tasks \
  --agent terminus-2 \
  --task-id "pixiu-*" \
  --output-path runs/pixiu-agent-test
```

### Using Harbor (TOML format)

```bash
# Oracle test
uv run harbor trials start -p datasets/pixiu/pixiu-cfa-cfa0 -a oracle

# With agent  
uv run harbor trials start -p datasets/pixiu/pixiu-cfa-cfa0 -a <agent> -m "<model>"

# Batch evaluation
uv run harbor jobs start -p datasets/pixiu -a oracle
```

---

## Verification & Testing

### Complete Clean Verification

```bash
cd /home/wendy/terminal-bench/adapters/pixiu

# Step 1: Clean generation (removes all cached data)
export HF_TOKEN="your_token"
./scripts/verify_clean_generation.sh

# Step 2: Harbor oracle test
./scripts/verify_harbor_oracle.sh

# Step 3: Terminal-Bench parity test
./scripts/verify_tb_parity.sh
```

### Expected Results

- **Generation:** 90 tasks (18 datasets × 5 samples)
- **Harbor Oracle:** 100% pass rate
- **TB Parity:** 100% pass rate

See [VERIFICATION_PLAN.md](VERIFICATION_PLAN.md) for complete verification process.

---

## Comparison with Terminal-Bench (Parity)

| Format | Agent | Tasks Tested | Pass Rate | Notes |
|--------|-------|--------------|-----------|-------|
| Terminal-Bench (YAML) | `oracle` | 90 | **100%** | Previous verification |
| Harbor (TOML) | `oracle` | 90 | **100%** | Current implementation |

### Key Implementation Details

1. **Template Consolidation:** Reduced from 11 specialized templates to 2 base templates with dynamic instruction generation
2. **Shell Escaping:** Properly handles financial amounts like `$1,496.5` and `C$110`
3. **Format Handling:** 
   - NER uses comma-separated format
   - FinRED uses semicolon-separated format
   - ECTSUM/EDTSUM support multiline answers
4. **Split Configuration:** Correct test/validation split for each dataset

---

## Known Issues

### Dataset Access Issues

The following datasets require special access or have technical issues:

| Dataset | Issue | Status |
|---------|-------|--------|
| `flare-fpb` | Gated dataset | Request access on HuggingFace |
| `flare-ccf`, `flare-ccfraud` | Not found on Hub | May be deprecated |
| `flare-convfinqa` | Data format error | Under investigation |

These are **dataset issues**, not adapter bugs. The adapter works correctly for all accessible datasets.

---

## Installation / Prerequisites

- **Docker:** Required for containerized execution
- **Python:** ≥ 3.11
- **UV:** Package manager
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Dependencies:**
  ```bash
  cd /home/wendy/terminal-bench
  uv sync --extra dev
  ```
- **HuggingFace Token:** Required for dataset access
  ```bash
  export HF_TOKEN="your_token_here"
  ```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: task.yaml` | Using TB harness on Harbor tasks | Harbor tasks use `task.toml`, not `task.yaml`. Use Harbor CLI or generate without `--harbor` flag |
| `HF_TOKEN not set` | Missing authentication | Export HF token: `export HF_TOKEN="..."` |
| Generation fails silently | Network/timeout issues | Check logs in `/tmp/pixiu_gen_*.log` |
| Docker build errors | Missing base image | Pre-pull: `docker pull python:3.11-slim` |
| Shell variable errors | Unescaped $ in answers | Adapter now escapes these automatically |
| Test format mismatches | Wrong separator expected | Adapter uses correct format per task type |

---

## Architecture

### Template Structure

**Before:** 11 specialized templates (one per task type)
```
template_harbor/
template_harbor_ner/
template_harbor_finqa/
template_harbor_tatqa/
... (8 more)
```

**After:** 2 base templates + dynamic instructions
```
template_harbor/          # Standard tasks
template_harbor_regression/  # TSA regression
+ _create_instruction_content()  # 11 variants
```

### Dynamic Instruction Generation

The adapter detects task type from dataset name and generates appropriate instructions:

```python
def _create_instruction_content(record):
    if 'ner' in dataset_name.lower():
        return "NER-specific instruction..."
    elif 'finqa' in dataset_name.lower():
        return "FinQA-specific instruction..."
    # ... 11 variants total
```

---

## Citation

```bibtex
@article{xie2023pixiu,
  title={PIXIU: A Large Language Model, Instruction Data and Evaluation Benchmark for Finance},
  author={Xie, Qianqian and Han, Weiguang and others},
  journal={arXiv preprint arXiv:2306.05443},
  year={2023}
}

@article{xie2024finben,
  title={The FinBen: An Holistic Financial Benchmark for Large Language Models},
  author={Xie, Qianqian and Han, Weiguang and Chen, Zhengyu and others},
  journal={arXiv preprint arXiv:2402.12659},
  year={2024}
}
```

---

## Authors & Contributions

Adapter maintained by the **Terminal-Bench Development Team**.

Key contributions:
- Template consolidation and dynamic instruction generation
- Shell escaping for financial amounts
- Format-specific test generation
- Harbor format migration

For feedback or issues, please open a pull request or issue on the main repository.

---

## Development Status

- ✅ **Core Adapter:** Complete
- ✅ **Harbor Migration:** Complete  
- ✅ **Terminal-Bench Parity:** Verified
- ✅ **Documentation:** Complete
- ⏳ **Full-scale Evaluation:** Pending (~1,800 tasks @ 100 samples/dataset)
- ⏳ **Harbor Registry Integration:** Pending

Last updated: December 7, 2025
