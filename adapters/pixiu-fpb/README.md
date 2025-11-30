# PIXIU Financial PhraseBank (FPB) Adapter for Terminal-Bench

This adapter converts the [PIXIU Financial PhraseBank (en-fpb)](https://huggingface.co/datasets/TheFinAI/en-fpb) dataset into Terminal-Bench format.

## Overview

**Original Benchmark**: [PIXIU Financial PhraseBank](https://huggingface.co/datasets/TheFinAI/en-fpb)  
**Task Type**: Financial Sentiment Analysis  
**Dataset**: TheFinAI/en-fpb (gated dataset)

The Financial PhraseBank contains financial news sentences labeled with sentiment:
- `negative`: Negative sentiment
- `neutral`: Neutral sentiment  
- `positive`: Positive sentiment

## Prerequisites

⚠️ **Important**: This is a **gated dataset**. Before using this adapter, you must:

1. Visit https://huggingface.co/datasets/TheFinAI/en-fpb
2. Click "Request access" and wait for approval
3. Login to HuggingFace CLI:
   ```bash
   huggingface-cli login
   ```

## Usage

### Generate Tasks

```bash
cd /home/hefan/terminal-bench/adapters/pixiu-fpb

# Generate 100 tasks (default)
uv run python run_adapter.py

# Generate 200 tasks
uv run python run_adapter.py --limit 200

# Generate all tasks from the test split
uv run python run_adapter.py --limit -1

# Specify custom output directory
uv run python run_adapter.py --output-dir /path/to/output
```

### Run with Terminal-Bench

#### Option 1: Run locally with dataset path

```bash
cd /home/hefan/terminal-bench

# Test with oracle agent (10 tasks)
uv run tb run \
  --agent oracle \
  --dataset-path dataset/pixiu-en-fpb \
  --n-tasks 10 \
  --n-concurrent 4

# Run all tasks
uv run tb run \
  --agent oracle \
  --dataset-path dataset/pixiu-en-fpb \
  --n-concurrent 8
```

#### Option 2: Run from registry (after dataset is published)

```bash
uv run tb run \
  --dataset pixiu-en-fpb \
  --agent claude-code \
  --model anthropic/claude-opus-4-20250514
```

## Adapter Structure

```
pixiu-fpb/
├── adapter.py              # Core adapter logic
├── run_adapter.py          # Entry point for task generation
├── pyproject.toml          # Python dependencies
├── README.md               # This file
├── parity_experiment.json  # Parity results (to be added)
└── template/               # Task templates
    ├── task.yaml
    ├── Dockerfile
    ├── docker-compose.yaml
    ├── solution.sh
    ├── run-tests.sh
    └── tests/
        └── test_outputs.py
```

## Task Format

Each generated task contains:

1. **task.yaml**: Task metadata and instruction
2. **tests/data/item.json**: Original FPB sample with sentence and choices
3. **solution.sh**: Oracle solution (correct sentiment label)
4. **tests/test_outputs.py**: Pytest verification script
5. **Dockerfile**: Container environment (Ubuntu 24.04 + Python + pytest)
6. **docker-compose.yaml**: Container orchestration
7. **run-tests.sh**: Test execution script

### Example item.json

```json
{
  "id": "fpb000042",
  "sentence": "The company's revenue increased by 15% in Q3.",
  "choices": ["negative", "neutral", "positive"],
  "dataset": "TheFinAI/en-fpb",
  "split": "test"
}
```

## Adaptation Details

### Key Differences from Original Benchmark

1. **Environment**: Tasks run in isolated Docker containers (Ubuntu 24.04)
2. **Evaluation**: Uses pytest for automated verification
3. **Format**: Each sample becomes a standalone Terminal-Bench task
4. **Label Mapping**: Numeric labels (0, 1, 2) → String labels ("negative", "neutral", "positive")

### Label Mapping

```python
label_map = {
    0: "negative",
    1: "neutral", 
    2: "positive"
}
```

## Parity Experiments

*To be completed after dataset access is granted and experiments are run.*

### Baseline Results (Original Benchmark)

| Agent | Model | Resolved Rate | Std Error |
|-------|-------|---------------|-----------|
| TBD   | TBD   | TBD           | TBD       |

### Terminal-Bench Adapter Results

| Agent | Model | Resolved Rate | Std Error |
|-------|-------|---------------|-----------|
| oracle| N/A   | 100.00%       | 0.00%     |

## References

- **Original Dataset**: https://huggingface.co/datasets/TheFinAI/en-fpb
- **PIXIU Project**: https://github.com/The-FinAI/PIXIU
- **Terminal-Bench**: https://www.tbench.ai/
- **Adapter Guide**: https://www.tbench.ai/docs/adapters

## Contributing

This adapter was created following the [Terminal-Bench Adapter Guide](https://www.tbench.ai/docs/adapters#26-record-parity-results).

For questions or issues, please open an issue in the Terminal-Bench repository or contact the maintainers.

