# Setup Instructions for PIXIU FPB Adapter

## Step 1: Request Dataset Access

The `TheFinAI/en-fpb` dataset is **gated** and requires approval before use.

### How to Request Access:

1. **Visit the dataset page**:  
   https://huggingface.co/datasets/TheFinAI/en-fpb

2. **Click "Request access"** button on the page

3. **Fill in the form** with your contact information

4. **Wait for approval** (usually takes a few hours to a few days)

5. **Check your email** for approval notification

## Step 2: Login to HuggingFace CLI

Once you have access, login to HuggingFace:

```bash
# Login with your HuggingFace token
huggingface-cli login

# Or set the token as environment variable
export HF_TOKEN="your_token_here"
```

To get your token:
- Go to https://huggingface.co/settings/tokens
- Create a new token or copy an existing one
- Make sure it has "Read access to contents of all public gated repos you can access"

## Step 3: Verify Access

Test that you can access the dataset:

```bash
conda activate pixiu_env
python3 -c "from datasets import load_dataset; ds = load_dataset('TheFinAI/en-fpb', split='test'); print(f'✅ Success! Loaded {len(ds)} samples')"
```

If successful, you should see:
```
✅ Success! Loaded XXXX samples
```

If you see an error about authentication, go back to Step 2.

## Step 4: Generate Tasks

Once you have access, generate the tasks:

```bash
cd /home/hefan/terminal-bench/adapters/pixiu-fpb

# Activate the environment
conda activate pixiu_env

# Generate 100 test tasks
uv run python run_adapter.py --limit 100

# Or generate 200 tasks
uv run python run_adapter.py --limit 200
```

Expected output:
```
Loading TheFinAI/en-fpb (split=test)...
Loaded 100 samples
Generating FPB tasks: 100%|████████| 100/100 [00:00<00:00, 1234.56it/s]
Generated 100 FPB tasks under /home/hefan/terminal-bench/dataset/pixiu-en-fpb
```

## Step 5: Verify with Oracle Agent

Test that the generated tasks work correctly:

```bash
cd /home/hefan/terminal-bench

# Test 10 tasks
conda activate pixiu_env
uv run tb run \
  --agent oracle \
  --dataset-path dataset/pixiu-en-fpb \
  --n-tasks 10 \
  --n-concurrent 4
```

Expected result:
```
Results Summary:
+-------------------+---------+
| Metric            | Value   |
+===================+=========+
| Resolved Trials   | 10      |
+-------------------+---------+
| Unresolved Trials | 0       |
+-------------------+---------+
| Accuracy          | 100.00% |
+-------------------+---------+
```

## Step 6: Run Full Evaluation

If the test passes, run the full evaluation:

```bash
cd /home/hefan/terminal-bench
conda activate pixiu_env

uv run tb run \
  --agent oracle \
  --dataset-path dataset/pixiu-en-fpb \
  --n-concurrent 8
```

## Troubleshooting

### Error: "Dataset 'TheFinAI/en-fpb' is a gated dataset"

**Solution**: You need to request access (Step 1) and login (Step 2)

### Error: "You must be authenticated to access it"

**Solution**: Run `huggingface-cli login` with your token

### Error: "ModuleNotFoundError: No module named 'datasets'"

**Solution**: Make sure you're in the `pixiu_env` conda environment:
```bash
conda activate pixiu_env
```

### Error: "Command 'uv' not found"

**Solution**: Install uv in the conda environment:
```bash
conda activate pixiu_env
pip install uv
```

## Next Steps

After successfully running the oracle agent:

1. **Update parity_experiment.json** with actual results
2. **Run experiments with other agents** (claude-code, codex, etc.)
3. **Compare results** with original PIXIU benchmark
4. **Update README.md** with parity experiment results
5. **Submit PR** to Terminal-Bench repository

## Questions?

- Check the [Terminal-Bench Adapter Guide](https://www.tbench.ai/docs/adapters)
- Join the `#adapters-spam` channel in Terminal-Bench Discord
- Open an issue in the Terminal-Bench repository

