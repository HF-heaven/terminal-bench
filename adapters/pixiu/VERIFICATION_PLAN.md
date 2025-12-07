# PIXIU Harbor Adapter - Verification Plan

## Overview
This document outlines the complete verification process for the PIXIU Harbor adapter to ensure:
1. Clean generation without cached results
2. 100% oracle pass rate (or documented exceptions)
3. Parity with Terminal-Bench format
4. Production readiness

## Verification Steps

### Step 1: Clean Environment Setup
Remove all cached data and Docker artifacts:

```bash
# 1. Remove all previous PIXIU tasks
rm -rf /tmp/pixiu_* /tmp/test_pixiu_*

# 2. Remove all PIXIU Docker images
docker images | grep -E "(pixiu|test-pixiu)" | awk '{print $3}' | xargs -r docker rmi -f

# 3. Clean adapter cache
cd /home/wendy/terminal-bench/adapters/pixiu
rm -rf __pycache__ *.pyc

# 4. Verify clean state
docker images | grep pixiu  # Should return nothing
ls /tmp/pixiu_*             # Should return "No such file or directory"
```

### Step 2: Dataset Verification
Confirm all accessible PIXIU datasets from HuggingFace:

**Working Datasets (18 total):**
1. `TheFinAI/flare-australian` (test split)
2. `TheFinAI/flare-causal20-sc` (test split)
3. `TheFinAI/flare-cfa` (test split)
4. `TheFinAI/flare-ectsum` (test split)
5. `TheFinAI/flare-edtsum` (test split)
6. `TheFinAI/flare-finqa` (test split)
7. `TheFinAI/flare-finred` (test split)
8. `TheFinAI/flare-fomc` (test split)
9. `TheFinAI/en-fpb` (test split)
10. `TheFinAI/flare-german` (test split)
11. `TheFinAI/flare-headlines` (test split)
12. `TheFinAI/flare-ner` (test split)
13. `TheFinAI/flare-tatqa` (test split)
14. `TheFinAI/flare-tsa` (test split)
15. `TheFinAI/flare-fiqasa` (test split)
16. `TheFinAI/flare-fnxl` (test split)
17. `TheFinAI/flare-fsrl` (test split)
18. `TheFinAI/en-forecasting-travelinsurance` (test split)

**Known Unavailable Datasets:**
- `TheFinAI/flare-ccf` - Does not exist on Hub
- `TheFinAI/flare-ccfraud` - Does not exist on Hub
- `TheFinAI/flare-convfinqa` - KeyError: 'text' (data format issue)
- `TheFinAI/flare-fpb` - Gated dataset (requires access request)
- Several others not in official PIXIU documentation

### Step 3: Generate Test Sample (5 tasks per dataset)
```bash
cd /home/wendy/terminal-bench/adapters/pixiu
export HF_TOKEN="<your_token>"

# Use the clean generation script
./scripts/verify_clean_generation.sh
```

Expected output:
- 90 tasks total (18 datasets × 5 samples)
- All tasks have proper Harbor structure
- No generation errors

### Step 4: Validate Task Structure
Each generated task should contain:

```
pixiu-{dataset}-{id}/
├── task.toml              # Harbor TOML config
├── instruction.md         # Task-specific instruction
├── Dockerfile             # Main container
├── environment/
│   └── Dockerfile         # Base environment
├── solution/
│   └── solve.sh          # Oracle solution
└── tests/
    ├── test_outputs.py   # pytest test
    └── data/
        └── item.json     # Test data
```

Validation checks:
- [ ] `task.toml` has correct version and metadata
- [ ] `instruction.md` contains task-specific content (not generic)
- [ ] `solution/solve.sh` produces correct answer format
- [ ] `tests/test_outputs.py` validates output correctly
- [ ] No hardcoded paths or references to `/tmp`

### Step 5: Oracle Testing (Harbor Format)
Since Harbor tasks use `.toml` config, they must be tested differently than Terminal-Bench's `.yaml` format.

**Option A: Manual Docker Testing (Slow but Thorough)**
```bash
cd /home/wendy/terminal-bench/adapters/pixiu
./scripts/verify_harbor_oracle.sh
```

**Option B: Harbor CLI (Requires Harbor installation)**
```bash
# Install Harbor CLI if not already installed
uv pip install harbor-cli

# Run oracle tests
uv run harbor jobs start -p /tmp/pixiu_harbor_test -a oracle
```

**Expected Results:**
- 90/90 tasks pass (100%)
- Or documented exceptions with explanations

### Step 6: Terminal-Bench Format Parity Test
Generate same tasks in Terminal-Bench format and verify oracle pass rate:

```bash
cd /home/wendy/terminal-bench/adapters/pixiu
./scripts/verify_tb_parity.sh
```

Expected: 100% pass rate for Terminal-Bench format

### Step 7: Known Issues Documentation
Document any tasks that fail consistently with oracle agent:

| Task ID | Dataset | Failure Type | Reason |
|---------|---------|--------------|--------|
| (none expected) | - | - | All should pass |

If failures occur:
1. Verify it's not a caching issue (re-run from clean state)
2. Check if it's a dataset issue (test with different samples)
3. Document as known issue if reproducible

### Step 8: Production Generation
After verification passes, generate full production dataset:

```bash
# Generate 100 samples per dataset (1,800 total tasks)
export HF_TOKEN="<your_token>"
cd /home/wendy/terminal-bench/adapters/pixiu

./scripts/generate_production.sh \
  --limit 100 \
  --output-path ../../datasets/pixiu
```

## Success Criteria

✅ **Clean Generation:** No errors during generation from scratch  
✅ **Structure Valid:** All tasks have correct Harbor format  
✅ **Oracle Pass Rate:** 100% (or documented exceptions)  
✅ **Parity:** Terminal-Bench format also passes at 100%  
✅ **Reproducible:** Multiple clean runs produce identical results  
✅ **Documentation:** README, known issues, and usage examples complete  

## Troubleshooting

### Issue: Docker build failures
**Cause:** Missing base images or network issues  
**Solution:** Pre-pull base images: `docker pull python:3.11-slim`

### Issue: HuggingFace auth errors
**Cause:** Invalid or expired HF_TOKEN  
**Solution:** Generate new token from https://huggingface.co/settings/tokens

### Issue: Task generation hangs
**Cause:** Dataset download timeout  
**Solution:** Increase timeout or use HF cache: `export HF_HOME=/path/to/cache`

### Issue: Oracle tests fail
**Cause:** Cached results from previous runs  
**Solution:** Run Step 1 (Clean Environment Setup) again

## Timeline

| Step | Estimated Time | Status |
|------|---------------|--------|
| Clean environment | 5 minutes | ⏳ Pending |
| Dataset verification | 10 minutes | ⏳ Pending |
| Generate test sample | 15 minutes | ⏳ Pending |
| Validate structure | 10 minutes | ⏳ Pending |
| Oracle testing | 30-60 minutes | ⏳ Pending |
| TB parity test | 30 minutes | ⏳ Pending |
| Document issues | 15 minutes | ⏳ Pending |
| Production generation | 2-3 hours | ⏳ Pending |

**Total:** ~4-5 hours for complete verification

## Next Steps After Verification

1. Create comprehensive README.md (similar to SWEBench)
2. Add adapter to Harbor registry
3. Submit to harbor-datasets repository
4. Update PIXIU documentation
5. Run full-scale evaluation with agents
