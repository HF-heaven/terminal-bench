#!/bin/bash
set -euo pipefail

cd /app

# Run pytest tests
pytest /tests/test_outputs.py -rA -s

# Write reward based on exit code
if [ $? -eq 0 ]; then
    echo "1" > /logs/verifier/reward.txt
else
    echo "0" > /logs/verifier/reward.txt
fi
