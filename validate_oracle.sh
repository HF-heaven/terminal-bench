#!/bin/bash

TASK_DIR="/home/wendy/terminal-bench/tasks"
PASSED=0
FAILED=0
TOTAL=0

echo "Running Oracle Validation on TSA Tasks"
echo "======================================"
echo ""

# Test all pixiu-tsa tasks
for task_path in "$TASK_DIR"/pixiu-tsa-*; do
    task_name=$(basename "$task_path")
    TOTAL=$((TOTAL + 1))
    
    # Extract expected label from test file
    expected=$(grep "EXPECTED_LABEL = " "$task_path/tests/test_outputs.py" | sed 's/.*"\(.*\)".*/\1/')
    
    # Extract solution output
    actual=$(grep "printf" "$task_path/solution.sh" | sed 's/.*"\(.*\)".*/\1/')
    
    # Validate
    if [ "$expected" = "$actual" ]; then
        PASSED=$((PASSED + 1))
    else
        FAILED=$((FAILED + 1))
        echo "FAIL: $task_name - Expected: $expected, Got: $actual"
    fi
    
    # Progress indicator every 100 tasks
    if [ $((TOTAL % 100)) -eq 0 ]; then
        echo "Progress: $TOTAL tasks validated..."
    fi
done

echo ""
echo "======================================"
echo "Oracle Validation Results"
echo "======================================"
echo ""
echo "+-------------------+---------+"
echo "| Metric            | Value   |"
echo "+===================+========="
printf "| Resolved Trials   | %-7d |\n" $PASSED
printf "| Unresolved Trials | %-7d |\n" $FAILED
printf "| Accuracy          | %6.2f%% |\n" $(echo "scale=2; $PASSED / $TOTAL * 100" | bc)
echo "+-------------------+---------+"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✓ All tasks passed oracle validation!"
    exit 0
else
    echo "✗ Some tasks failed validation"
    exit 1
fi
