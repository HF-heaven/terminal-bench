You are auditing a single-sample evaluation from the PIXIU FinBen benchmark.

Sample metadata:
  • PIXIU instance id: {pixiu_id}
  • Task type: {label_type}
  • Score range: {score_range}

Follow these rules exactly:
  1. Read `/tests/data/item.json` to review the prompt and task details.
  2. Analyze the financial text and determine the appropriate sentiment score.
  3. Write the final score as a floating-point number to `/app/answer.txt`.
  4. The score must be within the specified range.
  5. Do not write any explanation in `answer.txt`; it must contain only the numerical score.
  6. Leave `/tests/data/item.json` untouched so the verifier can re-read it.
